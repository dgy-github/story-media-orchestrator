"""Automated keyframe extraction, sharpness scoring, and 2x2 diagnostic contact sheets."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from io import BytesIO
import math
from pathlib import Path
import os
import re
import shutil
from tempfile import TemporaryDirectory
from typing import Any

from .manifest import ProjectManifest, Shot
from .registry import ArtifactRegistry

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_SCENE_CUT_RE = re.compile(
    r"(?:切换场景|次日|第二天|多年后|数年后|转场|换场景|next\s+day|meanwhile|scene\s+cut)",
    re.IGNORECASE,
)
_BEAT_SPLIT_RE = re.compile(r"(?:\s*(?:然后|随后|接着|最后|继而|与此同时|->|；|;)\s*)+")


def _load_font(size: int):
    from PIL import ImageFont

    candidates = (
        os.environ.get("STORY_MEDIA_FONT", ""),
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for path in candidates:
        if path and Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size=size)


def _frame_metrics_from_pil(pil_img) -> tuple[float, float, Any]:
    """Return (sharpness, mean_luma, small_gray_thumb) for a PIL RGB image."""
    import cv2
    import numpy as np

    rgb = pil_img.convert("RGB")
    arr = np.asarray(rgb)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_luma = float(gray.mean())
    thumb = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
    return round(sharpness, 2), round(mean_luma, 2), thumb


def _center_border_laplacian(pil_img) -> tuple[float, float, Any]:
    import cv2
    import numpy as np

    gray_128 = cv2.resize(
        np.asarray(pil_img.convert("L"), dtype=np.uint8),
        (128, 128),
        interpolation=cv2.INTER_AREA,
    )
    lap = cv2.Laplacian(gray_128, cv2.CV_64F)
    center_mask = np.zeros((128, 128), dtype=bool)
    center_mask[32:96, 32:96] = True
    center_sharp = float(lap[center_mask].var())
    border_sharp = float(lap[~center_mask].var())
    return center_sharp, border_sharp, gray_128


def _estimate_rigid_pan_and_tear(prev_gray_128, curr_gray_128, anchor_gray_128) -> dict[str, Any]:
    import cv2
    import numpy as np

    prev_f = prev_gray_128.astype(np.float32)
    curr_f = curr_gray_128.astype(np.float32)
    win = cv2.createHanningWindow((128, 128), cv2.CV_32F)
    (dx, dy), corr_resp = cv2.phaseCorrelate(prev_f * win, curr_f * win)
    pan_mag = math.hypot(float(dx), float(dy))

    warp_mat = np.float32([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]])
    warped_prev = cv2.warpAffine(
        prev_gray_128,
        warp_mat,
        (128, 128),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    raw_center_mad = float(
        np.abs(curr_gray_128[32:96, 32:96].astype(np.float32) - prev_gray_128[32:96, 32:96].astype(np.float32)).mean()
    )
    nonrigid_center_mad = float(
        np.abs(curr_gray_128[32:96, 32:96].astype(np.float32) - warped_prev[32:96, 32:96].astype(np.float32)).mean()
    )

    anchor_edges = cv2.Canny(anchor_gray_128[32:96, 32:96], 60, 160)
    prev_edges = cv2.Canny(prev_gray_128[32:96, 32:96], 60, 160)
    curr_edges = cv2.Canny(curr_gray_128[32:96, 32:96], 60, 160)
    anchor_comp = max(0, int(cv2.connectedComponents(anchor_edges)[0]) - 1)
    prev_comp = max(0, int(cv2.connectedComponents(prev_edges)[0]) - 1)
    curr_comp = max(0, int(cv2.connectedComponents(curr_edges)[0]) - 1)
    edge_component_delta = curr_comp - max(anchor_comp, prev_comp)

    return {
        "dx": float(dx),
        "dy": float(dy),
        "pan_mag": pan_mag,
        "corr_resp": float(corr_resp),
        "raw_center_mad": raw_center_mad,
        "nonrigid_center_mad": nonrigid_center_mad,
        "anchor_components": anchor_comp,
        "prev_components": prev_comp,
        "curr_components": curr_comp,
        "edge_component_delta": edge_component_delta,
    }


def _color_lighting_stats(pil_img) -> dict[str, float]:
    """Compute lighting, saturation, warmth, and LAB chromaticity statistics."""
    import cv2
    import numpy as np

    rgb = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
    small = cv2.resize(rgb, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    mean_r = float(small[:, :, 0].mean())
    mean_b = float(small[:, :, 2].mean())
    warmth_rg_delta = mean_r - mean_b
    mean_saturation = float(hsv[:, :, 1].mean())
    lab_l = float(lab[:, :, 0].mean())
    lab_a = float(lab[:, :, 1].mean())
    lab_b = float(lab[:, :, 2].mean())
    dark_clip_ratio = float((gray <= 12).mean())
    overexposed_ratio = float((gray >= 245).mean())

    return {
        "mean_saturation": round(mean_saturation, 2),
        "warmth_rg_delta": round(warmth_rg_delta, 2),
        "lab_l": round(lab_l, 2),
        "lab_a": round(lab_a, 2),
        "lab_b": round(lab_b, 2),
        "dark_clip_ratio": round(dark_clip_ratio, 4),
        "overexposed_ratio": round(overexposed_ratio, 4),
    }


def _vfx_frame_metrics(pil_img) -> dict[str, Any]:
    """Compute objective VFX texture, neon saturation clipping, and glow mask stats."""
    import cv2
    import numpy as np

    rgb = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
    small = cv2.resize(rgb, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Extreme neon saturation + bright value indicates plastic 2D paint-stroke energy bars.
    neon_mask = (sat >= 215) & (val >= 200)
    glow_mask = (val >= 220) & ((sat >= 120) | (val >= 242))

    neon_clip_ratio = float(neon_mask.mean())
    glow_area_ratio = float(glow_mask.mean())

    # Micro-detail inside glowing interior (eroded 1px to exclude outer silhouette step) distinguishes volumetric sparks from flat blobs.
    interior_mask = cv2.erode(glow_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) > 0
    eval_mask = interior_mask if int(interior_mask.sum()) >= 12 else glow_mask
    if int(eval_mask.sum()) >= 12:
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        glow_micro_detail = float(lap[eval_mask].var())
    else:
        glow_micro_detail = 0.0

    return {
        "neon_clip_ratio": round(neon_clip_ratio, 4),
        "glow_area_ratio": round(glow_area_ratio, 4),
        "glow_micro_detail": round(glow_micro_detail, 2),
        "glow_mask": glow_mask,
        "gray_small": gray,
    }


def compute_active_motion_window(
    video_path: str | Path | None,
    target_duration_s: float,
    *,
    prefer_early_start: bool = True,
    max_samples: int = 24,
) -> dict[str, Any]:
    """Find the highest-activity contiguous time window of length `target_duration_s` to cut H3 tail stalls."""
    if not video_path or target_duration_s <= 0:
        return {"available": False, "reason": "invalid_input"}
    path = Path(video_path)
    if not path.is_file():
        return {"available": False, "reason": "file_not_found"}

    import cv2
    import numpy as np

    with _open_video_capture(path) as cap:
        if not cap.isOpened():
            return {"available": False, "reason": "cannot_open_video"}
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        if fps <= 0.0:
            fps = 24.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 2:
            return {"available": False, "reason": "too_few_frames"}
        total_dur = frame_count / fps
        if total_dur <= target_duration_s + 0.12:
            return {
                "available": True,
                "start_s": 0.0,
                "end_s": round(total_dur, 3),
                "duration_s": round(total_dur, 3),
                "total_duration_s": round(total_dur, 3),
                "trimmed": False,
            }

        n_pts = max(8, min(int(max_samples), frame_count))
        indices = sorted(
            set(int(round(i * (frame_count - 1) / max(1, n_pts - 1))) for i in range(n_pts))
        )
        times: list[float] = []
        thumbs: list[Any] = []
        for idx in indices:
            pil_img = _read_video_frame_rgb(cap, idx)
            if pil_img is None:
                continue
            _, _, thumb = _frame_metrics_from_pil(pil_img)
            times.append(idx / fps)
            thumbs.append(thumb)

        if len(thumbs) < 3:
            return {
                "available": True,
                "start_s": 0.0,
                "end_s": round(min(total_dur, target_duration_s), 3),
                "duration_s": round(min(total_dur, target_duration_s), 3),
                "total_duration_s": round(total_dur, 3),
                "trimmed": True,
            }

        # Weight center-region motion 70% + full motion 30% so hero displacement drives window selection.
        seg_mids: list[float] = []
        seg_scores: list[float] = []
        th_h, th_w = thumbs[0].shape[:2]
        cy0, cy1 = int(round(th_h * 0.20)), int(round(th_h * 0.80))
        cx0, cx1 = int(round(th_w * 0.20)), int(round(th_w * 0.80))
        for i in range(len(thumbs) - 1):
            delta = np.abs(thumbs[i + 1].astype(np.float32) - thumbs[i].astype(np.float32))
            full_d = float(delta.mean())
            center_d = float(delta[cy0:cy1, cx0:cx1].mean())
            seg_mids.append(0.5 * (times[i] + times[i + 1]))
            seg_scores.append(0.70 * center_d + 0.30 * full_d)

        max_start = max(0.0, total_dur - target_duration_s)
        # Candidate starts: 0.0 plus dense grid up to max_start.
        candidates = [0.0]
        steps = 16
        for k in range(1, steps + 1):
            candidates.append(max_start * k / steps)

        best_start = 0.0
        best_val = -1.0
        for st in candidates:
            en = st + target_duration_s
            covered = [sc for mid, sc in zip(seg_mids, seg_scores) if st <= mid <= en]
            if not covered:
                continue
            mean_act = sum(covered) / len(covered)
            # Small bias toward keeping the opening of the action (drop trailing stall first).
            early_bonus = (1.0 - 0.12 * (st / max(0.1, max_start))) if prefer_early_start else 1.0
            score = mean_act * early_bonus
            if score > best_val:
                best_val = score
                best_start = st

        best_end = min(total_dur, best_start + target_duration_s)
        return {
            "available": True,
            "start_s": round(best_start, 3),
            "end_s": round(best_end, 3),
            "duration_s": round(best_end - best_start, 3),
            "total_duration_s": round(total_dur, 3),
            "trimmed": True,
            "activity_score": round(best_val, 3),
        }


_FACE_CASCADE_CACHE: Any = False


def _get_face_cascade():
    global _FACE_CASCADE_CACHE
    if _FACE_CASCADE_CACHE is not False:
        return _FACE_CASCADE_CACHE
    try:
        import cv2

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if cascade_path.is_file():
            clf = cv2.CascadeClassifier(str(cascade_path))
            _FACE_CASCADE_CACHE = clf if not clf.empty() else None
        else:
            _FACE_CASCADE_CACHE = None
    except Exception:
        _FACE_CASCADE_CACHE = None
    return _FACE_CASCADE_CACHE


def _detect_focus_roi_box(pil_img) -> tuple[tuple[int, int, int, int], str]:
    """Detect a face/portrait ROI box (x0, y0, x1, y1) for expression/mouth/hand inspection."""
    import cv2
    import numpy as np

    rgb = pil_img.convert("RGB")
    w, h = rgb.size
    if w < 32 or h < 32:
        return (0, 0, w, h), "full"

    gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)

    # 1) Try frontal face detection first (works well on 3D/semi-realistic anime portraits).
    clf = _get_face_cascade()
    if clf is not None:
        try:
            min_dim = max(24, int(min(w, h) * 0.10))
            faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(min_dim, min_dim))
            if len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda b: int(b[2]) * int(b[3]))
                pad_x = int(round(fw * 0.45))
                pad_top = int(round(fh * 0.35))
                pad_bot = int(round(fh * 0.55))
                x0 = max(0, int(fx) - pad_x)
                y0 = max(0, int(fy) - pad_top)
                x1 = min(w, int(fx + fw) + pad_x)
                y1 = min(h, int(fy + fh) + pad_bot)
                if (x1 - x0) >= 24 and (y1 - y0) >= 24:
                    return (x0, y0, x1, y1), "face"
        except Exception:
            pass

    # 2) Anime-friendly fallback: peak edge-energy centroid in the portrait upper-center zone.
    edges = cv2.Canny(gray, 50, 150).astype(np.float32)
    zy0, zy1 = int(h * 0.06), max(int(h * 0.06) + 1, int(h * 0.72))
    zx0, zx1 = int(w * 0.14), max(int(w * 0.14) + 1, int(w * 0.86))
    zone = edges[zy0:zy1, zx0:zx1]
    crop_w = max(48, int(round(w * 0.44)))
    crop_h = max(48, int(round(h * 0.52)))
    if float(zone.sum()) > 10.0:
        ys, xs = np.nonzero(zone > 0)
        cx = int(zx0 + float(xs.mean()))
        cy = int(zy0 + float(ys.mean()))
        source = "salient_focus"
    else:
        cx = w // 2
        cy = int(round(h * 0.38))
        source = "upper_center"

    x0 = max(0, min(w - crop_w, cx - crop_w // 2))
    y0 = max(0, min(h - crop_h, cy - crop_h // 2))
    return (x0, y0, min(w, x0 + crop_w), min(h, y0 + crop_h)), source


def _build_roi_zoom_strip(
    shot_id: str,
    panels: list[tuple[int, float, float, Any]],
    *,
    color_drift: float,
    luma_jump: float,
) -> tuple[Any, str]:
    """Build a 4-panel horizontal detail zoom strip focused on the character face/subject ROI."""
    from PIL import Image, ImageDraw

    if not panels:
        return None, "none"

    box, roi_source = _detect_focus_roi_box(panels[0][3])
    cell_w, cell_h = 260, 260
    header_h = 32
    pad = 6
    cols = min(4, len(panels))
    canvas_w = cols * cell_w + (cols + 1) * pad
    canvas_h = header_h + cell_h + 2 * pad

    strip = Image.new("RGB", (canvas_w, canvas_h), "#0f172a")
    draw = ImageDraw.Draw(strip)
    header_font = _load_font(13)
    badge_font = _load_font(12)

    summary = (
        f"{shot_id} ROI ZOOM ({roi_source}) | color_drift={color_drift:.1f} | luma_jump={luma_jump:.1f}"
    )
    draw.rectangle((0, 0, canvas_w, header_h), fill="#1e293b")
    draw.text((pad, 7), summary, font=header_font, fill="#f8fafc")

    x0_b, y0_b, x1_b, y1_b = box
    for slot in range(cols):
        f_idx, ts_s, sharp, pil_img = panels[slot]
        w, h = pil_img.size
        crop = pil_img.crop((max(0, min(w - 1, x0_b)), max(0, min(h - 1, y0_b)), max(1, min(w, x1_b)), max(1, min(h, y1_b))))
        resized = crop.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        px = pad + slot * (cell_w + pad)
        py = header_h + pad
        strip.paste(resized, (px, py))
        draw.rectangle((px, py, px + cell_w, py + 20), fill="#020617")
        draw.text(
            (px + 5, py + 2),
            f"#{slot + 1} f={f_idx} {ts_s:.2f}s s={sharp:.0f}",
            font=badge_font,
            fill="#fbbf24",
        )
    return strip, roi_source


def extract_image_diagnostics(image_path: str | Path | None) -> dict[str, Any]:
    """Compute resolution, sharpness, lighting, and dark-frame diagnostics on an image file."""
    if not image_path:
        return {"available": False, "reason": "missing_path"}
    path = Path(image_path)
    if not path.is_file():
        return {"available": False, "reason": "file_not_found", "path": str(path)}
    if path.suffix.lower() not in _IMAGE_SUFFIXES:
        return {"available": False, "reason": f"unsupported_image_suffix:{path.suffix}", "path": str(path)}

    from PIL import Image

    with Image.open(path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        sharpness, mean_luma, _ = _frame_metrics_from_pil(rgb)
        color_stats = _color_lighting_stats(rgb)

    return {
        "schema": "image-diagnostics/v1",
        "available": True,
        "path": str(path),
        "width": int(width),
        "height": int(height),
        "aspect_ratio": round(width / max(1, height), 4),
        "sharpness": sharpness,
        "mean_luma": mean_luma,
        "is_blank_or_dark": bool(mean_luma < 8.0),
        "mean_saturation": color_stats["mean_saturation"],
        "warmth_rg_delta": color_stats["warmth_rg_delta"],
        "dark_clip_ratio": color_stats["dark_clip_ratio"],
        "overexposed_ratio": color_stats["overexposed_ratio"],
    }


@contextmanager
def _open_video_capture(video_path: Path):
    import cv2

    cap = cv2.VideoCapture(str(video_path.resolve()))
    temp_dir: TemporaryDirectory | None = None
    try:
        if not cap.isOpened() and video_path.is_file():
            # Fallback for non-ASCII paths on Windows VideoCapture builds.
            cap.release()
            temp_dir = TemporaryDirectory(prefix="vidcap-")
            temp_copy = Path(temp_dir.name) / f"input{video_path.suffix or '.mp4'}"
            shutil.copy2(video_path, temp_copy)
            cap = cv2.VideoCapture(str(temp_copy))
        yield cap
    finally:
        cap.release()
        if temp_dir is not None:
            temp_dir.cleanup()


def _read_video_frame_rgb(cap, frame_idx: int):
    import cv2
    from PIL import Image

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
    ok, bgr = cap.read()
    if not ok or bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _pick_best_frame_in_window(cap, target_idx: int, total_frames: int) -> tuple[int, Any, float, float, Any] | None:
    best: tuple[tuple[int, float], int, Any, float, float, Any] | None = None
    low = max(0, target_idx - 2)
    high = min(max(0, total_frames - 1), target_idx + 2)
    candidates = [target_idx] + [i for i in range(low, high + 1) if i != target_idx]
    for idx in candidates:
        pil_img = _read_video_frame_rgb(cap, idx)
        if pil_img is None:
            continue
        sharp, luma, thumb = _frame_metrics_from_pil(pil_img)
        non_black = 1 if luma >= 8.0 else 0
        # Prefer non-black frames first, then sharper frames, then closer to target.
        rank = (non_black, round(sharp - abs(idx - target_idx) * 0.5, 4))
        if best is None or rank > best[0]:
            best = (rank, idx, pil_img, sharp, luma, thumb)
        if non_black == 1 and idx == target_idx and sharp > 25.0:
            break
    if best is None:
        return None
    _, idx, pil_img, sharp, luma, thumb = best
    return idx, pil_img, sharp, luma, thumb


def _build_dense_strip(
    shot_id: str,
    dense_panels: list[tuple[int, float, float, float, Any]],
    *,
    width: int,
    height: int,
    duration: float,
    freeze_ratio: float,
    flicker_jump: float,
):
    from PIL import Image, ImageDraw

    if not dense_panels:
        return None

    first_img = dense_panels[0][4]
    src_w, src_h = first_img.size
    cell_w = min(360, max(160, src_w))
    cell_h = max(90, int(round(cell_w * (src_h / max(1, src_w)))))
    header_h = 34
    pad = 6
    cols, rows = 4, 2
    canvas_w = cols * cell_w + (cols + 1) * pad
    canvas_h = header_h + rows * cell_h + (rows + 1) * pad

    sheet = Image.new("RGB", (canvas_w, canvas_h), "#0f172a")
    draw = ImageDraw.Draw(sheet)
    header_font = _load_font(14)
    badge_font = _load_font(12)

    summary = (
        f"{shot_id} DENSE 8x | {width}x{height} | dur={duration:.2f}s | "
        f"freeze={freeze_ratio:.2f} | max_luma_jump={flicker_jump:.1f}"
    )
    draw.rectangle((0, 0, canvas_w, header_h), fill="#1e293b")
    draw.text((pad, 8), summary, font=header_font, fill="#f8fafc")

    for slot in range(cols * rows):
        row = slot // cols
        col = slot % cols
        x0 = pad + col * (cell_w + pad)
        y0 = header_h + pad + row * (cell_h + pad)
        if slot < len(dense_panels):
            f_idx, ts_s, sharp, step_diff, pil_img = dense_panels[slot]
            resized = pil_img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            sheet.paste(resized, (x0, y0))
            is_frozen = slot > 0 and step_diff < 0.40
            draw.rectangle((x0, y0, x0 + cell_w, y0 + 20), fill="#3b0764" if is_frozen else "#020617")
            label = f"#{slot + 1} f={f_idx} {ts_s:.2f}s s={sharp:.0f} d={step_diff:.1f}"
            if is_frozen:
                label += " !STATIC"
            draw.text(
                (x0 + 5, y0 + 2),
                label,
                font=badge_font,
                fill="#e9d5ff" if is_frozen else "#38bdf8",
            )
        else:
            draw.rectangle((x0, y0, x0 + cell_w, y0 + cell_h), fill="#1e293b")
    return sheet


def _build_contact_sheet(
    shot_id: str,
    panels: list[tuple[int, float, float, Any]],
    *,
    width: int,
    height: int,
    duration: float,
    sharp_avg: float,
    motion_mean_diff: float,
    suspect_slot: int | None = None,
):
    from PIL import Image, ImageDraw

    if not panels:
        raise ValueError("at least one video frame is required to build a contact sheet")

    first_img = panels[0][3]
    src_w, src_h = first_img.size
    cell_w = min(480, max(160, src_w))
    cell_h = max(90, int(round(cell_w * (src_h / max(1, src_w)))))
    header_h = 34
    pad = 8
    cols, rows = 2, 2
    canvas_w = cols * cell_w + (cols + 1) * pad
    canvas_h = header_h + rows * cell_h + (rows + 1) * pad

    sheet = Image.new("RGB", (canvas_w, canvas_h), "#0f172a")
    draw = ImageDraw.Draw(sheet)
    header_font = _load_font(15)
    badge_font = _load_font(13)

    summary = (
        f"{shot_id} | {width}x{height} | dur={duration:.2f}s | "
        f"sharp={sharp_avg:.1f} | motion={motion_mean_diff:.2f}"
    )
    draw.rectangle((0, 0, canvas_w, header_h), fill="#1e293b")
    draw.text((pad, 7), summary, font=header_font, fill="#f8fafc")

    for slot in range(4):
        row = slot // 2
        col = slot % 2
        x0 = pad + col * (cell_w + pad)
        y0 = header_h + pad + row * (cell_h + pad)
        if slot < len(panels):
            panel_idx, timestamp_s, sharp, pil_img = panels[slot]
            resized = pil_img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            sheet.paste(resized, (x0, y0))
            is_suspect = suspect_slot == slot
            draw.rectangle((x0, y0, x0 + cell_w, y0 + 22), fill="#450a0a" if is_suspect else "#020617")
            label = f"#{slot + 1} f={panel_idx} t={timestamp_s:.2f}s sharp={sharp:.1f}"
            if is_suspect:
                label += " !CLIP?"
            draw.text(
                (x0 + 6, y0 + 3),
                label,
                font=badge_font,
                fill="#fca5a5" if is_suspect else "#38bdf8",
            )
            if is_suspect:
                draw.rectangle((x0, y0, x0 + cell_w - 1, y0 + cell_h - 1), outline="#ef4444", width=3)
        else:
            draw.rectangle((x0, y0, x0 + cell_w, y0 + cell_h), fill="#1e293b")

    return sheet


def extract_video_diagnostics(
    video_path: str | Path | None,
    *,
    registry: ArtifactRegistry | None = None,
    output_dir: str | Path | None = None,
    shot_id: str = "shot",
    sample_ratios: tuple[float, ...] = (0.06, 0.36, 0.66, 0.92),
) -> dict[str, Any]:
    """Extract keyframes, sharpness/motion telemetry, tail frame, and a 2x2 contact sheet."""
    if not video_path:
        return {"available": False, "reason": "missing_path"}
    path = Path(video_path)
    if not path.is_file():
        return {"available": False, "reason": "file_not_found", "path": str(path)}

    import cv2
    import numpy as np

    with _open_video_capture(path) as cap:
        if not cap.isOpened():
            return {"available": False, "reason": "cannot_open_video", "path": str(path)}
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0.0:
            fps = 24.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            # Probe sequentially for codecs without frame-count header.
            probed = 0
            while cap.grab():
                probed += 1
            frame_count = probed
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if frame_count <= 0:
            return {"available": False, "reason": "zero_frames", "path": str(path)}

        duration_seconds = round(frame_count / fps, 3)
        samples: list[dict[str, Any]] = []
        panels: list[tuple[int, float, float, Any]] = []
        thumbs: list[Any] = []

        for ratio in sample_ratios[:4]:
            clamped = max(0.0, min(0.99, float(ratio)))
            target_idx = min(frame_count - 1, max(0, int(round(clamped * (frame_count - 1)))))
            picked = _pick_best_frame_in_window(cap, target_idx, frame_count)
            if picked is None:
                continue
            picked_idx, pil_img, sharp, luma, thumb = picked
            if width <= 0 or height <= 0:
                width, height = pil_img.size
            timestamp_s = round(picked_idx / fps, 3)
            panels.append((picked_idx, timestamp_s, sharp, pil_img))
            thumbs.append(thumb)
            samples.append(
                {
                    "ratio": clamped,
                    "frame_index": picked_idx,
                    "timestamp_seconds": timestamp_s,
                    "sharpness": sharp,
                    "mean_luma": luma,
                    "is_blank_or_dark": bool(luma < 8.0),
                }
            )

        if not panels:
            return {"available": False, "reason": "no_decodable_frames", "path": str(path)}

        is_clipping_suspect = False
        clipping_mode: str | None = None
        clipping_suspect_frame: dict[str, Any] | None = None
        suspect_slot: int | None = None
        motion_blur_likely = False
        pan_shift: list[float] = [0.0, 0.0]

        if frame_count >= 8 and len(panels) >= 2:
            anchor_sharp = max(1.0, 0.5 * (float(panels[0][2]) + float(panels[-1][2])))
            anchor_center_sharp, anchor_border_sharp, anchor_gray_128 = _center_border_laplacian(panels[0][3])
            scan_start = max(0, int(round(0.20 * (frame_count - 1))))
            scan_end = max(scan_start, min(frame_count - 1, int(round(0.80 * (frame_count - 1)))))
            n_scan = min(10, scan_end - scan_start + 1)
            raw_indices = [
                int(round(scan_start + step_i * (scan_end - scan_start) / max(1, n_scan - 1)))
                for step_i in range(n_scan)
            ]
            scan_indices = list(dict.fromkeys(raw_indices))
            prev_scan_thumb = thumbs[0]
            prev_gray_128 = anchor_gray_128
            local_motion_max = 0.0
            worst_blur: dict[str, Any] | None = None
            worst_tear: dict[str, Any] | None = None
            pan_samples: list[tuple[float, float, float, float, float]] = []

            for scan_idx in scan_indices:
                pil_scan = _read_video_frame_rgb(cap, scan_idx)
                if pil_scan is None:
                    continue
                s_sharp, s_luma, s_thumb = _frame_metrics_from_pil(pil_scan)
                local_motion = float(
                    np.abs(s_thumb.astype(np.float32) - prev_scan_thumb.astype(np.float32)).mean()
                )
                prev_scan_thumb = s_thumb
                local_motion_max = max(local_motion_max, local_motion)

                c_sharp, b_sharp, curr_gray_128 = _center_border_laplacian(pil_scan)
                motion_stats = _estimate_rigid_pan_and_tear(prev_gray_128, curr_gray_128, anchor_gray_128)
                prev_gray_128 = curr_gray_128
                if s_luma < 8.0:
                    continue

                retain_ratio = s_sharp / anchor_sharp
                c_retain = c_sharp / max(1.0, anchor_center_sharp)
                b_retain = b_sharp / max(1.0, anchor_border_sharp)
                pan_samples.append(
                    (
                        motion_stats["dx"],
                        motion_stats["dy"],
                        motion_stats["pan_mag"],
                        motion_stats["corr_resp"],
                        b_retain,
                    )
                )

                anomaly = max(0.0, 1.0 - retain_ratio) * (1.0 + min(1.5, local_motion / 12.0))
                blur_rank = (round(anomaly, 5), -s_sharp)
                candidate = {
                    "idx": scan_idx,
                    "ts": round(scan_idx / fps, 3),
                    "ratio": round(scan_idx / max(1, frame_count - 1), 3),
                    "sharp": s_sharp,
                    "luma": s_luma,
                    "pil_img": pil_scan,
                    "thumb": s_thumb,
                    "retain_ratio": retain_ratio,
                    "center_retain": c_retain,
                    "border_retain": b_retain,
                    "local_motion": local_motion,
                    "anomaly": anomaly,
                    **motion_stats,
                }
                if worst_blur is None or blur_rank > worst_blur["rank"]:
                    worst_blur = {**candidate, "rank": blur_rank}

                # Sharp topology tear: crisp edges (not global blur), rigid-pan residual high, fragmented new islands in center
                is_tear_candidate = bool(
                    anchor_sharp >= 8.0
                    and retain_ratio >= 0.48
                    and motion_stats["nonrigid_center_mad"] >= 9.0
                    and (
                        motion_stats["edge_component_delta"] >= 3
                        or (
                            motion_stats["curr_components"] >= max(3, int(round(motion_stats["anchor_components"] * 1.6)) + 2)
                            and motion_stats["nonrigid_center_mad"] >= 14.0
                        )
                    )
                )
                if is_tear_candidate:
                    tear_score = motion_stats["nonrigid_center_mad"] * (1.0 + 0.25 * max(0, motion_stats["edge_component_delta"]))
                    tear_rank = (round(tear_score, 4), motion_stats["edge_component_delta"])
                    if worst_tear is None or tear_rank > worst_tear["rank"]:
                        worst_tear = {**candidate, "rank": tear_rank, "tear_score": tear_score}

            if worst_blur is not None:
                soft_mush_hit = bool(
                    anchor_sharp >= 8.0
                    and worst_blur["retain_ratio"] < 0.55
                    and (local_motion_max >= 3.0 or worst_blur["local_motion"] >= 2.5)
                )
                # Rigid camera pan with uniform center/border motion blur is camera blur, not character clipping
                strong_pan = bool(
                    worst_blur["pan_mag"] >= 2.0
                    and worst_blur["corr_resp"] >= 0.18
                    and worst_blur["raw_center_mad"] > 1.0
                    and worst_blur["nonrigid_center_mad"] < 0.78 * worst_blur["raw_center_mad"]
                )
                uniform_blur = bool(
                    worst_blur["border_retain"] < 0.78
                    and worst_blur["center_retain"] >= worst_blur["border_retain"] * 0.65
                )
                if worst_tear is None and strong_pan and (uniform_blur or soft_mush_hit):
                    motion_blur_likely = True

                if pan_samples:
                    best_pan = max(pan_samples, key=lambda item: item[2] * max(0.0, item[3]))
                    pan_shift = [round(best_pan[0], 2), round(best_pan[1], 2)]

                chosen: dict[str, Any] | None = None
                if worst_tear is not None:
                    is_clipping_suspect = True
                    clipping_mode = "sharp_topology_tear"
                    chosen = worst_tear
                elif soft_mush_hit and not motion_blur_likely:
                    is_clipping_suspect = True
                    clipping_mode = "soft_mush"
                    chosen = worst_blur
                else:
                    chosen = worst_blur

                if chosen is not None:
                    clipping_suspect_frame = {
                        "frame_index": chosen["idx"],
                        "timestamp_seconds": chosen["ts"],
                        "ratio": chosen["ratio"],
                        "sharpness": chosen["sharp"],
                        "anchor_sharpness": round(anchor_sharp, 2),
                        "sharpness_retain_ratio": round(chosen["retain_ratio"], 3),
                        "center_sharpness_retain": round(chosen["center_retain"], 3),
                        "border_sharpness_retain": round(chosen["border_retain"], 3),
                        "nonrigid_center_mad": round(chosen["nonrigid_center_mad"], 3),
                        "edge_component_delta": int(chosen["edge_component_delta"]),
                        "anomaly_score": round(chosen["anomaly"], 4),
                        "clipping_mode": clipping_mode,
                        "motion_blur_likely": motion_blur_likely,
                    }
                    if len(panels) >= 4 and not motion_blur_likely and (
                        is_clipping_suspect or chosen["sharp"] < min(panels[1][2], panels[2][2]) * 0.85
                    ):
                        target_slot = 1 if chosen["ratio"] < 0.52 else 2
                        panels[target_slot] = (chosen["idx"], chosen["ts"], chosen["sharp"], chosen["pil_img"])
                        thumbs[target_slot] = chosen["thumb"]
                        samples[target_slot] = {
                            "ratio": chosen["ratio"],
                            "frame_index": chosen["idx"],
                            "timestamp_seconds": chosen["ts"],
                            "sharpness": chosen["sharp"],
                            "mean_luma": chosen["luma"],
                            "is_blank_or_dark": bool(chosen["luma"] < 8.0),
                            "is_clipping_suspect": is_clipping_suspect,
                            "clipping_mode": clipping_mode,
                        }
                        if is_clipping_suspect:
                            suspect_slot = target_slot

        diffs: list[float] = []
        center_diffs: list[float] = []
        border_diffs: list[float] = []
        for idx in range(len(thumbs) - 1):
            delta_arr = np.abs(thumbs[idx + 1].astype(np.float32) - thumbs[idx].astype(np.float32))
            diffs.append(float(delta_arr.mean()))
            th_h, th_w = delta_arr.shape[:2]
            cy0, cy1 = max(0, int(round(th_h * 0.22))), max(1, int(round(th_h * 0.78)))
            cx0, cx1 = max(0, int(round(th_w * 0.22))), max(1, int(round(th_w * 0.78)))
            c_mask = np.zeros((th_h, th_w), dtype=bool)
            c_mask[cy0:cy1, cx0:cx1] = True
            if c_mask.any():
                center_diffs.append(float(delta_arr[c_mask].mean()))
            if (~c_mask).any():
                border_diffs.append(float(delta_arr[~c_mask].mean()))
        motion_mean_diff = round(sum(diffs) / len(diffs), 3) if diffs else 0.0
        center_motion_mean_diff = round(sum(center_diffs) / len(center_diffs), 3) if center_diffs else motion_mean_diff
        border_motion_mean_diff = round(sum(border_diffs) / len(border_diffs), 3) if border_diffs else motion_mean_diff
        is_static_suspect = bool(len(samples) >= 2 and motion_mean_diff < 1.2)
        pose_locked_vfx_suspect = bool(
            len(samples) >= 2
            and motion_mean_diff >= 8.0
            and center_motion_mean_diff < 2.2
            and border_motion_mean_diff > 0.0
            and (center_motion_mean_diff / border_motion_mean_diff) < 0.35
        )

        sharp_values = [item["sharpness"] for item in samples]
        sharp_min = round(min(sharp_values), 2)
        sharp_avg = round(sum(sharp_values) / len(sharp_values), 2)
        has_black_frames = any(item["is_blank_or_dark"] for item in samples)

        # Tail frame relay selection in [85%, 98%] window.
        tail_start = max(0, int(round(0.85 * (frame_count - 1))))
        tail_end = max(tail_start, min(frame_count - 1, int(round(0.98 * (frame_count - 1)))))
        step = max(1, (tail_end - tail_start + 1) // 8)
        best_tail: tuple[tuple[int, float], int, Any, float] | None = None
        for idx in range(tail_start, tail_end + 1, step):
            pil_img = _read_video_frame_rgb(cap, idx)
            if pil_img is None:
                continue
            sharp, luma, _ = _frame_metrics_from_pil(pil_img)
            rank = (1 if luma >= 8.0 else 0, sharp)
            if best_tail is None or rank > best_tail[0]:
                best_tail = (rank, idx, pil_img, sharp)
        if best_tail is None:
            last_panel = panels[-1]
            best_tail = ((1, last_panel[2]), last_panel[0], last_panel[3], last_panel[2])

        sheet_img = _build_contact_sheet(
            shot_id,
            panels,
            width=width,
            height=height,
            duration=duration_seconds,
            sharp_avg=sharp_avg,
            motion_mean_diff=motion_mean_diff,
            suspect_slot=suspect_slot,
        )

        # Dense 8-frame temporal decomposition across [4%..95%] for flywheel quality verification.
        dense_ratios = (0.04, 0.17, 0.30, 0.43, 0.56, 0.69, 0.82, 0.95)
        dense_panels: list[tuple[int, float, float, float, Any]] = []
        dense_lumas: list[float] = []
        dense_sharps: list[float] = []
        dense_step_diffs: list[float] = []
        dense_center_step_diffs: list[float] = []
        dense_color_stats: list[dict[str, float]] = []
        dense_vfx_frames: list[dict[str, Any]] = []
        prev_dense_thumb: Any = None
        peak_step_diff = -1.0
        video_motion_peak_seconds: float | None = None
        for d_ratio in dense_ratios:
            d_idx = min(frame_count - 1, max(0, int(round(d_ratio * (frame_count - 1)))))
            d_img = _read_video_frame_rgb(cap, d_idx)
            if d_img is None:
                continue
            d_sharp, d_luma, d_thumb = _frame_metrics_from_pil(d_img)
            d_ts = round(d_idx / fps, 3)
            if prev_dense_thumb is not None:
                d_delta = np.abs(d_thumb.astype(np.float32) - prev_dense_thumb.astype(np.float32))
                step_diff = round(float(d_delta.mean()), 2)
                dh, dw = d_delta.shape[:2]
                c_step_diff = round(
                    float(
                        d_delta[
                            int(round(dh * 0.20)) : int(round(dh * 0.80)),
                            int(round(dw * 0.20)) : int(round(dw * 0.80)),
                        ].mean()
                    ),
                    2,
                )
                dense_step_diffs.append(step_diff)
                dense_center_step_diffs.append(c_step_diff)
                if step_diff > peak_step_diff:
                    peak_step_diff = step_diff
                    video_motion_peak_seconds = d_ts
            else:
                step_diff = 0.0
            prev_dense_thumb = d_thumb
            dense_lumas.append(d_luma)
            dense_sharps.append(d_sharp)
            dense_color_stats.append(_color_lighting_stats(d_img))
            dense_vfx_frames.append(_vfx_frame_metrics(d_img))
            dense_panels.append((d_idx, d_ts, d_sharp, step_diff, d_img))

        temporal_freeze_ratio = (
            round(sum(1 for d in dense_step_diffs if d < 0.40) / len(dense_step_diffs), 3)
            if dense_step_diffs
            else 0.0
        )

        # Tail-stall detection: catches H3 "acted in first 2s, held pose for trailing 2-3s".
        trailing_stalls = 0
        for c_diff, f_diff in zip(reversed(dense_center_step_diffs), reversed(dense_step_diffs)):
            if c_diff < 2.2 or f_diff < 1.8:
                trailing_stalls += 1
            else:
                break
        tail_stall_ratio = (
            round(trailing_stalls / len(dense_center_step_diffs), 3)
            if dense_center_step_diffs
            else 0.0
        )
        earlier_center = dense_center_step_diffs[: max(0, len(dense_center_step_diffs) - trailing_stalls)]
        tail_stall_suspect = bool(
            len(dense_center_step_diffs) >= 4
            and (
                (trailing_stalls >= 2 and earlier_center and max(earlier_center) >= 4.5)
                or (trailing_stalls >= 3 and max(dense_center_step_diffs) >= 3.0)
            )
        )
        active_end_s = duration_seconds
        if tail_stall_suspect and len(dense_panels) >= 2:
            active_panel_idx = max(1, len(dense_panels) - 1 - trailing_stalls)
            active_end_s = round(min(duration_seconds, float(dense_panels[active_panel_idx][1]) + 0.35), 3)
        recommended_action_window = {
            "start_s": 0.0,
            "end_s": active_end_s,
            "duration_s": round(max(0.5, active_end_s), 3),
            "tail_stall_detected": tail_stall_suspect,
        }

        # Objective VFX quality metrics across dense frames.
        neon_clip_max = round(max((vf["neon_clip_ratio"] for vf in dense_vfx_frames), default=0.0), 4)
        glow_area_max = round(max((vf["glow_area_ratio"] for vf in dense_vfx_frames), default=0.0), 4)
        glow_area_first = round(dense_vfx_frames[0]["glow_area_ratio"], 4) if dense_vfx_frames else 0.0
        glow_detail_vals = [
            vf["glow_micro_detail"] for vf in dense_vfx_frames if vf["glow_area_ratio"] >= 0.04
        ]
        glow_micro_detail_min = round(min(glow_detail_vals), 2) if glow_detail_vals else 0.0
        glow_ious: list[float] = []
        glow_internal_deltas: list[float] = []
        glow_active_frames = 0
        for i, vf in enumerate(dense_vfx_frames):
            if vf["glow_area_ratio"] >= 0.05:
                glow_active_frames += 1
            if i + 1 < len(dense_vfx_frames):
                m1 = vf["glow_mask"]
                m2 = dense_vfx_frames[i + 1]["glow_mask"]
                union = int((m1 | m2).sum())
                if union >= 20:
                    inter = int((m1 & m2).sum())
                    glow_ious.append(inter / union)
                    if inter >= 16:
                        g1 = vf["gray_small"].astype(np.float32)
                        g2 = dense_vfx_frames[i + 1]["gray_small"].astype(np.float32)
                        glow_internal_deltas.append(float(np.abs(g2 - g1)[m1 & m2].mean()))
        glow_mask_iou_mean = round(sum(glow_ious) / len(glow_ious), 4) if glow_ious else 0.0
        glow_internal_delta_mean = (
            round(sum(glow_internal_deltas) / len(glow_internal_deltas), 2) if glow_internal_deltas else 0.0
        )

        flat_vfx_blob_suspect = bool(
            (glow_area_max >= 0.06 and neon_clip_max >= 0.055 and glow_micro_detail_min < 280.0)
            or (glow_area_max >= 0.12 and glow_micro_detail_min < 140.0)
        )
        vfx_static_sticker_suspect = bool(
            glow_active_frames >= 4
            and len(glow_ious) >= 3
            and glow_mask_iou_mean >= 0.86
            and glow_internal_delta_mean < 4.5
        )
        dense_anchor_sharp = max(1.0, float(dense_sharps[0])) if dense_sharps else 1.0
        vfx_bloat_collapse_suspect = bool(
            len(dense_sharps) >= 3
            and (glow_area_max - glow_area_first) >= 0.22
            and min(dense_sharps[1:]) < 0.55 * dense_anchor_sharp
        )
        vfx_metrics = {
            "neon_clip_ratio_max": neon_clip_max,
            "glow_area_ratio_max": glow_area_max,
            "glow_micro_detail_min": glow_micro_detail_min,
            "glow_mask_iou_mean": glow_mask_iou_mean,
            "glow_internal_delta_mean": glow_internal_delta_mean,
            "flat_vfx_blob_suspect": flat_vfx_blob_suspect,
            "vfx_static_sticker_suspect": vfx_static_sticker_suspect,
            "vfx_bloat_collapse_suspect": vfx_bloat_collapse_suspect,
        }
        flicker_max_luma_jump = (
            round(max(abs(dense_lumas[i + 1] - dense_lumas[i]) for i in range(len(dense_lumas) - 1)), 2)
            if len(dense_lumas) >= 2
            else 0.0
        )
        if len(dense_color_stats) >= 2:
            adj_lab_deltas = [
                math.hypot(
                    dense_color_stats[i + 1]["lab_a"] - dense_color_stats[i]["lab_a"],
                    dense_color_stats[i + 1]["lab_b"] - dense_color_stats[i]["lab_b"],
                )
                for i in range(len(dense_color_stats) - 1)
            ]
            color_drift_max_delta = round(max(adj_lab_deltas), 2)
            first_a, first_b = dense_color_stats[0]["lab_a"], dense_color_stats[0]["lab_b"]
            color_drift_span = round(
                max(
                    math.hypot(st["lab_a"] - first_a, st["lab_b"] - first_b)
                    for st in dense_color_stats
                ),
                2,
            )
            color_temp_max_jump = round(
                max(
                    abs(dense_color_stats[i + 1]["warmth_rg_delta"] - dense_color_stats[i]["warmth_rg_delta"])
                    for i in range(len(dense_color_stats) - 1)
                ),
                2,
            )
            saturation_max_jump = round(
                max(
                    abs(dense_color_stats[i + 1]["mean_saturation"] - dense_color_stats[i]["mean_saturation"])
                    for i in range(len(dense_color_stats) - 1)
                ),
                2,
            )
        else:
            color_drift_max_delta = 0.0
            color_drift_span = 0.0
            color_temp_max_jump = 0.0
            saturation_max_jump = 0.0

        dark_clip_ratio_avg = (
            round(sum(st["dark_clip_ratio"] for st in dense_color_stats) / len(dense_color_stats), 4)
            if dense_color_stats
            else 0.0
        )
        overexposed_ratio_avg = (
            round(sum(st["overexposed_ratio"] for st in dense_color_stats) / len(dense_color_stats), 4)
            if dense_color_stats
            else 0.0
        )
        color_drift_suspect = bool(color_drift_max_delta >= 18.0 or color_temp_max_jump >= 24.0)

        dense_sheet_img = _build_dense_strip(
            shot_id,
            dense_panels,
            width=width,
            height=height,
            duration=duration_seconds,
            freeze_ratio=temporal_freeze_ratio,
            flicker_jump=flicker_max_luma_jump,
        )
        roi_zoom_img, roi_box_source = _build_roi_zoom_strip(
            shot_id,
            panels,
            color_drift=color_drift_max_delta,
            luma_jump=flicker_max_luma_jump,
        )

    sheet_buf = BytesIO()
    sheet_img.save(sheet_buf, format="JPEG", quality=88)
    sheet_bytes = sheet_buf.getvalue()

    dense_bytes: bytes | None = None
    if dense_sheet_img is not None:
        dense_buf = BytesIO()
        dense_sheet_img.save(dense_buf, format="JPEG", quality=88)
        dense_bytes = dense_buf.getvalue()

    roi_bytes: bytes | None = None
    if roi_zoom_img is not None:
        roi_buf = BytesIO()
        roi_zoom_img.save(roi_buf, format="JPEG", quality=90)
        roi_bytes = roi_buf.getvalue()

    tail_buf = BytesIO()
    best_tail[2].save(tail_buf, format="PNG")
    tail_bytes = tail_buf.getvalue()

    safe_shot = "".join(c if c.isalnum() or c in "._-" else "_" for c in shot_id) or "shot"
    sheet_path: str | None = None
    dense_path: str | None = None
    roi_zoom_path: str | None = None
    tail_path: str | None = None
    if output_dir is not None:
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        sheet_file = out_root / f"{safe_shot}_contact_sheet.jpg"
        sheet_file.write_bytes(sheet_bytes)
        sheet_path = str(sheet_file)
        if dense_bytes is not None:
            dense_file = out_root / f"{safe_shot}_dense_strip.jpg"
            dense_file.write_bytes(dense_bytes)
            dense_path = str(dense_file)
        if roi_bytes is not None:
            roi_file = out_root / f"{safe_shot}_roi_zoom.jpg"
            roi_file.write_bytes(roi_bytes)
            roi_zoom_path = str(roi_file)
        tail_file = out_root / f"{safe_shot}_tail.png"
        tail_file.write_bytes(tail_bytes)
        tail_path = str(tail_file)

    contact_sheet_ref = registry.put_bytes(sheet_bytes) if registry is not None else None
    dense_strip_ref = (registry.put_bytes(dense_bytes) if (registry is not None and dense_bytes is not None) else None)
    roi_zoom_ref = (registry.put_bytes(roi_bytes) if (registry is not None and roi_bytes is not None) else None)
    best_tail_frame_ref = registry.put_bytes(tail_bytes) if registry is not None else None

    return {
        "schema": "video-diagnostics/v1",
        "available": True,
        "shot_id": shot_id,
        "path": str(path),
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
        "sharpness_min": sharp_min,
        "sharpness_avg": sharp_avg,
        "motion_mean_diff": motion_mean_diff,
        "center_motion_mean_diff": center_motion_mean_diff,
        "border_motion_mean_diff": border_motion_mean_diff,
        "pose_locked_vfx_suspect": pose_locked_vfx_suspect,
        "temporal_freeze_ratio": temporal_freeze_ratio,
        "tail_stall_ratio": tail_stall_ratio,
        "tail_stall_suspect": tail_stall_suspect,
        "dense_center_step_diffs": dense_center_step_diffs,
        "recommended_action_window": recommended_action_window,
        "vfx_metrics": vfx_metrics,
        "flicker_max_luma_jump": flicker_max_luma_jump,
        "color_drift_max_delta": color_drift_max_delta,
        "color_drift_span": color_drift_span,
        "color_temp_max_jump": color_temp_max_jump,
        "saturation_max_jump": saturation_max_jump,
        "dark_clip_ratio_avg": dark_clip_ratio_avg,
        "overexposed_ratio_avg": overexposed_ratio_avg,
        "color_drift_suspect": color_drift_suspect,
        "video_motion_peak_seconds": video_motion_peak_seconds,
        "is_static_suspect": is_static_suspect,
        "is_clipping_suspect": is_clipping_suspect,
        "clipping_mode": clipping_mode,
        "motion_blur_likely": motion_blur_likely,
        "pan_shift": pan_shift,
        "clipping_suspect_frame": clipping_suspect_frame,
        "has_black_frames": has_black_frames,
        "contact_sheet_path": sheet_path,
        "contact_sheet_ref": contact_sheet_ref,
        "dense_strip_path": dense_path,
        "dense_strip_ref": dense_strip_ref,
        "roi_zoom_path": roi_zoom_path,
        "roi_zoom_ref": roi_zoom_ref,
        "roi_box_source": roi_box_source,
        "best_tail_frame_path": tail_path,
        "best_tail_frame_ref": best_tail_frame_ref,
        "tail_frame_index": best_tail[1],
        "tail_sharpness": best_tail[3],
        "samples": samples,
    }


def extract_audio_diagnostics(audio_path: str | Path | None) -> dict[str, Any]:
    """Extract loudness, clipping, silence envelope, and onset peaks from a WAV audio track."""
    if not audio_path:
        return {"available": False, "reason": "missing_path"}
    path = Path(audio_path)
    if not path.is_file():
        return {"available": False, "reason": "file_not_found", "path": str(path)}
    if path.suffix.lower() != ".wav":
        return {"available": False, "reason": f"unsupported_audio_suffix:{path.suffix}", "path": str(path)}

    import wave
    import numpy as np

    try:
        with wave.open(str(path), "rb") as wf:
            channels = int(wf.getnchannels() or 1)
            sample_width = int(wf.getsampwidth() or 2)
            sample_rate = int(wf.getframerate() or 16000)
            n_frames = int(wf.getnframes() or 0)
            raw_bytes = wf.readframes(n_frames)
    except (wave.Error, OSError) as exc:
        return {"available": False, "reason": f"audio_read_error:{type(exc).__name__}", "path": str(path)}

    if n_frames <= 0 or sample_rate <= 0 or not raw_bytes:
        return {"available": False, "reason": "empty_audio", "path": str(path)}

    if sample_width == 1:
        samples = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sample_width == 3:
        raw_u8 = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 3)
        signed = (
            raw_u8[:, 0].astype(np.int32)
            | (raw_u8[:, 1].astype(np.int32) << 8)
            | (raw_u8[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed >= 0x800000, signed - 0x1000000, signed)
        samples = signed.astype(np.float32) / 8388608.0
    else:
        return {"available": False, "reason": f"unsupported_sample_width:{sample_width}", "path": str(path)}

    if channels > 1:
        usable = (samples.size // channels) * channels
        mono = samples[:usable].reshape(-1, channels).mean(axis=1)
    else:
        mono = samples

    duration_seconds = round(mono.size / float(sample_rate), 3)
    abs_mono = np.abs(mono)
    peak_amp = float(abs_mono.max()) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(mono * mono))) if mono.size else 0.0
    rms_db = round(20.0 * math.log10(max(rms, 1e-6)), 2)
    clipping_ratio = round(float((abs_mono >= 0.985).mean()), 5)

    # Compute 50ms frame RMS envelope for silence & onset peak analysis.
    win_size = max(1, int(round(sample_rate * 0.05)))
    n_wins = max(1, mono.size // win_size)
    trimmed = mono[: n_wins * win_size].reshape(n_wins, win_size)
    win_rms = np.sqrt(np.mean(trimmed * trimmed, axis=1))
    win_dt = win_size / float(sample_rate)

    silence_thresh = max(0.008, peak_amp * 0.05) if peak_amp > 0.02 else 0.008
    is_active = win_rms >= silence_thresh
    silence_ratio = round(float((~is_active).mean()), 3)

    active_indices = np.nonzero(is_active)[0]
    if active_indices.size > 0:
        leading_silence_s = round(float(active_indices[0]) * win_dt, 3)
        trailing_silence_s = round(float(n_wins - 1 - active_indices[-1]) * win_dt, 3)
    else:
        leading_silence_s = duration_seconds
        trailing_silence_s = 0.0

    # Longest internal silence gap between first and last active frames.
    max_internal_run = 0
    if active_indices.size >= 2:
        run = 0
        for idx in range(int(active_indices[0]), int(active_indices[-1]) + 1):
            if not is_active[idx]:
                run += 1
                if run > max_internal_run:
                    max_internal_run = run
            else:
                run = 0
    max_internal_silence_s = round(max_internal_run * win_dt, 3)

    # Coalesce speech/audio active windows.
    speech_active_windows: list[list[float]] = []
    in_seg = False
    seg_start = 0
    for idx, flag in enumerate(is_active):
        if flag and not in_seg:
            in_seg = True
            seg_start = idx
        elif not flag and in_seg:
            in_seg = False
            speech_active_windows.append([round(seg_start * win_dt, 2), round(idx * win_dt, 2)])
    if in_seg:
        speech_active_windows.append([round(seg_start * win_dt, 2), round(n_wins * win_dt, 2)])

    # Energy-flux onset peaks (sharp rising edges).
    onset_times_s: list[float] = []
    if n_wins >= 3 and peak_amp >= 0.02:
        flux = np.maximum(0.0, np.diff(win_rms, prepend=win_rms[0]))
        flux_thresh = max(0.015, float(flux.mean()) + 1.5 * float(flux.std()))
        for idx in range(1, n_wins - 1):
            if flux[idx] >= flux_thresh and flux[idx] >= flux[idx - 1] and flux[idx] >= flux[idx + 1]:
                ts = round(idx * win_dt, 3)
                if not onset_times_s or (ts - onset_times_s[-1]) >= 0.15:
                    onset_times_s.append(ts)
        onset_times_s = onset_times_s[:16]

    return {
        "schema": "audio-diagnostics/v1",
        "available": True,
        "path": str(path),
        "duration_seconds": duration_seconds,
        "sample_rate": sample_rate,
        "channels": channels,
        "rms_db": rms_db,
        "peak_amplitude": round(peak_amp, 4),
        "clipping_ratio": clipping_ratio,
        "silence_ratio": silence_ratio,
        "leading_silence_s": leading_silence_s,
        "trailing_silence_s": trailing_silence_s,
        "max_internal_silence_s": max_internal_silence_s,
        "onset_times_s": onset_times_s,
        "speech_active_windows": speech_active_windows[:12],
    }


def compute_av_sync_diagnostics(
    video_diag: dict[str, Any] | None,
    audio_diag: dict[str, Any] | None,
    *,
    duration_delta_max_s: float = 0.85,
) -> dict[str, Any]:
    """Compute audio-visual duration and peak-onset alignment telemetry."""
    v = video_diag if isinstance(video_diag, dict) else {}
    a = audio_diag if isinstance(audio_diag, dict) else {}
    if v.get("available") is not True or a.get("available") is not True:
        return {"available": False, "reason": "missing_video_or_audio"}

    vid_dur = float(v.get("duration_seconds") or 0.0)
    aud_dur = float(a.get("duration_seconds") or 0.0)
    delta = round(abs(vid_dur - aud_dur), 3)

    peak_ts = v.get("video_motion_peak_seconds")
    onsets = [float(t) for t in (a.get("onset_times_s") or []) if isinstance(t, (int, float))]
    nearest_onset_delta: float | None = None
    if isinstance(peak_ts, (int, float)) and onsets:
        nearest_onset_delta = round(min(abs(float(peak_ts) - t) for t in onsets), 3)

    return {
        "schema": "av-sync-diagnostics/v1",
        "available": True,
        "video_duration_seconds": round(vid_dur, 3),
        "audio_duration_seconds": round(aud_dur, 3),
        "duration_delta_seconds": delta,
        "duration_mismatch_suspect": bool(delta > float(duration_delta_max_s)),
        "video_motion_peak_seconds": peak_ts,
        "audio_onset_count": len(onsets),
        "nearest_onset_delta_seconds": nearest_onset_delta,
    }


def refresh_shot_diagnostics(
    manifest: ProjectManifest,
    root: str | Path,
    shot_id: str,
) -> dict[str, Any]:
    """Refresh image, video, audio, AV-sync, and genre-calibrated objective diagnostics for a shot."""
    from .qa_profiles import evaluate_objective_qa_flags, resolve_shot_qa_profile

    project_root = Path(root)
    diag_dir = project_root / "diagnostics"
    registry = ArtifactRegistry(project_root / "artifacts")
    shot = manifest.shot(shot_id)
    profile = resolve_shot_qa_profile(manifest, shot)

    image_diag = extract_image_diagnostics(shot.assets.get("image"))
    last_frame_diag = (
        extract_image_diagnostics(shot.assets.get("last_frame"))
        if shot.assets.get("last_frame")
        else {"available": False, "reason": "not_set"}
    )
    video_diag = (
        extract_video_diagnostics(
            shot.assets.get("video"),
            registry=registry,
            output_dir=diag_dir,
            shot_id=shot.id,
        )
        if shot.assets.get("video")
        else {"available": False, "reason": "not_rendered"}
    )
    audio_diag = (
        extract_audio_diagnostics(shot.assets.get("audio"))
        if shot.assets.get("audio")
        else {"available": False, "reason": "not_synthesized"}
    )
    av_sync = compute_av_sync_diagnostics(
        video_diag,
        audio_diag,
        duration_delta_max_s=float((profile.get("thresholds") or {}).get("audio_duration_delta_max_s", 0.85)),
    )

    bundle: dict[str, Any] = {
        "schema": "shot-diagnostics/v1",
        "shot_id": shot.id,
        "qa_profile_id": profile["id"],
        "qa_profile_resolved_from": profile.get("resolved_from"),
        "image": image_diag,
        "last_frame": last_frame_diag,
        "video": video_diag,
        "audio": audio_diag,
        "av_sync": av_sync,
    }
    bundle["objective_qa"] = evaluate_objective_qa_flags(
        bundle,
        profile,
        stage="video" if video_diag.get("available") else "image",
        has_dialogue=bool(getattr(shot, "dialogue_lines", None)),
    )
    shot.diagnostics = bundle
    return bundle


def _is_explicit_scene_cut(shot: Shot, action_text: str) -> bool:
    ctx = shot.scene_context if isinstance(shot.scene_context, dict) else {}
    if ctx.get("relay_from_prev_tail") is False:
        return True
    return bool(_SCENE_CUT_RE.search(action_text or ""))


def _compact_bible_summary(shot: Shot, character_bible: dict[str, Any] | None) -> str:
    if not isinstance(character_bible, dict) or not character_bible:
        return ""
    ctx = shot.scene_context if isinstance(shot.scene_context, dict) else {}
    haystack = f"{shot.prompt_override or ''} {ctx.get('visual_action') or ''} {shot.text or ''}"
    locks = set(str(x).strip() for x in (shot.character_locks or []) if str(x).strip())
    chunks: list[str] = []
    for key, profile in character_bible.items():
        if not isinstance(profile, dict):
            continue
        name = str(profile.get("name") or key or "").strip()
        aliases = [str(a).strip() for a in (profile.get("aliases") or []) if str(a).strip()]
        matched = (
            key in locks
            or name in locks
            or (name and name in haystack)
            or any(alias in haystack for alias in aliases)
        )
        if not matched:
            continue
        dna = str(profile.get("visual_dna") or "").strip()
        if not dna:
            parts = [str(profile.get(k) or "").strip() for k in ("appearance", "clothing")]
            dna = "，".join(p for p in parts if p)
        chunks.append(f"{name}({dna[:60]})" if dna else name)
    joined = "；".join(chunks)
    return joined[:120] if len(joined) > 120 else joined


def build_shot_frame_plan(
    shot: Shot,
    character_bible: dict[str, Any] | None = None,
    prev_tail_frame_ref: str | None = None,
    prev_scene_id: str | None = None,
) -> dict[str, Any]:
    """Decompose a shot into start/end frame prompts, motion vector, and relay continuity."""
    ctx = shot.scene_context if isinstance(shot.scene_context, dict) else {}
    raw_action = str(shot.prompt_override or ctx.get("visual_action") or shot.text or "").strip()
    beats = [part.strip(" ，,。.!！？") for part in _BEAT_SPLIT_RE.split(raw_action) if part and part.strip(" ，,。.!！？")]

    if len(beats) >= 2:
        start_beat = beats[0]
        end_beat = beats[-1]
    else:
        single = beats[0] if beats else (shot.text or shot.id).strip()
        start_beat = f"{single}（动作起始静止构图）"
        end_beat = f"{single}（动作完成定格瞬间）"

    prefix_parts: list[str] = []
    location = str(ctx.get("location") or "").strip()
    if location and "以动作描述中的" not in location:
        prefix_parts.append(location)
    framing = str(ctx.get("framing") or "").strip()
    if framing and framing != location and framing not in location:
        if location and location in framing:
            prefix_parts[-1] = framing
        else:
            prefix_parts.append(framing)
    char_summary = _compact_bible_summary(shot, character_bible)
    if char_summary:
        prefix_parts.append(char_summary)
    prefix = "，".join(prefix_parts)

    first_frame_prompt = f"{prefix}；{start_beat}" if prefix else start_beat
    last_frame_prompt = f"{prefix}；{end_beat}" if prefix else end_beat

    motion_bits: list[str] = []
    shot_motion = str(getattr(shot, "motion", "none") or "none").strip()
    if shot_motion != "none":
        motion_bits.append(f"camera {shot_motion}")
    if framing:
        motion_bits.append(framing)
    motion_bits.append(f"{start_beat} -> {end_beat}" if len(beats) >= 2 else raw_action or start_beat)
    motion_vector = "; ".join(motion_bits)

    relay = bool(
        prev_tail_frame_ref
        and (prev_scene_id is None or prev_scene_id == shot.scene_id)
        and not _is_explicit_scene_cut(shot, raw_action)
    )

    vid_diag = (shot.diagnostics or {}).get("video") if isinstance(shot.diagnostics, dict) else None
    extracted_keyframes: list[dict[str, Any]] = []
    contact_sheet_ref: str | None = None
    best_tail_frame_ref: str | None = None
    if isinstance(vid_diag, dict):
        contact_sheet_ref = vid_diag.get("contact_sheet_ref")
        best_tail_frame_ref = vid_diag.get("best_tail_frame_ref")
        for sample in vid_diag.get("samples") or []:
            if isinstance(sample, dict):
                extracted_keyframes.append(
                    {
                        "timestamp_s": sample.get("timestamp_seconds"),
                        "sharpness": sample.get("sharpness"),
                        "frame_index": sample.get("frame_index"),
                    }
                )

    return {
        "schema": "shot-frame-plan/v1",
        "shot_id": shot.id,
        "scene_id": shot.scene_id,
        "relay_from_prev_tail": relay,
        "prev_tail_frame_ref": prev_tail_frame_ref if relay else None,
        "first_frame_prompt": first_frame_prompt,
        "last_frame_prompt": last_frame_prompt,
        "motion_vector": motion_vector,
        "extracted_keyframes": extracted_keyframes,
        "contact_sheet_ref": contact_sheet_ref,
        "best_tail_frame_ref": best_tail_frame_ref,
    }


def _shot_focal_characters(manifest: ProjectManifest, shot: Shot) -> set[str]:
    """Identify focal characters in a shot without blindly absorbing whole-scene cast lists."""
    bible = manifest.character_bible if isinstance(manifest.character_bible, dict) else {}
    chars = (
        bible["characters"]
        if isinstance(bible.get("characters"), dict)
        else {k: v for k, v in bible.items() if isinstance(v, dict)}
    )
    ignore_speakers = {"旁白", "画外音", "narrator", "os", "内心独白", "inner_monologue", "voiceover", "角色"}

    def _match_tokens(raw_tokens: list[str], *, allow_raw: bool = False) -> set[str]:
        matched: set[str] = set()
        for tok in raw_tokens:
            clean = str(tok or "").strip()
            if not clean:
                continue
            low = clean.lower()
            if low in ignore_speakers:
                continue
            if clean in chars:
                matched.add(clean)
                continue
            found_key = False
            for cid, profile in chars.items():
                names = [cid, profile.get("key"), profile.get("name"), *(profile.get("aliases") or [])]
                if any(isinstance(n, str) and n.strip() and n.strip().lower() == low for n in names):
                    matched.add(cid)
                    found_key = True
            if allow_raw and not found_key:
                matched.add(f"raw:{low}")
        return matched

    if isinstance(shot.character_locks, list) and shot.character_locks:
        locked = _match_tokens([str(x) for x in shot.character_locks], allow_raw=True)
        if locked:
            return locked

    if isinstance(shot.dialogue_lines, list) and shot.dialogue_lines:
        speakers = [str(line.get("speaker") or "").strip() for line in shot.dialogue_lines if isinstance(line, dict)]
        spoken = _match_tokens([s for s in speakers if s], allow_raw=True)
        if spoken:
            return spoken

    ctx = shot.scene_context if isinstance(shot.scene_context, dict) else {}
    body_text = "\n".join(
        str(part)
        for part in (shot.prompt_override, shot.text, ctx.get("action"))
        if isinstance(part, str) and part.strip()
    )
    if body_text:
        explicit: set[str] = set()
        for cid, profile in chars.items():
            names = [cid, profile.get("key"), profile.get("name"), *(profile.get("aliases") or [])]
            if any(isinstance(n, str) and n.strip() and n.strip() in body_text for n in names):
                explicit.add(cid)
        if explicit:
            return explicit

    from .character_bible import resolve_shot_characters

    return {
        str(c.get("key") or c.get("id") or c.get("name"))
        for c in resolve_shot_characters(manifest, shot)
        if isinstance(c, dict) and (c.get("key") or c.get("id") or c.get("name"))
    }


def apply_tail_frame_relay(
    manifest: ProjectManifest,
    root: str | Path,
    prev_shot: Shot,
    curr_shot: Shot,
) -> dict[str, Any]:
    """Extract/reuse prev_shot's sharpest tail frame as curr_shot's continuity anchor when in the same scene."""
    project_root = Path(root)
    prev_tail_ref: str | None = None
    prev_tail_path: str | None = None
    relay_skipped_reason: str | None = None
    reanchored = False

    prev_ctx = prev_shot.scene_context if isinstance(prev_shot.scene_context, dict) else {}
    prev_depth = int(prev_ctx.get("_relay_chain_depth") or 0)
    curr_depth = 0

    if prev_shot.scene_id == curr_shot.scene_id:
        prev_focal = _shot_focal_characters(manifest, prev_shot)
        curr_focal = _shot_focal_characters(manifest, curr_shot)
        if prev_focal and curr_focal and prev_focal.isdisjoint(curr_focal):
            relay_skipped_reason = "reverse_shot_character_switch"
        else:
            prev_video = (prev_shot.assets or {}).get("video")
            prev_vid_diag = (
                ((prev_shot.diagnostics or {}).get("video") or {})
                if isinstance(prev_shot.diagnostics, dict)
                else {}
            )
            if prev_video and Path(prev_video).is_file() and not prev_vid_diag.get("best_tail_frame_ref"):
                refresh_shot_diagnostics(manifest, project_root, prev_shot.id)
                prev_vid_diag = (
                    ((prev_shot.diagnostics or {}).get("video") or {})
                    if isinstance(prev_shot.diagnostics, dict)
                    else {}
                )
            prev_tail_ref = prev_vid_diag.get("best_tail_frame_ref")
            prev_tail_path = prev_vid_diag.get("best_tail_frame_path")

            # Anti-generation-loss re-anchoring: after >=3 video hops or degraded tail sharpness, prefer crisp keyframe
            prev_img_diag = (
                ((prev_shot.diagnostics or {}).get("image") or {})
                if isinstance(prev_shot.diagnostics, dict)
                else {}
            )
            tail_sharp = float(prev_vid_diag.get("tail_sharpness") or 0.0)
            img_sharp = float(prev_img_diag.get("sharpness") or 0.0)
            if img_sharp <= 0.0:
                first_asset = (prev_shot.assets or {}).get("image")
                if first_asset and Path(first_asset).is_file() and Path(first_asset).suffix.lower() in _IMAGE_SUFFIXES:
                    img_diag = extract_image_diagnostics(first_asset)
                    img_sharp = float(img_diag.get("sharpness") or 0.0)
            needs_reanchor = bool(
                prev_depth >= 3
                or (0.0 < tail_sharp < 4.0 and img_sharp >= 12.0)
                or (prev_depth >= 1 and tail_sharp > 0.0 and img_sharp > 0.0 and tail_sharp < 0.35 * img_sharp)
            )
            if needs_reanchor:
                for role in ("last_frame", "image"):
                    cand = (prev_shot.assets or {}).get(role)
                    if cand and Path(cand).is_file() and Path(cand).suffix.lower() in _IMAGE_SUFFIXES:
                        prev_tail_path = str(cand)
                        registry = ArtifactRegistry(project_root / "artifacts")
                        prev_tail_ref = registry.put_bytes(Path(cand).read_bytes())
                        reanchored = True
                        break

            if (not prev_tail_path or not Path(prev_tail_path).is_file()) and getattr(
                manifest, "relay_tail_frames", False
            ):
                for role in ("last_frame", "image"):
                    cand = (prev_shot.assets or {}).get(role)
                    if cand and Path(cand).is_file() and Path(cand).suffix.lower() in _IMAGE_SUFFIXES:
                        prev_tail_path = str(cand)
                        if not prev_tail_ref:
                            registry = ArtifactRegistry(project_root / "artifacts")
                            prev_tail_ref = registry.put_bytes(Path(cand).read_bytes())
                        break

    plan = build_shot_frame_plan(
        curr_shot,
        manifest.character_bible,
        prev_tail_frame_ref=prev_tail_ref,
        prev_scene_id=prev_shot.scene_id,
    )
    if not isinstance(curr_shot.scene_context, dict):
        curr_shot.scene_context = {}

    if relay_skipped_reason:
        plan["relay_from_prev_tail"] = False
        plan["relay_skipped_reason"] = relay_skipped_reason
        curr_shot.scene_context["relay_skipped_reason"] = relay_skipped_reason
        curr_shot.scene_context.pop("_relay_ref_image", None)
        curr_shot.scene_context.pop("_relay_source_shot_id", None)
        curr_shot.scene_context.pop("_relay_source_digest", None)
        curr_shot.scene_context["_relay_chain_depth"] = 0
    elif (
        plan["relay_from_prev_tail"]
        and prev_tail_path
        and Path(prev_tail_path).is_file()
        and Path(prev_tail_path).suffix.lower() in _IMAGE_SUFFIXES
    ):
        curr_depth = 1 if reanchored else (prev_depth + 1)
        tail_digest = hashlib.sha256(Path(prev_tail_path).read_bytes()).hexdigest()[:16]
        curr_shot.scene_context["_relay_ref_image"] = str(prev_tail_path)
        curr_shot.scene_context["_relay_source_shot_id"] = prev_shot.id
        curr_shot.scene_context["_relay_source_digest"] = tail_digest
        curr_shot.scene_context["_relay_chain_depth"] = curr_depth
        curr_shot.scene_context.pop("relay_skipped_reason", None)
        curr_shot.scene_context.pop("_relay_continuity_stale", None)
        curr_shot.scene_context.pop("_relay_stale_upstream_shot_id", None)
        plan["relay_reanchored"] = reanchored
        plan["relay_chain_depth"] = curr_depth
    else:
        curr_shot.scene_context.pop("_relay_ref_image", None)
        curr_shot.scene_context.pop("_relay_source_shot_id", None)
        curr_shot.scene_context.pop("_relay_source_digest", None)
        curr_shot.scene_context["_relay_chain_depth"] = 0

    curr_shot.scene_context["_frame_plan"] = plan
    return plan


def mark_downstream_relay_staleness(manifest: ProjectManifest, reworked_shot_id: str) -> list[str]:
    """Flag downstream shots that relayed their continuity from a subsequently reworked upstream shot."""
    upstream_ids = {reworked_shot_id}
    affected: list[str] = []
    seen_target = False
    for shot in manifest.shots:
        if shot.id == reworked_shot_id:
            seen_target = True
            continue
        if not seen_target:
            continue
        ctx = shot.scene_context if isinstance(shot.scene_context, dict) else {}
        src = ctx.get("_relay_source_shot_id")
        if src in upstream_ids:
            if not isinstance(shot.scene_context, dict):
                shot.scene_context = {}
            shot.scene_context["_relay_continuity_stale"] = True
            shot.scene_context["_relay_stale_upstream_shot_id"] = reworked_shot_id
            upstream_ids.add(shot.id)
            affected.append(shot.id)
    return affected
