"""In-pipeline verification gates for assembled chapter and multi-chapter master MP4s."""
from __future__ import annotations

import json
import math
import re
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import cv2

from .h3_bundle import extract_dialogue_tags, normalize_dialogue_text
from .local_media import find_ffmpeg

MASTER_VERIFY_SCHEMA = "master-verify-report/v1"


@dataclass(frozen=True)
class VerifyFinding:
    """Single verification error or warning emitted by master verification."""

    severity: str
    code: str
    message: str
    detail: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.detail is not None:
            out["detail"] = self.detail
        return out


@dataclass
class MasterVerifyReport:
    """Structured release verification report for an assembled MP4."""

    schema: str
    media_path: str
    profile: str
    ok: bool
    errors: list[VerifyFinding] = field(default_factory=list)
    warnings: list[VerifyFinding] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    human_ear_check_points: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "media_path": self.media_path,
            "profile": self.profile,
            "ok": self.ok,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "metrics": self.metrics,
            "human_ear_check_points": self.human_ear_check_points,
        }


def _run_ffmpeg(args: list[str], *, ffmpeg_bin: str | None = None) -> str:
    exe = ffmpeg_bin or find_ffmpeg()
    proc = subprocess.run(
        [exe, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {' '.join(args)}\n{proc.stdout[-2000:]}")
    return proc.stdout


def probe_video_stream(mp4_path: Path) -> dict[str, Any]:
    """Probe MP4 video dimensions, FPS, frame count, and duration via OpenCV."""
    p = Path(mp4_path)
    if not p.is_file():
        return {"exists": False, "bytes": 0, "width": 0, "height": 0, "fps": 0.0, "frames": 0, "dur_sec": 0.0}
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return {"exists": True, "bytes": p.stat().st_size, "width": 0, "height": 0, "fps": 0.0, "frames": 0, "dur_sec": 0.0}
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {
        "exists": True,
        "bytes": p.stat().st_size,
        "width": w,
        "height": h,
        "fps": round(fps, 4),
        "frames": frames,
        "dur_sec": round(frames / fps, 3) if fps > 0 else 0.0,
    }


def compute_video_stream_md5(mp4_path: Path, *, ffmpeg_bin: str | None = None) -> str:
    """Compute MD5 hash over raw demuxed video packets (`-c copy`) for lossless-copy verification."""
    out = _run_ffmpeg(
        ["-v", "error", "-i", str(mp4_path), "-map", "0:v:0", "-c", "copy", "-f", "md5", "-"],
        ffmpeg_bin=ffmpeg_bin,
    )
    m = re.search(r"MD5=([0-9a-fA-F]+)", out)
    if not m:
        raise RuntimeError(f"Unable to parse video stream MD5 for {mp4_path}: {out}")
    return m.group(1).lower()


def _parse_ebur128_output(raw_log: str, *, include_peak: bool) -> tuple[dict[str, Any], dict[float, float]]:
    m_map: dict[float, float] = {}
    for t_s, m_s in re.findall(r"t:\s*([\d.]+)\s+TARGET:[^\r\n]*?\sM:\s*([-\d.]+)", raw_log):
        m_map[round(float(t_s), 1)] = float(m_s)

    summary_idx = raw_log.rfind("Summary:")
    summary = raw_log[summary_idx:] if summary_idx != -1 else raw_log
    i_m = re.search(r"\bI:\s*([-\d.]+)\s*LUFS", summary)
    lra_m = re.search(r"\bLRA:\s*([-\d.]+)\s*LU", summary)
    tp_m = re.search(r"\bPeak:\s*([-\d.]+)\s*dBFS", summary) if include_peak else None
    if not i_m or not lra_m or (include_peak and not tp_m):
        raise RuntimeError(f"Failed to parse ebur128 summary:\n{raw_log[-1500:]}")

    return (
        {
            "integrated_lufs": float(i_m.group(1)),
            "lra_lu": float(lra_m.group(1)),
            "true_peak_dbtp": float(tp_m.group(1)) if tp_m else None,
        },
        m_map,
    )


def measure_ebur128(
    media_path: Path,
    *,
    include_peak: bool = True,
    ffmpeg_bin: str | None = None,
) -> tuple[dict[str, Any], dict[float, float]]:
    """Measure ITU-R BS.1770 integrated loudness, LRA, true peak, and 0.1s momentary LUFS map."""
    af = "ebur128=peak=true" if include_peak else "ebur128"
    out = _run_ffmpeg(
        ["-hide_banner", "-nostats", "-i", str(media_path), "-vn", "-af", af, "-f", "null", "-"],
        ffmpeg_bin=ffmpeg_bin,
    )
    return _parse_ebur128_output(out, include_peak=include_peak)


def _parse_hhmmss(txt: str) -> float:
    h, m, s = txt.split(":")
    return int(h) * 3600.0 + int(m) * 60.0 + float(s)


def _parse_detect_windows(raw_log: str, prefix: str, total_dur: float) -> list[dict[str, float]]:
    windows: list[dict[str, float]] = []
    if prefix == "black":
        for st_s, en_s, d_s in re.findall(
            r"black_start:\s*([-\d.]+)\s+black_end:\s*([-\d.]+)\s+black_duration:\s*([-\d.]+)",
            raw_log,
        ):
            windows.append({"start": float(st_s), "end": float(en_s), "duration": float(d_s)})
        return windows

    open_start: float | None = None
    start_pat = re.compile(rf"{prefix}_start:\s*([-\d.]+)")
    end_pat = re.compile(rf"{prefix}_end:\s*([-\d.]+)\s*\|\s*{prefix}_duration:\s*([-\d.]+)")
    for line in raw_log.splitlines():
        m_st = start_pat.search(line)
        if m_st and f"{prefix}_end:" not in line:
            open_start = float(m_st.group(1))
            continue
        m_en = end_pat.search(line)
        if m_en:
            en = float(m_en.group(1))
            dur = float(m_en.group(2))
            st = open_start if open_start is not None else max(0.0, en - dur)
            windows.append({"start": st, "end": en, "duration": dur})
            open_start = None
    if open_start is not None and total_dur > open_start:
        windows.append({"start": open_start, "end": total_dur, "duration": round(total_dur - open_start, 3)})
    return windows


def scan_media_anomalies(
    mp4_path: Path,
    *,
    black_min_d: float = 0.3,
    black_pix_th: float = 0.06,
    freeze_noise: float = 0.002,
    freeze_min_d: float = 1.5,
    silence_db: float = -55.0,
    silence_min_d: float = 2.0,
    ffmpeg_bin: str | None = None,
) -> dict[str, Any]:
    """Run single-pass decoded video+audio anomaly scan (black, freeze, silence, ebur128)."""
    fc = (
        f"[0:v]blackdetect=d={black_min_d}:pix_th={black_pix_th},"
        f"freezedetect=n={freeze_noise}:d={freeze_min_d}[v];"
        f"[0:a]silencedetect=n={silence_db}dB:d={silence_min_d},ebur128=peak=true[a]"
    )
    out = _run_ffmpeg(
        [
            "-hide_banner", "-nostats", "-i", str(mp4_path),
            "-filter_complex", fc,
            "-map", "[v]", "-map", "[a]", "-f", "null", "-",
        ],
        ffmpeg_bin=ffmpeg_bin,
    )
    dur_m = re.search(r"Duration:\s*(\d+:\d+:\d+(?:\.\d+)?)", out)
    container_dur = _parse_hhmmss(dur_m.group(1)) if dur_m else 0.0
    loudness, m_map = _parse_ebur128_output(out, include_peak=True)

    return {
        "container_dur_sec": round(container_dur, 3),
        "black_windows": _parse_detect_windows(out, "black", container_dur),
        "freeze_windows": _parse_detect_windows(out, "freeze", container_dur),
        "silence_windows": _parse_detect_windows(out, "silence", container_dur),
        "loudness": loudness,
        "momentary_lufs": m_map,
    }


def _power_mean_lufs(vals: list[float]) -> float:
    valid = [v for v in vals if v > -70.0]
    if not valid:
        return -70.0
    return 10.0 * math.log10(sum(10.0 ** (v / 10.0) for v in valid) / len(valid))


def _cue_samples(m_map: dict[float, float], start: float, end: float) -> list[float]:
    span = end - start
    a = start + 0.4 if span > 0.45 else start
    steps = int((end - a) / 0.1) + 1
    samples = [m_map.get(round(a + 0.1 * k, 1), -120.0) for k in range(max(1, steps))]
    valid = [v for v in samples if v > -70.0]
    if valid:
        return valid
    # Fallback around midpoint for sub-second synthetic cues
    mid_t = round((start + end) * 0.5, 1)
    mid_v = m_map.get(mid_t, -120.0)
    return [mid_v] if mid_v > -70.0 else []


def evaluate_dialogue_consistency(momentary_lufs: dict[float, float], global_cues: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-line LUFS distribution across dialogue/VO cues."""
    if not global_cues:
        return {
            "count": 0,
            "median_lufs": None,
            "min_lufs": None,
            "max_lufs": None,
            "min_rel_median": None,
            "max_rel_median": None,
            "spread_lu": 0.0,
            "std_lu": 0.0,
            "rms_dev_from_median_lu": 0.0,
            "lines": [],
        }

    lines: list[dict[str, Any]] = []
    vals: list[float] = []
    for cue in global_cues:
        st = float(cue["start"])
        en = float(cue["end"])
        samples = _cue_samples(momentary_lufs, st, en)
        lufs = _power_mean_lufs(samples)
        vals.append(lufs)
        lines.append({
            "chapter": str(cue.get("chapter") or ""),
            "shot_id": str(cue.get("shot_id") or ""),
            "style": str(cue.get("style") or "Dialogue"),
            "text": str(cue.get("text") or ""),
            "start": round(st, 3),
            "end": round(en, 3),
            "lufs": round(lufs, 3),
        })

    med = float(statistics.median(vals))
    vmin = min(vals)
    vmax = max(vals)
    mean_v = sum(vals) / len(vals)
    std_v = math.sqrt(sum((v - mean_v) ** 2 for v in vals) / len(vals))
    rms_med = math.sqrt(sum((v - med) ** 2 for v in vals) / len(vals))
    return {
        "count": len(vals),
        "median_lufs": round(med, 3),
        "min_lufs": round(vmin, 3),
        "max_lufs": round(vmax, 3),
        "min_rel_median": round(vmin - med, 3),
        "max_rel_median": round(vmax - med, 3),
        "spread_lu": round(vmax - vmin, 3),
        "std_lu": round(std_v, 3),
        "rms_dev_from_median_lu": round(rms_med, 3),
        "lines": lines,
    }


def reconcile_subtitles_with_bundles(
    chapters_cues: list[dict[str, Any]],
    bundle_dirs: Mapping[str, Path],
) -> list[VerifyFinding]:
    """Compare chapter subtitle cues against `<d>` tags in source `h3_prompts.json` bundles."""
    findings: list[VerifyFinding] = []
    for ch_rec in chapters_cues:
        cid = str(ch_rec.get("chapter") or "").zfill(2)
        bdir = bundle_dirs.get(cid)
        if bdir is None or not (Path(bdir) / "h3_prompts.json").is_file():
            findings.append(VerifyFinding("error", "MISSING_SOURCE_BUNDLE", f"Missing source h3_prompts.json for Ch{cid}"))
            continue
        shots = json.loads((Path(bdir) / "h3_prompts.json").read_text(encoding="utf-8")).get("shots", [])
        cue_by_shot: dict[str, str] = {}
        for c in ch_rec.get("subtitle_cues") or []:
            sid = str(c["shot_id"]).zfill(2)
            cue_by_shot[sid] = cue_by_shot.get(sid, "") + str(c.get("text") or "")

        for s in shots:
            digits = re.findall(r"\d+", str(s.get("id") or ""))
            sid = digits[-1].zfill(2) if digits else str(s.get("id") or "")
            script_text = "".join(extract_dialogue_tags(str(s.get("h3_prompt") or "")))
            sub_text = cue_by_shot.get(sid, "")
            if normalize_dialogue_text(script_text) != normalize_dialogue_text(sub_text):
                findings.append(
                    VerifyFinding(
                        "error",
                        "SUBTITLE_SCRIPT_MISMATCH",
                        f"Ch{cid} S{sid} subtitles diverge from source <d> script",
                        {"chapter": cid, "shot_id": sid, "script": script_text, "subtitles": sub_text},
                    )
                )
    return findings


def _overlaps_allowed(w: dict[str, float], allowed: list[tuple[float, float]], tol: float) -> bool:
    for a0, a1 in allowed:
        if w["start"] >= (a0 - tol) and w["end"] <= (a1 + tol):
            return True
    return False


def verify_master_release(
    mp4_path: Path,
    *,
    profile: str = "archive",
    expected_frames: int | None = None,
    expected_width: int = 1920,
    expected_height: int = 1080,
    expected_fps: float = 24.0,
    title_card_windows: list[tuple[float, float]] | None = None,
    reference_video_mp4: Path | None = None,
    global_cues: list[dict[str, Any]] | None = None,
    chapters_cues: list[dict[str, Any]] | None = None,
    bundle_dirs: Mapping[str, Path] | None = None,
    min_lufs: float | None = None,
    max_lufs: float | None = None,
    max_true_peak_dbtp: float | None = None,
    strict_freeze: bool = False,
    window_tolerance_sec: float = 1.0,
    ffmpeg_bin: str | None = None,
) -> MasterVerifyReport:
    """Run all automated stream, anomaly, loudness, MD5, and script-reconciliation gates."""
    p = Path(mp4_path)
    errors: list[VerifyFinding] = []
    warnings: list[VerifyFinding] = []

    if profile not in ("archive", "web"):
        raise ValueError(f"Unsupported verification profile: {profile!r}")

    eff_min_lufs = (-16.5 if profile == "web" else None) if min_lufs is None else min_lufs
    eff_max_lufs = (-15.5 if profile == "web" else None) if max_lufs is None else max_lufs
    eff_max_tp = (-1.0 if profile == "web" else -5.0) if max_true_peak_dbtp is None else max_true_peak_dbtp

    vprobe = probe_video_stream(p)
    if not vprobe["exists"]:
        errors.append(VerifyFinding("error", "MISSING_FILE", f"Target MP4 does not exist: {p}"))
        return MasterVerifyReport(MASTER_VERIFY_SCHEMA, str(p), profile, False, errors, warnings)

    if (vprobe["width"], vprobe["height"]) != (expected_width, expected_height):
        errors.append(
            VerifyFinding(
                "error",
                "RESOLUTION_MISMATCH",
                f"Expected {expected_width}x{expected_height}, got {vprobe['width']}x{vprobe['height']}",
                vprobe,
            )
        )
    if abs(float(vprobe["fps"]) - expected_fps) >= 0.05:
        errors.append(
            VerifyFinding("error", "FPS_MISMATCH", f"Expected {expected_fps} fps, got {vprobe['fps']}", vprobe)
        )
    if expected_frames is not None and vprobe["frames"] != expected_frames:
        errors.append(
            VerifyFinding(
                "error",
                "FRAME_COUNT_MISMATCH",
                f"Expected {expected_frames} frames, got {vprobe['frames']}",
                vprobe,
            )
        )

    md5_info: dict[str, Any] = {}
    if reference_video_mp4 is not None:
        ref_md5 = compute_video_stream_md5(Path(reference_video_mp4), ffmpeg_bin=ffmpeg_bin)
        out_md5 = compute_video_stream_md5(p, ffmpeg_bin=ffmpeg_bin)
        md5_info = {"reference_md5": ref_md5, "target_md5": out_md5, "match": ref_md5 == out_md5}
        if ref_md5 != out_md5:
            errors.append(
                VerifyFinding("error", "VIDEO_MD5_MISMATCH", "Output video bitstream differs from reference MP4", md5_info)
            )

    scan = scan_media_anomalies(p, ffmpeg_bin=ffmpeg_bin)
    total_dur = scan["container_dur_sec"] or vprobe["dur_sec"]
    allowed_cards = list(title_card_windows or [])

    for bw in scan["black_windows"]:
        is_trailing_fade = (bw["end"] >= total_dur - 0.2) and (bw["duration"] <= 1.5)
        if is_trailing_fade:
            continue
        if not _overlaps_allowed(bw, allowed_cards, window_tolerance_sec):
            errors.append(
                VerifyFinding(
                    "error",
                    "UNEXPECTED_BLACK_SCREEN",
                    f"Unexpected black screen {bw['start']:.2f}s–{bw['end']:.2f}s ({bw['duration']:.2f}s)",
                    bw,
                )
            )

    for sw in scan["silence_windows"]:
        if not _overlaps_allowed(sw, allowed_cards, window_tolerance_sec):
            errors.append(
                VerifyFinding(
                    "error",
                    "UNEXPECTED_AUDIO_SILENCE",
                    f"Unexpected silence {sw['start']:.2f}s–{sw['end']:.2f}s ({sw['duration']:.2f}s)",
                    sw,
                )
            )

    for fw in scan["freeze_windows"]:
        if _overlaps_allowed(fw, allowed_cards, window_tolerance_sec):
            continue
        finding = VerifyFinding(
            "error" if strict_freeze else "warning",
            "UNEXPECTED_VIDEO_FREEZE",
            f"Frozen frames {fw['start']:.2f}s–{fw['end']:.2f}s ({fw['duration']:.2f}s)",
            fw,
        )
        (errors if strict_freeze else warnings).append(finding)

    max_drift_sec = 0.0
    if allowed_cards:
        for _, c_end in allowed_cards:
            matched_ends = [w["end"] for w in scan["silence_windows"] if abs(w["end"] - c_end) <= window_tolerance_sec]
            if matched_ends:
                drift = min(abs(e - c_end) for e in matched_ends)
                max_drift_sec = max(max_drift_sec, drift)
        if max_drift_sec > 0.15:
            errors.append(
                VerifyFinding(
                    "error",
                    "BOUNDARY_SYNC_DRIFT",
                    f"Title-card boundary audio/video drift {max_drift_sec:.3f}s exceeds 0.150s",
                    {"max_boundary_drift_sec": round(max_drift_sec, 4)},
                )
            )

    loudness = scan["loudness"]
    i_lufs = float(loudness["integrated_lufs"])
    tp_dbtp = float(loudness["true_peak_dbtp"])
    if eff_min_lufs is not None and i_lufs < eff_min_lufs:
        errors.append(
            VerifyFinding("error", "LOUDNESS_TOO_LOW", f"Integrated loudness {i_lufs:.1f} < {eff_min_lufs:.1f} LUFS", loudness)
        )
    elif profile == "archive" and min_lufs is None and i_lufs < -30.0:
        warnings.append(
            VerifyFinding("warning", "ARCHIVE_LOUDNESS_LOW", f"Archive loudness {i_lufs:.1f} LUFS is quiet for web playback", loudness)
        )
    if eff_max_lufs is not None and i_lufs > eff_max_lufs:
        errors.append(
            VerifyFinding("error", "LOUDNESS_TOO_HIGH", f"Integrated loudness {i_lufs:.1f} > {eff_max_lufs:.1f} LUFS", loudness)
        )
    if eff_max_tp is not None and tp_dbtp > eff_max_tp:
        errors.append(
            VerifyFinding("error", "TRUE_PEAK_EXCEEDED", f"True peak {tp_dbtp:.1f} > {eff_max_tp:.1f} dBTP", loudness)
        )

    dlg_summary: dict[str, Any] | None = None
    ear_points: list[dict[str, Any]] = []
    if global_cues:
        dlg_full = evaluate_dialogue_consistency(scan["momentary_lufs"], global_cues)
        dlg_lines = dlg_full.pop("lines")
        dlg_summary = dlg_full
        if dlg_lines:
            quietest = min(dlg_lines, key=lambda x: x["lufs"])
            loudest = max(dlg_lines, key=lambda x: x["lufs"])
            ear_points.append({"role": "quietest_line", "audited_by_human": False, **quietest})
            ear_points.append({"role": "loudest_line", "audited_by_human": False, **loudest})

    if scan["momentary_lufs"]:
        t_peak, m_peak = max(scan["momentary_lufs"].items(), key=lambda kv: kv[1])
        ear_points.append({
            "role": "loudest_moment",
            "audited_by_human": False,
            "t_sec": t_peak,
            "momentary_lufs": m_peak,
        })

    if chapters_cues and bundle_dirs is not None:
        for finding in reconcile_subtitles_with_bundles(chapters_cues, bundle_dirs):
            (errors if finding.severity == "error" else warnings).append(finding)

    metrics: dict[str, Any] = {
        "video": vprobe,
        "loudness": loudness,
        "black_window_count": len(scan["black_windows"]),
        "silence_window_count": len(scan["silence_windows"]),
        "freeze_window_count": len(scan["freeze_windows"]),
        "max_boundary_drift_sec": round(max_drift_sec, 4),
    }
    if md5_info:
        metrics["video_md5"] = md5_info
    if dlg_summary is not None:
        metrics["dialogue_consistency"] = dlg_summary

    return MasterVerifyReport(
        schema=MASTER_VERIFY_SCHEMA,
        media_path=str(p),
        profile=profile,
        ok=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
        human_ear_check_points=ear_points,
    )


def _resolve_bundle_dirs(bundle_root: Path, chapters: list[dict[str, Any]]) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    root = Path(bundle_root)
    for ch in chapters:
        cid = str(ch["chapter"]).zfill(2)
        explicit = ch.get("bundle_dir")
        candidates = [
            Path(explicit) if explicit else None,
            root / f"xinhuo-ji-ch{cid}" / f"h3-ch{cid}-v3",
            root / f"xinhuo-ji-v2-ch{cid}" / f"h3-ch{cid}-v3",
            root / f"ch{cid}",
        ]
        found = next((c for c in candidates if c is not None and (c / "h3_prompts.json").is_file()), None)
        if found is not None:
            resolved[cid] = found
    return resolved


def verify_series_audit(
    series_audit_path: Path,
    *,
    media_override: Path | None = None,
    profile: str = "archive",
    reference_video_mp4: Path | None = None,
    bundle_root: Path | None = None,
    card_duration_sec: float = 2.0,
    strict_freeze: bool = False,
    ffmpeg_bin: str | None = None,
) -> MasterVerifyReport:
    """Verify a full multi-chapter feature MP4 from `v3_series_full_audit.json` or `series_timeline.json`."""
    audit_p = Path(series_audit_path)
    data = json.loads(audit_p.read_text(encoding="utf-8"))
    chapters = data["chapters"]

    target_mp4 = Path(media_override) if media_override is not None else Path(
        data.get("full_master", {}).get("path") or data.get("master_mp4") or ""
    )
    expected_frames = data.get("total_frames_including_cards") or data.get("total_frames")

    card_windows: list[tuple[float, float]] = []
    global_cues: list[dict[str, Any]] = []
    cur = 0.0
    for ch in chapters:
        cid = str(ch["chapter"]).zfill(2)
        card_dur = float(ch.get("title_card_dur_sec", card_duration_sec))
        card_start = cur
        ch_start = cur + card_dur
        if card_dur > 0:
            card_windows.append((card_start, ch_start))
        for c in ch.get("subtitle_cues") or []:
            g_st = float(c["global_start"])
            g_en = float(c["global_end"])
            # Distinguish chapter-local cues vs already series-global cues in series_timeline.json
            if "ch_global_start" in c:
                abs_st = float(c["ch_global_start"])
                abs_en = float(c["ch_global_end"])
            else:
                abs_st = ch_start + g_st
                abs_en = ch_start + g_en
            global_cues.append({
                "chapter": cid,
                "shot_id": str(c["shot_id"]).zfill(2),
                "style": str(c.get("style") or "Dialogue"),
                "text": str(c.get("text") or ""),
                "start": abs_st,
                "end": abs_en,
            })
        cur = ch_start + float(ch["dur_sec"])

    bundle_dirs = _resolve_bundle_dirs(Path(bundle_root), chapters) if bundle_root is not None else None
    return verify_master_release(
        target_mp4,
        profile=profile,
        expected_frames=int(expected_frames) if expected_frames is not None else None,
        title_card_windows=card_windows,
        reference_video_mp4=reference_video_mp4,
        global_cues=global_cues,
        chapters_cues=chapters if bundle_dirs is not None else None,
        bundle_dirs=bundle_dirs,
        strict_freeze=strict_freeze,
        ffmpeg_bin=ffmpeg_bin,
    )


def verify_chapter_speech(chapter_dir: Path, **kwargs: Any):
    """Lazy-delegating wrapper for Gate 2 local VAD+ASR chapter speech verification."""
    from .speech_audit import audit_chapter_speech

    return audit_chapter_speech(chapter_dir, **kwargs)
