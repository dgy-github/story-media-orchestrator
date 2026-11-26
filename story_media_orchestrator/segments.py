"""Normalize mixed image/video shots for the automated editing timeline."""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

from .editing import (
    MOTIONS,
    TRANSITIONS,
    XFADE_TRANSITIONS,
    probe_duration,
    resolve_canvas_size,
    resolve_safe_shot_transition,
    shot_has_character_presence,
    snapped_xfade_overlap,
)


def normalize_durations(shots):
    """Share cumulative 25fps boundaries across video, narration and captions."""
    seconds, previous = 0.0, 0
    for shot in shots:
        if not math.isfinite(shot.duration) or shot.duration <= 0:
            raise ValueError('shot duration must be positive and finite')
        seconds += shot.duration
        boundary = max(previous + 1, round(seconds * 25))
        shot.duration = (boundary - previous) / 25
        previous = boundary


def _motion_zoompan(motion: str, duration: float, width: int, height: int) -> str:
    count = max(1, round(duration * 25) - 1)
    if motion == "push_in":
        z, x, y = f"1+0.12*on/{count}", "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"
    elif motion == "pull_out":
        z, x, y = f"1.12-0.12*on/{count}", "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"
    elif motion == "pan":
        z, x, y = "1.12", f"(iw-iw/zoom)*on/{count}", "(ih-ih/zoom)/2"
    elif motion == "pan_left":
        z, x, y = "1.12", f"(iw-iw/zoom)*(1-on/{count})", "(ih-ih/zoom)/2"
    elif motion == "pan_up":
        z, x, y = "1.12", "(iw-iw/zoom)/2", f"(ih-ih/zoom)*(1-on/{count})"
    elif motion == "ken_burns":
        z, x, y = f"1+0.12*on/{count}", f"(iw-iw/zoom)*(0.2+0.6*on/{count})", f"(ih-ih/zoom)*(0.2+0.6*on/{count})"
    else:
        return ""
    return f"zoompan=z='{z}':x='{x}':y='{y}':d=1:s={width}x{height}:fps=25"


def _impact_shake_filter(duration: float, width: int, height: int, *, impact_ratio: float = 0.42) -> str:
    ow = max(width + 16, int(round(width * 1.06)) // 2 * 2)
    oh = max(height + 12, int(round(height * 1.06)) // 2 * 2)
    cx = (ow - width) // 2
    cy = (oh - height) // 2
    ax = max(4, min(cx, int(round(width * 0.018))))
    ay = max(3, min(cy, int(round(height * 0.018))))
    clamped_ratio = max(0.10, min(0.85, float(impact_ratio)))
    f0 = max(0, int(round((duration * clamped_ratio - 0.04) * 25)))
    f1 = max(f0 + 2, int(round(min(duration, duration * clamped_ratio + 0.18) * 25)))
    x_expr = f"{cx}+if(between(n,{f0},{f1}),{ax}*sin(n*2.7),0)"
    y_expr = f"{cy}+if(between(n,{f0},{f1}),{ay}*cos(n*3.1),0)"
    return f"scale={ow}:{oh},crop=w={width}:h={height}:x='{x_expr}':y='{y_expr}'"


def render_segments(
    ffmpeg,
    shots,
    directory,
    *,
    aspect_ratio: str = "16:9",
    scale_mode: str = "contain",
    alignment_mode: str = "pad",
    character_bible=None,
):
    normalize_durations(shots)
    width, height = resolve_canvas_size(aspect_ratio)
    if scale_mode not in {"contain", "cover", "blur_pad"}:
        raise ValueError(f"unsupported scale_mode: {scale_mode}")
    if alignment_mode not in {"pad", "elastic"}:
        raise ValueError(f"unsupported alignment_mode: {alignment_mode}")

    Path(directory).mkdir(parents=True, exist_ok=True)
    segments = []
    for index, shot in enumerate(shots):
        video = shot.assets.get("video") if shot.mode == "render" else None
        source = video or shot.assets["image"]
        target = Path(directory) / f"{index:04d}.mp4"
        motion = getattr(shot, "motion", "none")
        raw_transition = getattr(shot, "transition", "cut")
        if motion not in MOTIONS or raw_transition not in TRANSITIONS:
            raise ValueError("unsupported shot motion or transition")
        next_shot = shots[index + 1] if index + 1 < len(shots) else None
        transition = resolve_safe_shot_transition(
            shot, next_shot, character_bible=character_bible
        )

        overlap = 0.0
        if index + 1 < len(shots) and transition in XFADE_TRANSITIONS:
            overlap = snapped_xfade_overlap(shot.duration, shots[index + 1].duration)
        visual_duration = shot.duration + overlap

        filter_chains: list[str] = []
        if scale_mode == "blur_pad":
            filter_chains.append(
                "[0:v]fps=25,split=2[fg][bg];"
                f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},gblur=sigma=24,eq=brightness=-0.12[bgp];"
                f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[fgp];"
                "[bgp][fgp]overlay=(W-w)/2:(H-h)/2,setsar=1,format=yuv420p[scaled]"
            )
            cur = "[scaled]"
        elif scale_mode == "cover":
            filter_chains.append(
                f"[0:v]fps=25,scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1,format=yuv420p[scaled]"
            )
            cur = "[scaled]"
        else:
            filter_chains.append(
                f"[0:v]fps=25,scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p[scaled]"
            )
            cur = "[scaled]"

        if not video and motion != "none":
            zp = _motion_zoompan(motion, visual_duration, width, height)
            filter_chains.append(f"{cur}{zp}[mot]")
            cur = "[mot]"

        if video:
            ctx_map = getattr(shot, "scene_context", None) if isinstance(getattr(shot, "scene_context", None), dict) else {}
            raw_dur = ctx_map.get("_video_duration")
            if alignment_mode == "elastic" and not raw_dur:
                raw_dur = probe_duration(ffmpeg, video)
            trim_start = ctx_map.get("video_trim_start")
            trim_end = ctx_map.get("video_trim_end")
            if (
                trim_start is None
                and trim_end is None
                and ctx_map.get("auto_trim_tail_stall") is True
                and raw_dur
                and float(raw_dur) > visual_duration + 0.35
            ):
                from .frame_decomposer import compute_active_motion_window

                win = compute_active_motion_window(video, visual_duration)
                if win.get("available") and win.get("trimmed"):
                    trim_start = float(win["start_s"])
                    trim_end = float(win["end_s"])
            if (
                isinstance(trim_start, (int, float))
                and isinstance(trim_end, (int, float))
                and float(trim_end) > float(trim_start) + 0.15
            ):
                ts0 = max(0.0, float(trim_start))
                ts1 = float(trim_end)
                if raw_dur and float(raw_dur) > 0:
                    ts1 = min(float(raw_dur), ts1)
                if ts1 > ts0 + 0.15:
                    filter_chains.append(
                        f"{cur}trim=start={ts0:.4f}:end={ts1:.4f},setpts=PTS-STARTPTS[trm]"
                    )
                    cur = "[trm]"
                    raw_dur = ts1 - ts0
                    ctx_map["_applied_video_trim"] = [round(ts0, 3), round(ts1, 3)]
            if alignment_mode == "elastic" and raw_dur and raw_dur > 0:
                ratio = visual_duration / float(raw_dur)
                if ratio <= 1.6:
                    filter_chains.append(
                        f"{cur}setpts={ratio:.4f}*PTS,fps=25,tpad=stop_mode=clone:stop_duration=0.25[el]"
                    )
                    cur = "[el]"
                elif shot_has_character_presence(shot, character_bible):
                    stretched_dur = 1.6 * float(raw_dur)
                    rem_dur = max(0.25, visual_duration - stretched_dur)
                    stretched_frames = max(1, round(stretched_dur * 25))
                    rem_frames = max(1, round(rem_dur * 25))
                    filter_chains.append(
                        f"{cur}setpts=1.6000*PTS,fps=25,tpad=stop_mode=clone:stop_duration={rem_dur:.4f},"
                        f"zoompan=z='if(lte(in,{stretched_frames}),1,min(1.05,1+0.05*(in-{stretched_frames})/{rem_frames}))':"
                        f"x='(iw-iw/zoom)/2':y='(ih-ih/zoom)/2':d=1:s={width}x{height}:fps=25[el]"
                    )
                    if isinstance(getattr(shot, "scene_context", None), dict):
                        shot.scene_context["_elastic_character_hold"] = True
                    cur = "[el]"
                else:
                    cycles = max(1, math.ceil(visual_duration / (2.0 * float(raw_dur))))
                    filter_chains.append(
                        f"{cur}split=2[fwd_src][rev_src];"
                        "[rev_src]reverse[rev_clip];"
                        "[fwd_src][rev_clip]concat=n=2:v=1:a=0[boom_unit]"
                    )
                    if cycles == 1:
                        filter_chains.append(f"[boom_unit]fps=25,tpad=stop_mode=clone:stop_duration=0.25[el]")
                    else:
                        split_labels = "".join(f"[b{k}]" for k in range(cycles))
                        filter_chains.append(
                            f"[boom_unit]split={cycles}{split_labels};"
                            f"{split_labels}concat=n={cycles}:v=1:a=0,fps=25,tpad=stop_mode=clone:stop_duration=0.25[el]"
                        )
                    cur = "[el]"
            else:
                filter_chains.append(f"{cur}tpad=stop_mode=clone:stop_duration={visual_duration}[el]")
                cur = "[el]"

            if motion != "none":
                zp = _motion_zoompan(motion, visual_duration, width, height)
                if zp:
                    filter_chains.append(f"{cur}{zp}[vmot]")
                    cur = "[vmot]"

        ctx_dict = getattr(shot, "scene_context", None)
        if isinstance(ctx_dict, dict) and ctx_dict.get("impact_beat"):
            impact_ratio = float(ctx_dict.get("impact_at_ratio", 0.42) or 0.42)
            shk = _impact_shake_filter(visual_duration, width, height, impact_ratio=impact_ratio)
            filter_chains.append(f"{cur}{shk}[shk]")
            cur = "[shk]"

        if transition == "fade":
            fade = min(0.35, visual_duration / 4)
            filter_chains.append(
                f"{cur}fade=t=in:st=0:d={fade},fade=t=out:st={visual_duration - fade}:d={fade},setsar=1,format=yuv420p[vout]"
            )
        else:
            filter_chains.append(f"{cur}setsar=1,format=yuv420p[vout]")

        command = [ffmpeg, "-y", "-xerror"]
        if not video:
            command += ["-loop", "1"]
        command += [
            "-i",
            str(source),
            "-an",
            "-filter_complex",
            ";".join(filter_chains),
            "-map",
            "[vout]",
            "-t",
            str(visual_duration),
            "-c:v",
            "libx264",
            str(target),
        ]
        subprocess.run(command, check=True, capture_output=True)
        segments.append(target)
    return segments


def align_audio_segments(ffmpeg, shots, directory, *, use_video_audio=True, sample_rate=16000, channels=1):
    audio_segments = []
    for index, shot in enumerate(shots):
        audio_target = Path(directory) / f"{index:04d}.wav"
        source = shot.assets['audio']
        video = shot.assets.get('video') if use_video_audio and shot.mode == 'render' else None
        if video:
            probe = subprocess.run([ffmpeg, '-hide_banner', '-i', str(video)], capture_output=True)
            if b'Audio:' in probe.stderr:
                source = video
        subprocess.run([ffmpeg, "-y", "-xerror", "-i", str(source), "-map", "0:a:0",
                        "-af", "apad", "-t", str(shot.duration), "-ar", str(sample_rate), "-ac", str(channels),
                        "-c:a", "pcm_s16le", str(audio_target)], check=True, capture_output=True)
        audio_segments.append(audio_target)
    return audio_segments
