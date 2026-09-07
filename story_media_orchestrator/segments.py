"""Normalize mixed image/video shots for the existing preview timeline."""
from pathlib import Path
import subprocess
import math


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


def render_segments(ffmpeg, shots, directory):
    normalize_durations(shots)
    segments = []
    filters = "fps=25,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p"
    for index, shot in enumerate(shots):
        video = shot.assets.get("video") if shot.mode == "render" else None
        source = video or shot.assets["image"]
        target = Path(directory) / f"{index:04d}.mp4"
        command = [ffmpeg, "-y", "-xerror"]
        if not video:
            command += ["-loop", "1"]
        effects = filters
        motion = getattr(shot, 'motion', 'none')
        transition = getattr(shot, 'transition', 'cut')
        if motion not in {'none', 'push_in', 'pan'} or transition not in {'cut', 'fade'}:
            raise ValueError('unsupported shot motion or transition')
        if not video and motion != 'none':
            count = max(1, round(shot.duration * 25) - 1)
            zoom = f'1+0.12*on/{count}' if motion == 'push_in' else '1.12'
            x = '(iw-iw/zoom)/2' if motion == 'push_in' else f'(iw-iw/zoom)*on/{count}'
            effects += f",zoompan=z='{zoom}':x='{x}':y='(ih-ih/zoom)/2':d=1:s=1280x720:fps=25"
        if video:
            effects += f',tpad=stop_mode=clone:stop_duration={shot.duration}'
        if transition == 'fade':
            fade = min(0.35, shot.duration / 4)
            effects += f',fade=t=in:st=0:d={fade},fade=t=out:st={shot.duration-fade}:d={fade}'
        command += ["-i", str(source), "-an", "-vf", effects,
                    "-t", str(shot.duration), "-c:v", "libx264", str(target)]
        subprocess.run(command, check=True, capture_output=True)
        segments.append(target)
    return segments


def align_audio_segments(ffmpeg, shots, directory, *, use_video_audio=True):
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
                        "-af", "apad", "-t", str(shot.duration), "-ar", "16000", "-ac", "1",
                        "-c:a", "pcm_s16le", str(audio_target)], check=True, capture_output=True)
        audio_segments.append(audio_target)
    return audio_segments
