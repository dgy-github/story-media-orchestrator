"""CLI/project/model-adapter/FFmpeg recovery integration, without cloud calls."""
from pathlib import Path
import subprocess
from unittest.mock import Mock

from story_media_orchestrator.cli import main
from story_media_orchestrator.manifest import ProjectManifest
from story_media_orchestrator.local_media import find_ffmpeg
from story_media_orchestrator.registry import ArtifactRegistry
from story_media_orchestrator.project_video import ProjectVideoProvider
from story_media_orchestrator.adapters import StoryVideoAdapter


def test_cli_render_fallback_reload_recovery_and_offline_assembly(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'story-video-agent'))
    from story_video_agent import VideoPromptWorkflow
    ffmpeg = find_ffmpeg()
    model_clip = tmp_path / 'model.mp4'
    subprocess.run([ffmpeg, '-y', '-f', 'lavfi', '-i', 'color=red:s=1280x720:r=25',
                    '-t', '5', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(model_clip)],
                   check=True, capture_output=True)
    project = tmp_path / 'project'
    main(['create', str(project), 'The courier opens the door.'])
    registry = ArtifactRegistry(project / 'artifacts')
    ref = registry.put_bytes(model_clip.read_bytes())
    adapter = StoryVideoAdapter(VideoPromptWorkflow('project'), comfy=object())
    execute = Mock(side_effect=[TimeoutError('poll interrupted'), {'state':'succeeded', 'artifact_ref':ref}])
    adapter._execute_comfy = execute
    build = Mock(return_value=ProjectVideoProvider(adapter, registry))
    monkeypatch.setattr('story_media_orchestrator.project_video.build_project_video_provider', build)
    main(['run', str(project), '--mode', 'render', '--real-preview', '--image-source', 'local'])
    failed = ProjectManifest.load(project / 'project.json')
    assert failed.shots[0].error.startswith('video fallback:')
    image = Path(failed.shots[0].assets['image'])
    audio = Path(failed.shots[0].assets['audio'])
    timestamps = image.stat().st_mtime_ns, audio.stat().st_mtime_ns
    main(['resume', str(project)])
    restored = ProjectManifest.load(project / 'project.json')
    assert restored.shots[0].mode == 'render' and restored.shots[0].error is None
    assert restored.shots[0].duration == 5
    assert timestamps == (image.stat().st_mtime_ns, audio.stat().st_mtime_ns)
    pixel = subprocess.run([ffmpeg, '-i', restored.output, '-vf', 'scale=1:1', '-frames:v', '1',
                            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], check=True, capture_output=True).stdout
    assert pixel[0] > 200 and pixel[2] < 40
    frames = subprocess.run([ffmpeg, '-i', restored.output, '-vf', 'scale=1:1',
                             '-f', 'rawvideo', '-pix_fmt', 'gray', '-'], check=True, capture_output=True).stdout
    assert len(frames) == 125, 'five-second model video was truncated by shorter silent audio'
    build.side_effect = AssertionError('offline assembly must not initialize model')
    main(['resume', str(project)])
    assert execute.call_count == 2
