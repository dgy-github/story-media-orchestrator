from pathlib import Path
import pytest
from story_media_orchestrator.registry import ArtifactRegistry


def test_video_preview_verifies_source_and_repairs_cached_copy(tmp_path):
    registry = ArtifactRegistry(tmp_path)
    content = b'\x00\x00\x00\x18ftypmp42' + b'video-bytes'
    ref = registry.put_bytes(content)
    target = Path(registry.video_preview(ref))
    assert target.suffix == '.mp4'
    assert target.read_bytes() == content
    target.write_bytes(b'corrupt-copy')
    assert Path(registry.video_preview(ref)).read_bytes() == content
    (tmp_path / ref.rsplit('/', 1)[-1]).write_bytes(b'corrupt-original')
    with pytest.raises(RuntimeError, match='integrity'):
        registry.video_preview(ref)


def test_video_preview_rejects_unknown_container(tmp_path):
    registry = ArtifactRegistry(tmp_path)
    with pytest.raises(ValueError, match='supported'):
        registry.video_preview(registry.put_bytes(b'not-a-video'))
