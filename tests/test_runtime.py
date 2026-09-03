import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from story_media_orchestrator import RuntimeConfig


class RuntimeConfigTests(unittest.TestCase):
    def test_reads_sibling_roots_and_artifact_root(self):
        with tempfile.TemporaryDirectory() as image, tempfile.TemporaryDirectory() as video:
            with mock.patch.dict(os.environ, {
                "STORY_IMAGE_AGENT_ROOT": image,
                "STORY_VIDEO_AGENT_ROOT": video,
                "STORY_MEDIA_ARTIFACT_ROOT": str(Path(image) / "artifacts"),
            }, clear=True):
                config = RuntimeConfig.from_environment()
            self.assertEqual(config.image_root, Path(image).resolve())
            self.assertEqual(config.video_root, Path(video).resolve())

    def test_requires_existing_sibling_roots(self):
        with mock.patch.dict(os.environ, {
            "STORY_IMAGE_AGENT_ROOT": "Z:/missing-image-agent",
            "STORY_VIDEO_AGENT_ROOT": "Z:/missing-video-agent",
        }, clear=True), self.assertRaises(RuntimeError):
            RuntimeConfig.from_environment()


if __name__ == "__main__":
    unittest.main()
