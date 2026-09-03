import os
import tempfile
import unittest
from unittest import mock
from story_media_orchestrator import OrchestratorConfig

class ConfigTests(unittest.TestCase):
    def test_model_defaults_and_overrides(self):
        with tempfile.TemporaryDirectory() as image, tempfile.TemporaryDirectory() as video:
            with mock.patch.dict(os.environ, {"STORY_IMAGE_AGENT_ROOT": image, "STORY_VIDEO_AGENT_ROOT": video,
                                              "STORY_VIDEO_TURBO": "false", "STORY_VIDEO_STEPS": "20",
                                              "STORY_IMAGE_MODEL": "test-image"}, clear=True):
                cfg = OrchestratorConfig.from_environment()
            self.assertEqual(cfg.models.image_model, "test-image")
            self.assertFalse(cfg.models.video_turbo)
            self.assertEqual(cfg.models.video_steps, 20)

    def test_invalid_turbo_rejected(self):
        with tempfile.TemporaryDirectory() as image, tempfile.TemporaryDirectory() as video:
            with mock.patch.dict(os.environ, {"STORY_IMAGE_AGENT_ROOT": image, "STORY_VIDEO_AGENT_ROOT": video,
                                              "STORY_VIDEO_TURBO": "yes"}, clear=True), self.assertRaises(ValueError):
                OrchestratorConfig.from_environment()

if __name__ == "__main__": unittest.main()
