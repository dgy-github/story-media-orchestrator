import unittest
from pathlib import Path

from story_media_orchestrator.ui import DEFAULTS, apply_internal_defaults, validate_ui_config


class UiConfigTests(unittest.TestCase):
    def test_paths_are_internal_defaults(self):
        config = {
            "MINIMAX_H3_COMFYUI_BASE_URL": "http://127.0.0.1:8000",
            "STORY_VIDEO_TURBO": "false",
            "STORY_VIDEO_STEPS": "20",
        }
        completed = apply_internal_defaults(config)
        self.assertEqual(completed["STORY_IMAGE_AGENT_ROOT"], DEFAULTS["STORY_IMAGE_AGENT_ROOT"])
        self.assertEqual(validate_ui_config(config), [])

    def test_explicit_missing_path_is_rejected(self):
        config = {
            "STORY_IMAGE_AGENT_ROOT": str(Path("Z:/definitely-missing")),
            "MINIMAX_H3_COMFYUI_BASE_URL": "http://127.0.0.1:8000",
            "STORY_VIDEO_TURBO": "false",
            "STORY_VIDEO_STEPS": "20",
        }
        self.assertTrue(any("STORY_IMAGE_AGENT_ROOT" in error for error in validate_ui_config(config)))


if __name__ == "__main__":
    unittest.main()
