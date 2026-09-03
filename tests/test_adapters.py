import base64
import tempfile
import unittest
from pathlib import Path

from story_media_orchestrator import ArtifactRegistry, StoryCampaignAdapter, StoryImageAdapter


class _ImageWorkflow:
    def build_production_plan(self, scene, spans, candidate_count=1):
        return {"schema": "image-production-plan/v1", "candidates": [{"request_id": "r1"}]}
    def finalize_candidate(self, plan, request_id, quality):
        return {**plan, "final_request": {"request_id": request_id}, "evaluation": quality}


class _Provider:
    def generate(self, request):
        return {"content_base64": base64.b64encode(b"png").decode()}


class AdapterTests(unittest.TestCase):
    def test_story_adapter_accepts_wrapped_package(self):
        adapter = StoryCampaignAdapter(lambda _: {"package": {"schema": "story-package/v1"}})
        self.assertEqual(adapter.run({})["schema"], "story-package/v1")

    def test_image_adapter_persists_frame(self):
        with tempfile.TemporaryDirectory() as d:
            result = StoryImageAdapter(_ImageWorkflow(), _Provider(), ArtifactRegistry(Path(d))).run(
                {"action": "look"}, ["scene:1"], quality={"passed": True})
            self.assertEqual(result["first_frame_ref"], result["last_frame_ref"])
            self.assertTrue(result["first_frame_ref"].startswith("artifact://sha256/"))


if __name__ == "__main__":
    unittest.main()
