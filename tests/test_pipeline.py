import unittest

from story_media_orchestrator import ArtifactRegistry, SingleSceneOrchestrator, OrchestrationError
import tempfile
from pathlib import Path


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def story(payload):
            self.calls.append("story")
            return {"schema": "story-package/v1", "scenes": [{"id": "s1", "source_spans": ["scene:1"]}]}

        def image(**kwargs):
            self.calls.append(("image", kwargs))
            return {"schema": "image-production-plan/v1", "final_artifacts": [
                "artifact://sha256/" + "a" * 64, "artifact://sha256/" + "b" * 64]}

        def video(**kwargs):
            self.calls.append(("video", kwargs))
            return {"schema": "video-generation-pipeline/v2", "stages": []}

        self.orchestrator = SingleSceneOrchestrator(story, image, video)

    def test_single_scene_wires_first_and_last_frames(self):
        result = self.orchestrator.run(story_input={})
        self.assertEqual(result["schema"], "story-media-run/v1")
        video_call = self.calls[-1][1]
        self.assertTrue(video_call["first_frame_ref"].endswith("a" * 64))
        self.assertTrue(video_call["last_frame_ref"].endswith("b" * 64))

    def test_missing_spans_fails_closed(self):
        def story(_):
            return {"schema": "story-package/v1", "scenes": [{"id": "s1"}]}
        runner = SingleSceneOrchestrator(story, self.orchestrator.image_agent, self.orchestrator.video_agent)
        with self.assertRaises(OrchestrationError):
            runner.run(story_input={})

    def test_artifact_registry_is_content_addressed_and_verifies_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = ArtifactRegistry(Path(directory))
            ref = registry.put_json({"schema": "story-package/v1", "id": "s1"})
            self.assertEqual(registry.get_json(ref)["id"], "s1")
            digest = ref.rsplit("/", 1)[-1]
            Path(directory, digest).write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                registry.get_json(ref)


if __name__ == "__main__":
    unittest.main()
