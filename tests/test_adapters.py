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
            self.assertEqual(result["source_spans"], ["scene:1"])


if __name__ == "__main__":
    unittest.main()


def test_image_without_evaluation_remains_pending(tmp_path):
    from unittest.mock import Mock
    workflow = _ImageWorkflow()
    workflow.finalize_candidate = Mock(side_effect=AssertionError("must not approve unreviewed image"))
    result = StoryImageAdapter(workflow, _Provider(), ArtifactRegistry(tmp_path)).run({"action":"look"}, ["scene:1"])
    assert result["review_status"] == "pending"
    assert result["plan"]["evaluation"]["passed"] is None
    assert "final_request" not in result["plan"]
    assert result["frame_mode"] == "same_image"


def test_video_does_not_fabricate_quality():
    from unittest.mock import Mock
    from story_media_orchestrator.adapters import StoryVideoAdapter
    workflow = Mock()
    workflow.build_pipeline.return_value = {"schema":"video-generation-pipeline/v2"}
    result = StoryVideoAdapter(workflow).run(first_frame_ref="artifact://sha256/a", last_frame_ref=None, story_spans=["scene-1"], scene={"action":"walk"})
    assert workflow.build_pipeline.call_args.kwargs["quality"] is None
    assert result["review_status"] == "pending"

def test_candidate_batch_preserves_successes_when_one_fails(tmp_path):
    from unittest.mock import Mock
    workflow = Mock()
    workflow.build_production_plan.return_value = {
        "candidates": [{"request_id": f"r{i}"} for i in range(3)]}
    provider = Mock()
    provider.generate.side_effect = [
        {"content_base64": base64.b64encode(b"first").decode(), "mime_type": "image/png"},
        TimeoutError("private provider URL must not be echoed"),
        {"content_base64": base64.b64encode(b"third").decode(), "mime_type": "image/png"},
    ]
    registry = ArtifactRegistry(tmp_path)
    result = StoryImageAdapter(workflow, provider, registry).run(
        {"action": "walk"}, ["original-scene"], candidate_count=3, include_preview=True)
    assert result["status"] == "partial"
    assert len(result["candidate_results"]) == 3
    assert result["candidate_results"][1] == {"request_id": "r1", "status": "failed", "error": "TimeoutError", "error_detail": "等待超时，任务可能仍在运行；请恢复原批次"}
    assert registry.get_bytes(result["candidate_results"][2]["artifact_ref"]) == b"third"
    assert result["candidate_results"][0]["preview_url"].startswith("data:image/png;base64,")
    assert result["selected_request_id"] == "r0"
    workflow.finalize_candidate.assert_not_called()


def test_invalid_candidate_count_never_calls_provider(tmp_path):
    import pytest
    from unittest.mock import Mock
    provider = Mock()
    adapter = StoryImageAdapter(Mock(), provider, ArtifactRegistry(tmp_path))
    for count in [0, 9, True, 1.5, "2"]:
        with pytest.raises(ValueError, match="candidate_count"):
            adapter.run({}, ["scene"], candidate_count=count)
    provider.generate.assert_not_called()

def test_original_image_workflow_receives_negative_and_composition(tmp_path, monkeypatch):
    from pathlib import Path
    from unittest.mock import Mock
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'story-image-agent'))
    from story_image_agent import ImagePromptWorkflow
    provider = Mock()
    provider.generate.return_value = {'content_base64':base64.b64encode(b'image').decode()}
    StoryImageAdapter(ImagePromptWorkflow('test'), provider, ArtifactRegistry(tmp_path)).run(
        {'action':'walk','negative':'no collage','framing':'wide shot','continuity':'blue coat'}, ['scene'])
    request = provider.generate.call_args.args[0]
    assert request['negative_prompt'] == 'no collage'
    assert 'wide shot' in request['prompt']
    assert 'blue coat' in request['prompt']
