"""Real runtime entrypoint used by the Rust desktop shell."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
from .runtime import build_runtime_from_environment
from .adapters import HttpStoryCampaignAdapter, StoryImageAdapter, StoryVideoAdapter
from .registry import ArtifactRegistry
from .quality import evaluate_artifact

def main() -> int:
    payload = json.loads(sys.stdin.read() or "{}")
    base = Path(__file__).resolve().parents[1]
    os.environ.setdefault("STORY_CAMPAIGN_ROOT", r"D:\github_dgy\microcodex-short-drama-studio")
    os.environ.setdefault("STORY_IMAGE_AGENT_ROOT", r"D:\github_dgy\story-image-agent")
    os.environ.setdefault("STORY_VIDEO_AGENT_ROOT", r"D:\github_dgy\story-video-agent")
    os.environ.setdefault("STORY_MEDIA_ARTIFACT_ROOT", str(base / ".artifacts"))
    stage = payload.pop("_stage", "all") if isinstance(payload, dict) else "all"
    if stage == "all":
        result = build_runtime_from_environment().run(story_input=payload)
    elif stage == "story":
        result = HttpStoryCampaignAdapter(os.environ["STORY_SIDECAR_URL"], os.environ["STORY_SIDECAR_TOKEN"]).run(payload)
    else:
        image_root = Path(os.environ["STORY_IMAGE_AGENT_ROOT"]); video_root = Path(os.environ["STORY_VIDEO_AGENT_ROOT"])
        import sys; sys.path.insert(0, str(image_root)); sys.path.insert(0, str(video_root))
        registry = ArtifactRegistry(Path(os.environ["STORY_MEDIA_ARTIFACT_ROOT"]))
        if stage == "image":
            import story_image_agent as pkg
            result = StoryImageAdapter(pkg.ImagePromptWorkflow("story-media-orchestrator"), pkg.DashScopeImageProvider.from_nanocodex_config(), registry).run(payload["scene"], payload.get("source_spans", []))
        elif stage == "video":
            import story_video_agent as pkg
            result = StoryVideoAdapter(pkg.VideoPromptWorkflow("story-media-orchestrator"), comfy=pkg.ComfyUIAdapter.from_environment(), registry=registry).run(first_frame_ref=payload["first_frame_ref"], last_frame_ref=payload.get("last_frame_ref"), story_spans=payload.get("source_spans", []), scene=payload.get("scene", {}), prompt=payload.get("prompt"))
        else: raise ValueError(f"unknown stage: {stage}")
    if isinstance(result, dict):
        result["quality_evaluation"] = evaluate_artifact(result, stage)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
