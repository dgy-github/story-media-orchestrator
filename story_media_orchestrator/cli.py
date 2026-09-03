"""Real runtime entrypoint used by the Rust desktop shell."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
from .runtime import build_runtime_from_environment

def main() -> int:
    payload = json.loads(sys.stdin.read() or "{}")
    base = Path(__file__).resolve().parents[1]
    os.environ.setdefault("STORY_CAMPAIGN_ROOT", r"D:\github_dgy\microcodex-short-drama-studio")
    os.environ.setdefault("STORY_IMAGE_AGENT_ROOT", r"D:\github_dgy\story-image-agent")
    os.environ.setdefault("STORY_VIDEO_AGENT_ROOT", r"D:\github_dgy\story-video-agent")
    os.environ.setdefault("STORY_MEDIA_ARTIFACT_ROOT", str(base / ".artifacts"))
    result = build_runtime_from_environment().run(story_input=payload)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
