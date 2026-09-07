"""Unified CLI entrypoint."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from .manifest import ProjectManifest
from .preview import render_preview

def _legacy() -> int:
    from .runtime import build_runtime_from_environment
    from .adapters import HttpStoryCampaignAdapter, StoryImageAdapter, StoryVideoAdapter
    from .registry import ArtifactRegistry
    from .quality import evaluate_artifact
    payload = json.loads(sys.stdin.read() or '{}')
    base = Path(__file__).resolve().parents[1]
    os.environ.setdefault('STORY_CAMPAIGN_ROOT', r'D:\github_dgy\microcodex-short-drama-studio')
    os.environ.setdefault('STORY_IMAGE_AGENT_ROOT', r'D:\github_dgy\story-image-agent')
    os.environ.setdefault('STORY_VIDEO_AGENT_ROOT', r'D:\github_dgy\story-video-agent')
    os.environ.setdefault('STORY_MEDIA_ARTIFACT_ROOT', str(base / '.artifacts'))
    stage = payload.pop('_stage', 'all')
    if stage == 'all':
        result = build_runtime_from_environment().run(story_input=payload)
    elif stage == 'story':
        result = HttpStoryCampaignAdapter(os.environ['STORY_SIDECAR_URL'], os.environ['STORY_SIDECAR_TOKEN']).run(payload)
    else:
        image_root = Path(os.environ['STORY_IMAGE_AGENT_ROOT']); video_root = Path(os.environ['STORY_VIDEO_AGENT_ROOT'])
        sys.path[:0] = [str(image_root), str(video_root)]
        registry = ArtifactRegistry(Path(os.environ['STORY_MEDIA_ARTIFACT_ROOT']))
        if stage == 'image':
            import story_image_agent as pkg
            result = StoryImageAdapter(pkg.ImagePromptWorkflow('story-media-orchestrator'), pkg.DashScopeImageProvider.from_nanocodex_config(), registry).run(payload['scene'], payload.get('source_spans', []))
        elif stage == 'video':
            import story_video_agent as pkg
            result = StoryVideoAdapter(pkg.VideoPromptWorkflow('story-media-orchestrator'), comfy=pkg.ComfyUIAdapter.from_environment(), registry=registry).run(first_frame_ref=payload['first_frame_ref'], last_frame_ref=payload.get('last_frame_ref'), story_spans=payload.get('source_spans', []), scene=payload.get('scene', {}), prompt=payload.get('prompt'))
        else:
            raise ValueError(f'unknown stage: {stage}')
    if isinstance(result, dict): result['quality_evaluation'] = evaluate_artifact(result, stage)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0

def main(argv=None) -> int:
    if argv is None and len(sys.argv) == 1: return _legacy()
    parser = argparse.ArgumentParser(prog='story-media')
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create'); create.add_argument('project'); create.add_argument('story')
    for name in ('run', 'resume', 'retry', 'review'):
        p = sub.add_parser(name); p.add_argument('project'); p.add_argument('shot', nargs='?')
    args = parser.parse_args(argv); project = Path(args.project); manifest_path = project / 'project.json'
    if args.command == 'create':
        project.mkdir(parents=True, exist_ok=True); ProjectManifest.create(project.name, args.story).save(manifest_path); print(f'created {manifest_path}'); return 0
    manifest = ProjectManifest.load(manifest_path)
    if args.command == 'review':
        if not args.shot: raise SystemExit('review requires a shot id')
        shot = manifest.shot(args.shot); shot.review = 'approved'; manifest.save(manifest_path); print(f'approved {args.shot}'); return 0
    if args.command == 'retry' and args.shot: manifest.shot(args.shot).status = 'planned'
    output = render_preview(manifest, project); manifest.save(manifest_path); print(f'preview: {output}'); return 0

if __name__ == '__main__': raise SystemExit(main())
