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
    os.environ.setdefault('STORY_CAMPAIGN_ROOT', str(base.parent / 'microcodex-short-drama-studio'))
    os.environ.setdefault('STORY_IMAGE_AGENT_ROOT', str(base.parent / 'story-image-agent'))
    os.environ.setdefault('STORY_VIDEO_AGENT_ROOT', str(base.parent / 'story-video-agent'))
    os.environ.setdefault('STORY_MEDIA_ARTIFACT_ROOT', str(base / '.artifacts'))
    stage = payload.pop('_stage', 'all')
    if stage == 'all':
        result = build_runtime_from_environment().run(story_input=payload)
    elif stage == 'story':
        url = os.environ.get('STORY_SIDECAR_URL', '').strip()
        token = os.environ.get('STORY_SIDECAR_TOKEN', '')
        if not url or not token:
            raise RuntimeError('故事服务未连接：需要原故事运行时的服务地址与令牌，请先启动或配置原故事运行时')
        result = HttpStoryCampaignAdapter(url, token).run(payload)
    else:
        image_root = Path(os.environ['STORY_IMAGE_AGENT_ROOT']); video_root = Path(os.environ['STORY_VIDEO_AGENT_ROOT'])
        sys.path[:0] = [str(image_root), str(video_root)]
        registry = ArtifactRegistry(Path(os.environ['STORY_MEDIA_ARTIFACT_ROOT']))
        if stage == 'image':
            import story_image_agent as pkg
            provider = pkg.DashScopeImageProvider.from_nanocodex_config()
            if payload.get('image_size'):
                provider.size = provider._normalize_size(payload['image_size'])
            result = StoryImageAdapter(pkg.ImagePromptWorkflow('story-media-orchestrator'), provider, registry).run(payload['scene'], payload.get('source_spans', []), candidate_count=payload.get('candidate_count', 1), include_preview=True, batch_id=payload.get('batch_id'), prompt_revision=payload.get('prompt_revision'))
        elif stage == 'video' and payload.get('video_provider') == 'dashscope':
            from story_video_agent.dashscope import DashScopeWanAdapter
            from story_image_agent import DashScopeImageProvider
            from .wan_video import run_wan_video
            configured = DashScopeImageProvider.from_nanocodex_config()
            provider = DashScopeWanAdapter(configured.api_key, base_url=configured.base_url)
            result = run_wan_video(provider, registry, prompt=payload.get('prompt'),
                model=payload.get('video_model', 'wanx2.1-t2v-turbo'),
                duration_seconds=payload.get('duration_seconds', 5),
                size=payload.get('video_size', '1280*720'), negative_prompt=payload.get('negative_prompt'), batch_id=payload.get('batch_id'))
        elif stage == 'video':
            if payload.get('video_provider', 'comfyui') != 'comfyui':
                raise ValueError('unknown video provider')
            import story_video_agent as pkg
            from .config import ModelConfig
            turbo = payload.get('turbo', False)
            if not isinstance(turbo, bool):
                raise ValueError('turbo must be boolean')
            result = StoryVideoAdapter(pkg.VideoPromptWorkflow('story-media-orchestrator'), models=ModelConfig(video_turbo=turbo, video_steps=8 if turbo else 20), comfy=pkg.ComfyUIAdapter.from_environment(), registry=registry).run(first_frame_ref=payload['first_frame_ref'], last_frame_ref=payload.get('last_frame_ref'), story_spans=payload.get('source_spans', []), scene=payload.get('scene', {}), prompt=payload.get('prompt'), seed=payload.get('seed', 0), width=payload.get('width', 864), height=payload.get('height', 480), length=payload.get('length', 124), fps=payload.get('fps', 24.0))
        else:
            raise ValueError(f'unknown stage: {stage}')
    if stage == 'video' and result.get('execution', {}).get('state') == 'succeeded':
        try:
            result['output_file'] = registry.video_preview(result['execution']['artifact_ref'])
        except (ValueError, RuntimeError, OSError) as exc:
            result['playback_error'] = str(exc)
    if isinstance(result, dict): result['quality_evaluation'] = evaluate_artifact(result, stage)
    if stage == 'video' and result.get('execution', {}).get('state') == 'succeeded':
        from .video_result import persist_video_result
        result = persist_video_result(result, payload, registry)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0

def main(argv=None) -> int:
    # Desktop IPC writes UTF-8 bytes even on Windows GBK installations.
    # Decode strictly at the boundary rather than persisting surrogate escapes.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure') and not stream.isatty():
            stream.reconfigure(encoding='utf-8', errors='strict')
    if argv is None and len(sys.argv) == 1: return _legacy()
    parser = argparse.ArgumentParser(prog='story-media')
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create'); create.add_argument('project'); create.add_argument('story', nargs='?', default=''); create.add_argument('--create-stdin', action='store_true')
    for name in ('candidates', 'select', 'effects'):
        p = sub.add_parser(name)
        p.add_argument('project'); p.add_argument('shot')
        p.add_argument('--candidate-count', type=int, default=2)
        p.add_argument('--candidate-id')
        p.add_argument('--frame-role', choices=('image', 'last_frame'), default='image')
        p.add_argument('--motion', choices=('none', 'push_in', 'pan'), default='none')
        p.add_argument('--transition', choices=('cut', 'fade'), default='cut')
    for name in ('run', 'resume', 'retry', 'review'):
        p = sub.add_parser(name); p.add_argument('project'); p.add_argument('shot', nargs='?')
        if name in ('run', 'resume'): p.add_argument('--mode', choices=('preview', 'render'))
        if name in ('run', 'resume', 'retry'): p.add_argument('--real-preview', action='store_true', default=None)
        if name in ('run', 'resume', 'retry'):
            p.add_argument('--require-review', action='store_true')
            p.add_argument('--auto', action='store_true')
            p.add_argument('--image-source', choices=('local', 'dashscope'))
            p.add_argument('--tts-source', choices=('silent', 'windows'))
            p.add_argument('--video-provider', choices=('comfyui', 'dashscope'))
            p.add_argument('--video-options', type=json.loads)
            p.add_argument('--image-model')
            p.add_argument('--image-size')
    args = parser.parse_args(argv); project = Path(args.project); manifest_path = project / 'project.json'
    if args.command == 'create':
        if manifest_path.exists(): raise SystemExit('project already exists; use resume or a new directory')
        payload = json.load(sys.stdin) if args.create_stdin else {'story': args.story}
        package = payload.get('story_package')
        manifest = ProjectManifest.from_story_package(project.name, package) if package is not None else ProjectManifest.create(project.name, payload['story'])
        job = payload.get('story_job')
        if job is not None and (not isinstance(job, dict) or job.get('schema') != 'story-job/v1'):
            raise ValueError('expected story-job/v1')
        manifest.story_job = job
        if not manifest.story.strip(): raise ValueError('story must not be empty')
        project.mkdir(parents=True, exist_ok=True); manifest.save(manifest_path); print(f'created {manifest_path}'); return 0
    manifest = ProjectManifest.load(manifest_path)
    if args.command in {'candidates', 'select', 'effects'}:
        from .candidates import generate_candidates, select_candidate
        if args.command == 'candidates':
            from .review_gate import block_unapproved
            if manifest.review_enforced and block_unapproved(manifest, project, 'storyboard'):
                print(manifest.status)
                return 0
            generate_candidates(manifest, project, args.shot, args.candidate_count)
        elif args.command == 'select':
            select_candidate(manifest, args.shot, args.candidate_id, args.frame_role)
        else:
            shot = manifest.shot(args.shot)
            shot.motion, shot.transition = args.motion, args.transition
            manifest.approvals.pop('assets', None)
            manifest.output = None
        manifest.save(manifest_path)
        return 0
    tts_source = getattr(args, 'tts_source', None)
    if tts_source is not None and tts_source != manifest.tts_source:
        manifest.tts_source = tts_source
        for shot in manifest.shots:
            shot.assets.pop('audio', None)
    if getattr(args, 'mode', None): manifest.mode = args.mode
    video_route = getattr(args, 'video_provider', None)
    video_options = getattr(args, 'video_options', None)
    from .project_video import validate_video_options, effective_video_options
    if video_options is not None:
        validate_video_options(video_options)
    next_route = video_route if video_route is not None else manifest.video_provider
    next_options = video_options if video_options is not None else manifest.video_options
    changed = (next_route != manifest.video_provider or
               effective_video_options(next_route, next_options) !=
               effective_video_options(manifest.video_provider, manifest.video_options))
    manifest.video_provider = next_route
    manifest.video_options = next_options
    if changed:
        for shot in manifest.shots:
            shot.assets.pop('video', None)
            shot.mode = 'preview'
            shot.error = None
    for field in ('image_source', 'image_model', 'image_size'):
        value = getattr(args, field, None)
        if value is not None:
            setattr(manifest, field, value)
    if args.command == 'review':
        if args.shot == 'all':
            from .review_gate import approve
            approve(manifest)
            manifest.save(manifest_path)
            print('Project review approved; resume to continue')
            return 0
        if not args.shot: raise SystemExit('review requires a shot id')
        shot = manifest.shot(args.shot); shot.review = 'approved'; manifest.save(manifest_path); print(f'approved {args.shot}'); return 0
    if args.command == 'retry':
        if not args.shot: raise SystemExit('retry requires a shot id')
        shot = manifest.shot(args.shot)
        shot.status = 'planned'
        shot.assets.clear()
        shot.image_task.clear()
        shot.cache_key = None
        shot.error = None
    video_provider = None
    if getattr(args, 'require_review', False):
        manifest.review_enforced = True
    enforce_review = manifest.review_enforced and not getattr(args, 'auto', False)
    if enforce_review:
        from .review_gate import block_unapproved
        if block_unapproved(manifest, project, 'storyboard'):
            print(manifest.status)
            return 0
    if getattr(args, 'real_preview', None) is not None:
        manifest.real_preview = args.real_preview
    if manifest.mode == 'render':
        from .project_video import build_project_video_provider, needs_video_generation
        try:
            if needs_video_generation(manifest):
                video_provider = build_project_video_provider(manifest, project)
        except Exception as exc:
            manifest.status = 'failed'
            manifest.error = f'video provider setup failed: {type(exc).__name__}: {exc}'
            manifest.save(manifest_path)
            raise
    output = render_preview(manifest, project, video_provider=video_provider, real_preview=getattr(args, 'real_preview', None), enforce_review=enforce_review); manifest.save(manifest_path); print(f'preview: {output}' if output else manifest.status); return 0

if __name__ == '__main__': raise SystemExit(main())
