"""Unified CLI entrypoint."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from .manifest import ProjectManifest
from .preview import render_preview

def _legacy() -> int:
    from .runtime import build_runtime_from_environment, resolve_sibling_root
    from .adapters import HttpStoryCampaignAdapter, StoryImageAdapter, StoryVideoAdapter
    from .registry import ArtifactRegistry
    from .quality import evaluate_artifact
    payload = json.loads(sys.stdin.read() or '{}')
    base = Path(__file__).resolve().parents[1]
    os.environ.setdefault('STORY_CAMPAIGN_ROOT', str(resolve_sibling_root('microcodex-short-drama-studio', 'microcodex-story-campaign-adapter', 'sidecar')))
    os.environ.setdefault('STORY_IMAGE_AGENT_ROOT', str(resolve_sibling_root('story-image-agent', 'microcodex-story-image-agent', module_name='story_image_agent')))
    os.environ.setdefault('STORY_VIDEO_AGENT_ROOT', str(resolve_sibling_root('story-video-agent', 'microcodex-story-video-agent', module_name='story_video_agent')))
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
            comfy_client = pkg.ComfyUIAdapter.from_environment()
            if comfy_client is not None and hasattr(comfy_client, 'timeout'):
                comfy_client.timeout = float(os.environ.get('MINIMAX_H3_COMFYUI_TIMEOUT') or '180.0')
            result = StoryVideoAdapter(pkg.VideoPromptWorkflow('story-media-orchestrator'), models=ModelConfig(video_turbo=turbo, video_steps=8 if turbo else 20), comfy=comfy_client, registry=registry).run(first_frame_ref=payload['first_frame_ref'], last_frame_ref=payload.get('last_frame_ref'), story_spans=payload.get('source_spans', []), scene=payload.get('scene', {}), prompt=payload.get('prompt'), seed=payload.get('seed', 0), width=payload.get('width', 864), height=payload.get('height', 480), length=payload.get('length', 124), fps=payload.get('fps', 24.0))
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
    create.add_argument('--auto-split-shots', action='store_true', default=False)
    create.add_argument('--max-shot-seconds', type=float, default=4.8)
    for name in ('candidates', 'select', 'effects'):
        p = sub.add_parser(name)
        p.add_argument('project'); p.add_argument('shot')
        p.add_argument('--candidate-count', type=int, default=2)
        p.add_argument('--candidate-id')
        p.add_argument('--frame-role', choices=('image', 'last_frame'), default='image')
        p.add_argument('--motion', choices=('none', 'push_in', 'pull_out', 'pan', 'pan_left', 'pan_up', 'ken_burns'), default='none')
        p.add_argument('--transition', choices=('cut', 'fade', 'xfade_dissolve', 'xfade_wipeleft', 'xfade_slideup'), default='cut')
        p.add_argument('--video-route', choices=('auto', 'preview', 'i2v_fast', 'i2v_quality', 'kf2v', 'r2v', 't2v', 'comfy_i2v', 'comfy_fl2v', 'comfy_t2v'), default=None)
        p.add_argument('--character-locks', default=None)
    ch = sub.add_parser('character')
    ch.add_argument('project')
    ch.add_argument('action_name', nargs='?', choices=('list', 'sync', 'set', 'portrait', 'lock-shot'), default='list')
    ch.add_argument('--character', dest='character_name')
    ch.add_argument('--appearance')
    ch.add_argument('--clothing')
    ch.add_argument('--visual-dna')
    ch.add_argument('--seed', type=int)
    ch.add_argument('--ref-image')
    ch.add_argument('--clear-ref', action='store_true')
    ch.add_argument('--from-photo', dest='from_photo')
    ch.add_argument('--deidentify-traits', dest='deidentify_traits')
    ch.add_argument('--3d-contour', dest='three_d_contour', action='store_true', default=None)
    ch.add_argument('--max-identity-similarity', type=float, default=None)
    ch.add_argument('--role')
    ch.add_argument('--gender')
    ch.add_argument('--shot', dest='target_shot')
    ch.add_argument('--locks')
    ej = sub.add_parser('export-jianying')
    ej.add_argument('project')
    ej.add_argument('--output-dir')
    mcp_cmd = sub.add_parser('mcp')
    mcp_cmd.add_argument('--transport', choices=('stdio', 'sse', 'streamable-http'), default='stdio')
    mcp_cmd.add_argument('--host', default='127.0.0.1')
    mcp_cmd.add_argument('--port', type=int, default=8780)
    mcp_cmd.add_argument('--flywheel-dir')
    mcp_cmd.add_argument('--profile', choices=('full', 'core', 'producer', 'critic'), default='full')
    doc = sub.add_parser('doctor')
    doc.add_argument('project', nargs='?', default=None)
    doc.add_argument('--probe-network', action='store_true', default=False)
    doc.add_argument('--strict', action='store_true', default=False)
    doc.add_argument('--timeout', type=float, default=5.0)
    sup = sub.add_parser('supervisor')
    sup.add_argument('project')
    sup.add_argument('action_name', nargs='?', choices=('snapshot', 'diagnose', 'flywheel'), default='snapshot')
    sup.add_argument('--shot', dest='supervisor_shot')
    sup.add_argument('--flywheel-dir')
    lhb = sub.add_parser('lint-h3-bundle')
    lhb.add_argument('bundle_dir')
    lhb.add_argument('--max-frames', type=int, default=175)
    lhb.add_argument('--no-images', action='store_true')
    lhb.add_argument('--strict-chapter', action='store_true')
    lhb.add_argument('--json-out')
    lhm = sub.add_parser('lint-handoff')
    lhm.add_argument('manifest_path')
    lhm.add_argument('--check-files', action='store_true')
    lhm.add_argument('--json-out')
    vm = sub.add_parser('verify-master')
    vm.add_argument('target')
    vm.add_argument('--profile', choices=('archive', 'web'), default='archive')
    vm.add_argument('--media')
    vm.add_argument('--reference-video')
    vm.add_argument('--expected-frames', type=int)
    vm.add_argument('--bundle-root')
    vm.add_argument('--strict-freeze', action='store_true')
    vm.add_argument('--json-out')
    anc = sub.add_parser('assemble-native-chapter')
    anc.add_argument('chapter_dir')
    anc.add_argument('--mix', required=True)
    anc.add_argument('--audit')
    anc.add_argument('--no-verify', action='store_true')
    mw = sub.add_parser('master-web')
    mw.add_argument('source_mp4')
    mw.add_argument('out_mp4')
    mw.add_argument('--audit', required=True)
    mw.add_argument('--nosub-in')
    mw.add_argument('--nosub-out')
    mw.add_argument('--target-lufs', type=float, default=-16.0)
    mw.add_argument('--json-out')
    asp = sub.add_parser('audit-speech')
    asp.add_argument('target')
    asp.add_argument('--bundle')
    asp.add_argument('--audit')
    asp.add_argument('--expected-line')
    asp.add_argument('--speaker', default='Speaker')
    asp.add_argument('--style', choices=('Dialogue', 'Voiceover'), default='Dialogue')
    asp.add_argument('--model', default=os.environ.get('STORY_MEDIA_ASR_MODEL', 'small'))
    asp.add_argument('--device', default=os.environ.get('STORY_MEDIA_ASR_DEVICE', 'cpu'))
    asp.add_argument('--compute-type', default=os.environ.get('STORY_MEDIA_ASR_COMPUTE_TYPE', 'int8'))
    asp.add_argument('--cer-warn', type=float, default=0.20)
    asp.add_argument('--cer-fail', type=float, default=0.45)
    asp.add_argument('--emit-mix')
    asp.add_argument('--json-out')
    cp = sub.add_parser('critic-panel')
    cp.add_argument('project')
    cp.add_argument('shot', nargs='?', default=None)
    cp.add_argument('--all', dest='all_shots', action='store_true')
    cp.add_argument('--shots', dest='shot_ids', default=None)
    cp.add_argument('--only-pending', action='store_true')
    cp.add_argument('--stage', choices=('image', 'video'), default='video')
    cp.add_argument('--policy', choices=('escalate', 'always', 'primary_only'), default='escalate')
    cp.add_argument('--escalate-after-reworks', type=int, default=2)
    cp.add_argument('--force-tiebreaker', action='store_true')
    cp.add_argument('--no-refresh', action='store_true')
    cp.add_argument('--persist', action='store_true')
    cp.add_argument('--lease-token')
    cp.add_argument('--json-out')
    nr = sub.add_parser('nightly-run')
    nr.add_argument('project')
    nr.add_argument('--stage', choices=('image', 'video'), default='video')
    nr.add_argument('--policy', choices=('escalate', 'always', 'primary_only'), default='escalate')
    nr.add_argument('--all-shots', dest='all_shots', action='store_true')
    nr.add_argument('--no-persist', action='store_true')
    nr.add_argument('--run-render', action='store_true')
    nr.add_argument('--h3-bundle')
    nr.add_argument('--max-auto-rework-rounds', type=int, default=0)
    nr.add_argument('--max-rework-cost', type=float, default=5.0)
    nr.add_argument('--auto-finalize', action='store_true')
    nr.add_argument('--no-verify-master', action='store_true')
    nr.add_argument('--delivery-dir')
    nr.add_argument('--client-id')
    nr.add_argument('--lease-token')
    nr.add_argument('--json-out')
    pd = sub.add_parser('package-delivery')
    pd.add_argument('project')
    pd.add_argument('--output-dir')
    pd.add_argument('--client-id')
    pd.add_argument('--nosub')
    pd.add_argument('--speech-audit')
    pd.add_argument('--allow-incomplete', action='store_true')
    pd.add_argument('--no-verify-master', action='store_true')
    pd.add_argument('--export-jianying', action='store_true')
    pd.add_argument('--json-out')
    icw = sub.add_parser('init-client-workspace')
    icw.add_argument('clients_root')
    icw.add_argument('client_id')
    icw.add_argument('project_id')
    icw.add_argument('--story', default='')
    aus = sub.add_parser('audit-script')
    aus.add_argument('project')
    aus.add_argument('--json-out')
    rhs = sub.add_parser('rehearse-script')
    rhs.add_argument('project')
    rhs.add_argument('--genre')
    rhs.add_argument('--flywheel-dir')
    rhs.add_argument('--json-out')
    psb = sub.add_parser('patch-script')
    psb.add_argument('project')
    psb.add_argument('shot')
    psb.add_argument('--text', dest='new_text')
    psb.add_argument('--visual-action')
    psb.add_argument('--kinetic-action')
    psb.add_argument('--framing')
    psb.add_argument('--location')
    psb.add_argument('--continuity')
    psb.add_argument('--narration')
    psb.add_argument('--duration', type=float)
    psb.add_argument('--invalidate-audio', action='store_true')
    psb.add_argument('--reason', default='')
    psb.add_argument('--lease-token')
    ibs = sub.add_parser('insert-bridge')
    ibs.add_argument('project')
    ibs.add_argument('after_shot')
    ibs.add_argument('--text', required=True)
    ibs.add_argument('--visual-action', default='')
    ibs.add_argument('--framing', default='中景')
    ibs.add_argument('--location', default='')
    ibs.add_argument('--continuity', default='')
    ibs.add_argument('--story-role', default='causal_bridge')
    ibs.add_argument('--duration', type=float, default=3.5)
    ibs.add_argument('--shot-id', dest='new_shot_id')
    ibs.add_argument('--lease-token')
    for name in ('run', 'resume', 'retry', 'review'):
        p = sub.add_parser(name); p.add_argument('project'); p.add_argument('shot', nargs='?')
        if name in ('run', 'resume'): p.add_argument('--mode', choices=('preview', 'render'))
        if name in ('run', 'resume', 'retry'): p.add_argument('--real-preview', action='store_true', default=None)
        if name in ('run', 'resume', 'retry'):
            p.add_argument('--require-review', action='store_true')
            p.add_argument('--auto', action='store_true')
            p.add_argument('--image-source', choices=('local', 'dashscope', 'codex'))
            p.add_argument('--tts-source', choices=('silent', 'windows', 'cosyvoice', 'edge'))
            p.add_argument('--video-provider', choices=('comfyui', 'dashscope', 'auto'))
            p.add_argument('--video-routing-policy', choices=('manual', 'smart_i2v', 'hybrid_economy', 'quality_first'))
            p.add_argument('--video-fallback', dest='video_fallback_provider', choices=('none', 'dashscope', 'comfyui', 'preview'))
            p.add_argument('--consistency-mode', choices=('off', 'auto', 'prompt_lock', 'anchor_ref'))
            p.add_argument('--style-preset', choices=('none', 'realistic_drama', 'cinematic', 'anime_2d', 'guofeng_3d', 'ink_wash', 'cyberpunk', 'stylized_3d_anime'))
            p.add_argument('--video-options', type=json.loads)
            p.add_argument('--image-model')
            p.add_argument('--image-size')
            p.add_argument('--aspect-ratio', choices=('16:9', '9:16', '1:1'))
            p.add_argument('--scale-mode', choices=('contain', 'cover', 'blur_pad'))
            p.add_argument('--alignment-mode', choices=('pad', 'elastic'))
            p.add_argument('--subtitle-style', choices=('soft', 'burn_in'))
            p.add_argument('--speech-mode', choices=('auto', 'full_text', 'dialogue_only', 'narrator_and_dialogue'))
            p.add_argument('--voice-cast', type=json.loads)
            p.add_argument('--bgm', dest='bgm_source')
            p.add_argument('--bgm-volume', type=float)
            p.add_argument('--bgm-ducking', action=argparse.BooleanOptionalAction, default=None)
            p.add_argument('--budget-limit', dest='budget_limit_cny', type=float)
            p.add_argument('--flywheel-dir', dest='flywheel_dir')
            p.add_argument('--auto-rework', dest='auto_rework', action='store_true', default=None)
            p.add_argument('--no-auto-rework', dest='auto_rework', action='store_false')
            p.add_argument('--max-rework-rounds', type=int, default=None)
            p.add_argument('--relay-tail-frames', dest='relay_tail_frames', action='store_true', default=None)
            p.add_argument('--execution-strategy', choices=('interleaved', 'stage_batched'))
            p.add_argument('--stage', dest='target_stage', choices=('all', 'keyframes', 'tts', 'video', 'assemble'), default=None)
            p.add_argument('--lipsync-provider', choices=('off', 'http', 'comfyui', 'command'))
            p.add_argument('--auto-split-shots', dest='auto_split_shots', action='store_true', default=None)
            p.add_argument('--max-shot-seconds', dest='max_shot_seconds', type=float, default=None)
            p.add_argument('--export-jianying', action='store_true')
            p.add_argument('--edit-options', type=json.loads)
    args = parser.parse_args(argv)
    if args.command == 'mcp':
        from .mcp_server import main as mcp_main
        mcp_argv = ['--transport', args.transport, '--host', args.host, '--port', str(args.port), '--profile', args.profile]
        if args.flywheel_dir:
            mcp_argv.extend(['--flywheel-dir', args.flywheel_dir])
        return mcp_main(mcp_argv)
    if args.command == 'doctor':
        from .doctor import run_preflight_doctor
        report = run_preflight_doctor(
            args.project,
            probe_network=bool(args.probe_network),
            timeout=float(args.timeout),
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if report['overall_status'] == 'blocked':
            return 1
        if args.strict and (not report['ready_for_h3_production'] or not report['closed_source_safe']):
            return 1
        return 0
    if args.command == 'lint-h3-bundle':
        from .h3_bundle import validate_h3_bundle
        report = validate_h3_bundle(
            Path(args.bundle_dir),
            max_frames=int(args.max_frames),
            check_images=not bool(args.no_images),
            require_chapter_field=bool(args.strict_chapter),
        )
        payload = report.to_dict()
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0 if payload.get('ok') else 1
    if args.command == 'patch-script':
        from .script_audit import patch_shot_script_beat
        payload = patch_shot_script_beat(
            args.project,
            args.shot,
            new_text=args.new_text,
            visual_action=args.visual_action,
            kinetic_action=args.kinetic_action,
            framing=args.framing,
            location=args.location,
            continuity=args.continuity,
            narration=args.narration,
            duration=args.duration,
            invalidate_audio=bool(args.invalidate_audio),
            reason=args.reason,
            lease_token=args.lease_token,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0 if payload.get('patched') else 1
    if args.command == 'insert-bridge':
        from .script_audit import insert_script_bridge_shot
        payload = insert_script_bridge_shot(
            args.project,
            args.after_shot,
            text=args.text,
            visual_action=args.visual_action,
            framing=args.framing,
            location=args.location,
            continuity=args.continuity,
            story_role=args.story_role,
            duration=float(args.duration),
            new_shot_id=args.new_shot_id,
            lease_token=args.lease_token,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0 if payload['ok'] else 1
    if args.command == 'verify-master':
        from .master_verifier import verify_master_release, verify_series_audit
        target_p = Path(args.target)
        if target_p.suffix.lower() == '.json':
            report = verify_series_audit(
                target_p,
                media_override=Path(args.media) if args.media else None,
                profile=args.profile,
                reference_video_mp4=Path(args.reference_video) if args.reference_video else None,
                bundle_root=Path(args.bundle_root) if args.bundle_root else None,
                strict_freeze=bool(args.strict_freeze),
            )
        else:
            report = verify_master_release(
                target_p,
                profile=args.profile,
                expected_frames=args.expected_frames,
                reference_video_mp4=Path(args.reference_video) if args.reference_video else None,
                strict_freeze=bool(args.strict_freeze),
            )
        payload = report.to_dict()
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0 if payload.get('ok') else 1
    if args.command == 'patch-script':
        from .script_audit import patch_shot_script_beat
        payload = patch_shot_script_beat(
            args.project,
            args.shot,
            new_text=args.new_text,
            visual_action=args.visual_action,
            kinetic_action=args.kinetic_action,
            framing=args.framing,
            location=args.location,
            continuity=args.continuity,
            narration=args.narration,
            duration=args.duration,
            invalidate_audio=bool(args.invalidate_audio),
            reason=args.reason,
            lease_token=args.lease_token,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0 if payload.get('patched') else 1
    if args.command == 'insert-bridge':
        from .script_audit import insert_script_bridge_shot
        payload = insert_script_bridge_shot(
            args.project,
            args.after_shot,
            text=args.text,
            visual_action=args.visual_action,
            framing=args.framing,
            location=args.location,
            continuity=args.continuity,
            story_role=args.story_role,
            duration=float(args.duration),
            new_shot_id=args.new_shot_id,
            lease_token=args.lease_token,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0 if report.ok else 1
    if args.command == 'assemble-native-chapter':
        from .native_assembly import assemble_native_chapter
        chap_dir = Path(args.chapter_dir)
        mix_data = json.loads(Path(args.mix).read_text(encoding='utf-8'))
        audit_p = Path(args.audit) if args.audit else chap_dir / 'full_9shot_audit.json'
        audit_data = json.loads(audit_p.read_text(encoding='utf-8'))
        cid = str(audit_data.get('chapter') or mix_data.get('chapter') or '01')
        res = assemble_native_chapter(
            audit_data['shots'],
            chap_dir,
            chapter_id=cid,
            shot_gains_db=mix_data['shot_gains_db'],
            subtitle_cues=mix_data.get('subtitle_cues') or [],
            audio_source_override=mix_data.get('audio_source_override'),
            verify=not bool(args.no_verify),
        )
        print(json.dumps(res, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command == 'master-web':
        from .mastering import MasteringProfile, build_timeline_from_series_audit, master_web_release
        if bool(args.nosub_in) != bool(args.nosub_out):
            raise SystemExit('master-web requires both --nosub-in and --nosub-out together')
        segments, global_cues, card_windows = build_timeline_from_series_audit(Path(args.audit))
        prof = MasteringProfile(target_lufs=float(args.target_lufs))
        main_res = master_web_release(
            Path(args.source_mp4),
            Path(args.out_mp4),
            segments=segments,
            global_cues=global_cues,
            title_card_windows=card_windows,
            profile=prof,
        )
        out_payload: dict[str, object] = {'master': main_res}
        if args.nosub_in and args.nosub_out:
            nosub_res = master_web_release(
                Path(args.nosub_in),
                Path(args.nosub_out),
                segments=segments,
                global_cues=global_cues,
                title_card_windows=card_windows,
                profile=prof,
                precomputed_shot_gains=main_res['per_shot_gains_db'],
            )
            out_payload['nosub'] = nosub_res
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(out_payload, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command == 'audit-speech':
        from .speech_audit import (
            FasterWhisperBackend,
            audit_chapter_speech,
            audit_shot_speech,
            build_auto_chapter_mix,
        )
        backend = FasterWhisperBackend(
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
        )
        target_p = Path(args.target)
        if target_p.is_dir():
            report = audit_chapter_speech(
                target_p,
                bundle_dir=Path(args.bundle) if args.bundle else None,
                audit_json=Path(args.audit) if args.audit else None,
                backend=backend,
                cer_warn_threshold=float(args.cer_warn),
                cer_fail_threshold=float(args.cer_fail),
            )
            if args.emit_mix:
                audit_src_p = Path(args.audit) if args.audit else target_p / 'full_9shot_audit.json'
                shots_list = json.loads(audit_src_p.read_text(encoding='utf-8')).get('shots') or []
                mix_payload = build_auto_chapter_mix(report, shots=shots_list)
                Path(args.emit_mix).write_text(json.dumps(mix_payload, ensure_ascii=False, indent=2), encoding='utf-8')
        else:
            report = audit_shot_speech(
                target_p,
                expected_line=args.expected_line,
                speaker=args.speaker,
                style=args.style,
                backend=backend,
                cer_warn_threshold=float(args.cer_warn),
                cer_fail_threshold=float(args.cer_fail),
            )
        payload = report.to_dict()
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0 if payload.get('ok') else 1
    if args.command == 'patch-script':
        from .script_audit import patch_shot_script_beat
        payload = patch_shot_script_beat(
            args.project,
            args.shot,
            new_text=args.new_text,
            visual_action=args.visual_action,
            kinetic_action=args.kinetic_action,
            framing=args.framing,
            location=args.location,
            continuity=args.continuity,
            narration=args.narration,
            duration=args.duration,
            invalidate_audio=bool(args.invalidate_audio),
            reason=args.reason,
            lease_token=args.lease_token,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0 if payload.get('patched') else 1
    if args.command == 'insert-bridge':
        from .script_audit import insert_script_bridge_shot
        payload = insert_script_bridge_shot(
            args.project,
            args.after_shot,
            text=args.text,
            visual_action=args.visual_action,
            framing=args.framing,
            location=args.location,
            continuity=args.continuity,
            story_role=args.story_role,
            duration=float(args.duration),
            new_shot_id=args.new_shot_id,
            lease_token=args.lease_token,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0 if report.ok else 1
    if args.command == 'critic-panel':
        from .external_critics import run_external_critic_panel, run_project_critic_panel
        if args.all_shots or args.shot is None or args.shot_ids:
            parsed_ids = [s.strip() for s in (args.shot_ids or '').split(',') if s.strip()] or None
            payload = run_project_critic_panel(
                args.project,
                stage=args.stage,
                policy=args.policy,
                shot_ids=parsed_ids,
                only_pending=bool(args.only_pending),
                escalate_after_reworks=int(args.escalate_after_reworks),
                force_tiebreaker=bool(args.force_tiebreaker),
                refresh_diagnostics=not bool(args.no_refresh),
                persist_verdict=bool(args.persist),
                lease_token=args.lease_token,
            )
            ok = bool(payload.get('ok'))
        else:
            payload = run_external_critic_panel(
                args.project,
                args.shot,
                stage=args.stage,
                policy=args.policy,
                escalate_after_reworks=int(args.escalate_after_reworks),
                force_tiebreaker=bool(args.force_tiebreaker),
                refresh_diagnostics=not bool(args.no_refresh),
                persist_verdict=bool(args.persist),
                lease_token=args.lease_token,
            )
            ok = payload.get('decision') in {'passed', 'warning'}
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0 if payload.get('ok') else 1
    if args.command == 'patch-script':
        from .script_audit import patch_shot_script_beat
        payload = patch_shot_script_beat(
            args.project,
            args.shot,
            new_text=args.new_text,
            visual_action=args.visual_action,
            kinetic_action=args.kinetic_action,
            framing=args.framing,
            location=args.location,
            continuity=args.continuity,
            narration=args.narration,
            duration=args.duration,
            invalidate_audio=bool(args.invalidate_audio),
            reason=args.reason,
            lease_token=args.lease_token,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0 if payload.get('patched') else 1
    if args.command == 'insert-bridge':
        from .script_audit import insert_script_bridge_shot
        payload = insert_script_bridge_shot(
            args.project,
            args.after_shot,
            text=args.text,
            visual_action=args.visual_action,
            framing=args.framing,
            location=args.location,
            continuity=args.continuity,
            story_role=args.story_role,
            duration=float(args.duration),
            new_shot_id=args.new_shot_id,
            lease_token=args.lease_token,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0 if ok else 1
    if args.command == 'nightly-run':
        from .nightly_runner import run_unattended_episode
        payload = run_unattended_episode(
            args.project,
            stage=args.stage,
            policy=args.policy,
            only_pending=not bool(args.all_shots),
            persist_verdict=not bool(args.no_persist),
            run_render_stages=bool(args.run_render),
            h3_bundle_dir=args.h3_bundle,
            max_auto_rework_rounds=int(args.max_auto_rework_rounds),
            max_rework_cost_cny=float(args.max_rework_cost),
            auto_finalize=bool(args.auto_finalize),
            verify_master=not bool(args.no_verify_master),
            package_delivery_dir=args.delivery_dir,
            client_id=args.client_id,
            report_out=args.json_out,
            lease_token=args.lease_token,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0 if payload.get('ok') else 1
    if args.command == 'patch-script':
        from .script_audit import patch_shot_script_beat
        payload = patch_shot_script_beat(
            args.project,
            args.shot,
            new_text=args.new_text,
            visual_action=args.visual_action,
            kinetic_action=args.kinetic_action,
            framing=args.framing,
            location=args.location,
            continuity=args.continuity,
            narration=args.narration,
            duration=args.duration,
            invalidate_audio=bool(args.invalidate_audio),
            reason=args.reason,
            lease_token=args.lease_token,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0 if payload.get('patched') else 1
    if args.command == 'insert-bridge':
        from .script_audit import insert_script_bridge_shot
        payload = insert_script_bridge_shot(
            args.project,
            args.after_shot,
            text=args.text,
            visual_action=args.visual_action,
            framing=args.framing,
            location=args.location,
            continuity=args.continuity,
            story_role=args.story_role,
            duration=float(args.duration),
            new_shot_id=args.new_shot_id,
            lease_token=args.lease_token,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0 if payload.get('ok') else 1
    if args.command == 'package-delivery':
        from .delivery_bundle import package_commercial_delivery
        payload = package_commercial_delivery(
            args.project,
            output_dir=args.output_dir,
            client_id=args.client_id,
            nosub_mp4=args.nosub,
            speech_audit_json=args.speech_audit,
            require_all_passed=not bool(args.allow_incomplete),
            verify_master=not bool(args.no_verify_master),
            export_jianying_if_missing=bool(args.export_jianying),
        )
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0 if payload.get('ok') else 1
    if args.command == 'patch-script':
        from .script_audit import patch_shot_script_beat
        payload = patch_shot_script_beat(
            args.project,
            args.shot,
            new_text=args.new_text,
            visual_action=args.visual_action,
            kinetic_action=args.kinetic_action,
            framing=args.framing,
            location=args.location,
            continuity=args.continuity,
            narration=args.narration,
            duration=args.duration,
            invalidate_audio=bool(args.invalidate_audio),
            reason=args.reason,
            lease_token=args.lease_token,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0 if payload.get('patched') else 1
    if args.command == 'insert-bridge':
        from .script_audit import insert_script_bridge_shot
        payload = insert_script_bridge_shot(
            args.project,
            args.after_shot,
            text=args.text,
            visual_action=args.visual_action,
            framing=args.framing,
            location=args.location,
            continuity=args.continuity,
            story_role=args.story_role,
            duration=float(args.duration),
            new_shot_id=args.new_shot_id,
            lease_token=args.lease_token,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0 if payload.get('ok') else 1
    if args.command == 'init-client-workspace':
        from .delivery_bundle import init_isolated_client_workspace
        payload = init_isolated_client_workspace(
            args.clients_root,
            args.client_id,
            args.project_id,
            story=args.story,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command == 'rehearse-script':
        from .script_audit import rehearse_project_script
        payload = rehearse_project_script(
            args.project,
            genre=args.genre,
            server_flywheel_dir=args.flywheel_dir,
        )
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0 if payload.get('p0_blocking_count', 0) == 0 else 1
    if args.command == 'patch-script':
        from .script_audit import patch_shot_script_beat
        payload = patch_shot_script_beat(
            args.project,
            args.shot,
            new_text=args.new_text,
            visual_action=args.visual_action,
            kinetic_action=args.kinetic_action,
            framing=args.framing,
            location=args.location,
            continuity=args.continuity,
            narration=args.narration,
            duration=args.duration,
            invalidate_audio=bool(args.invalidate_audio),
            reason=args.reason,
            lease_token=args.lease_token,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0 if payload.get('patched') else 1
    if args.command == 'insert-bridge':
        from .script_audit import insert_script_bridge_shot
        payload = insert_script_bridge_shot(
            args.project,
            args.after_shot,
            text=args.text,
            visual_action=args.visual_action,
            framing=args.framing,
            location=args.location,
            continuity=args.continuity,
            story_role=args.story_role,
            duration=float(args.duration),
            new_shot_id=args.new_shot_id,
            lease_token=args.lease_token,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0 if report.ok else 1
    if args.command == 'lint-handoff':
        from .production_playbook import lint_handoff_manifest
        payload = lint_handoff_manifest(
            Path(args.manifest_path),
            require_first_frame_exists=bool(args.check_files),
        )
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0 if payload.get('ok') else 1
    if args.command == 'patch-script':
        from .script_audit import patch_shot_script_beat
        payload = patch_shot_script_beat(
            args.project,
            args.shot,
            new_text=args.new_text,
            visual_action=args.visual_action,
            kinetic_action=args.kinetic_action,
            framing=args.framing,
            location=args.location,
            continuity=args.continuity,
            narration=args.narration,
            duration=args.duration,
            invalidate_audio=bool(args.invalidate_audio),
            reason=args.reason,
            lease_token=args.lease_token,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0 if payload.get('patched') else 1
    if args.command == 'insert-bridge':
        from .script_audit import insert_script_bridge_shot
        payload = insert_script_bridge_shot(
            args.project,
            args.after_shot,
            text=args.text,
            visual_action=args.visual_action,
            framing=args.framing,
            location=args.location,
            continuity=args.continuity,
            story_role=args.story_role,
            duration=float(args.duration),
            new_shot_id=args.new_shot_id,
            lease_token=args.lease_token,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command == 'audit-script':
        from .script_audit import audit_project_script
        payload = audit_project_script(args.project)
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0
    project = Path(args.project); manifest_path = project / 'project.json'
    if args.command == 'create':
        if manifest_path.exists(): raise SystemExit('project already exists; use resume or a new directory')
        payload = json.load(sys.stdin) if args.create_stdin else {'story': args.story}
        package = payload.get('story_package')
        manifest = (
            ProjectManifest.from_story_package(
                project.name,
                package,
                auto_split_shots=bool(getattr(args, 'auto_split_shots', False)),
                max_shot_seconds=float(getattr(args, 'max_shot_seconds', 4.8) or 4.8),
            )
            if package is not None
            else ProjectManifest.create(
                project.name,
                payload['story'],
                auto_split_shots=bool(getattr(args, 'auto_split_shots', False)),
                max_shot_seconds=float(getattr(args, 'max_shot_seconds', 4.8) or 4.8),
            )
        )
        job = payload.get('story_job')
        if job is not None and (not isinstance(job, dict) or job.get('schema') != 'story-job/v1'):
            raise ValueError('expected story-job/v1')
        manifest.story_job = job
        if not manifest.story.strip(): raise ValueError('story must not be empty')
        project.mkdir(parents=True, exist_ok=True); manifest.save(manifest_path); print(f'created {manifest_path}'); return 0
    manifest = ProjectManifest.load(manifest_path)
    if args.command == 'character':
        from .character_bible import (
            generate_character_portrait,
            invalidate_character_shots,
            sync_character_bible,
            upsert_character_profile,
        )
        act = args.action_name
        if act == 'list':
            if not manifest.character_bible:
                sync_character_bible(manifest)
                manifest.save(manifest_path)
            print(json.dumps(manifest.character_bible, ensure_ascii=False), flush=True)
            return 0
        if act == 'sync':
            bible = sync_character_bible(manifest)
            manifest.save(manifest_path)
            print(json.dumps(bible, ensure_ascii=False), flush=True)
            return 0
        if act == 'set':
            if not args.character_name:
                raise SystemExit('character set requires --character')
            profile = upsert_character_profile(
                manifest,
                args.character_name,
                appearance=args.appearance,
                clothing=args.clothing,
                visual_dna=args.visual_dna,
                seed=args.seed,
                ref_image=args.ref_image,
                role=args.role,
                gender=args.gender,
                clear_ref=bool(args.clear_ref),
            )
            invalidated = invalidate_character_shots(
                manifest,
                args.character_name,
                invalidate_audio=bool(args.gender is not None or args.role is not None),
            )
            manifest.save(manifest_path)
            print(json.dumps({'profile': profile, 'invalidated_shots': invalidated}, ensure_ascii=False), flush=True)
            return 0
        if act == 'portrait':
            if not args.character_name:
                raise SystemExit('character portrait requires --character')
            if any(
                val is not None
                for val in (args.appearance, args.clothing, args.visual_dna, args.seed, args.role, args.gender)
            ):
                upsert_character_profile(
                    manifest,
                    args.character_name,
                    appearance=args.appearance,
                    clothing=args.clothing,
                    visual_dna=args.visual_dna,
                    seed=args.seed,
                    role=args.role,
                    gender=args.gender,
                )
            profile = generate_character_portrait(
                manifest,
                project,
                args.character_name,
                from_photo=args.from_photo,
                deidentify_traits=args.deidentify_traits,
                three_d_contour=args.three_d_contour,
                max_identity_similarity=getattr(args, 'max_identity_similarity', None),
            )
            invalidated = invalidate_character_shots(manifest, args.character_name)
            manifest.save(manifest_path)
            print(json.dumps({'profile': profile, 'invalidated_shots': invalidated}, ensure_ascii=False), flush=True)
            return 0
        if act == 'lock-shot':
            if not args.target_shot:
                raise SystemExit('character lock-shot requires --shot')
            shot = manifest.shot(args.target_shot)
            shot.character_locks = [x.strip() for x in (args.locks or '').split(',') if x.strip()]
            shot.assets.pop('image', None)
            shot.image_task.clear()
            shot.cache_key = None
            manifest.output = None
            manifest.save(manifest_path)
            print(json.dumps({'shot': shot.id, 'character_locks': shot.character_locks}, ensure_ascii=False), flush=True)
            return 0
    if args.command == 'export-jianying':
        from .jianying import export_jianying_draft
        draft_dir = export_jianying_draft(manifest, project, args.output_dir)
        manifest.save(manifest_path)
        print(f'jianying_draft: {draft_dir}')
        return 0
    if args.command == 'supervisor':
        act = args.action_name
        if act == 'snapshot':
            from .supervisor import get_pipeline_snapshot
            print(json.dumps(get_pipeline_snapshot(project), ensure_ascii=False, indent=2), flush=True)
            return 0
        if act == 'diagnose':
            if not args.supervisor_shot:
                raise SystemExit('supervisor diagnose requires --shot')
            from .frame_decomposer import refresh_shot_diagnostics
            diag = refresh_shot_diagnostics(manifest, project, args.supervisor_shot)
            manifest.save(manifest_path)
            print(json.dumps(diag, ensure_ascii=False, indent=2), flush=True)
            return 0
        if act == 'flywheel':
            from .flywheel import get_flywheel_stats, resolve_flywheel_dir
            fly_dir = resolve_flywheel_dir(manifest, project, args.flywheel_dir)
            if fly_dir is None:
                raise SystemExit('unable to resolve flywheel directory')
            print(json.dumps(get_flywheel_stats(fly_dir), ensure_ascii=False, indent=2), flush=True)
            return 0
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
            if getattr(args, 'video_route', None) is not None:
                from .video_router import validate_shot_video_route
                next_shot_route = validate_shot_video_route(args.video_route)
                if next_shot_route != shot.video_route:
                    shot.video_route = next_shot_route
                    shot.assets.pop('video', None)
                    shot.mode = 'preview'
                    shot.error = None
            if getattr(args, 'character_locks', None) is not None:
                next_locks = [x.strip() for x in args.character_locks.split(',') if x.strip()]
                if next_locks != shot.character_locks:
                    shot.character_locks = next_locks
                    shot.assets.pop('image', None)
                    shot.image_task.clear()
                    shot.cache_key = None
            manifest.approvals.pop('assets', None)
            manifest.output = None
        manifest.save(manifest_path)
        return 0
    tts_source = getattr(args, 'tts_source', None)
    if tts_source is not None and tts_source != manifest.tts_source:
        manifest.tts_source = tts_source
        for shot in manifest.shots:
            shot.assets.pop('audio', None)
    edit_options = getattr(args, 'edit_options', None) or {}
    if edit_options and not isinstance(edit_options, dict):
        raise ValueError('edit-options must be a JSON object')
    prev_aspect_ratio = manifest.aspect_ratio
    prev_speech_mode = manifest.speech_mode
    prev_voice_cast = dict(manifest.voice_cast)
    for key in ('aspect_ratio', 'scale_mode', 'alignment_mode', 'subtitle_style', 'speech_mode', 'bgm_volume', 'bgm_ducking'):
        val = getattr(args, key, None)
        if val is None and key in edit_options and edit_options[key] is not None:
            val = edit_options[key]
        if val is not None:
            setattr(manifest, key, float(val) if key == 'bgm_volume' else bool(val) if key == 'bgm_ducking' else val)
    voice_cast = getattr(args, 'voice_cast', None)
    if voice_cast is None and isinstance(edit_options.get('voice_cast'), dict):
        voice_cast = edit_options['voice_cast']
    if voice_cast is not None:
        if not isinstance(voice_cast, dict):
            raise ValueError('voice-cast must be a JSON object')
        manifest.voice_cast = dict(voice_cast)
    if getattr(args, 'bgm_source', None) is not None:
        raw_bgm = args.bgm_source
        manifest.bgm_source = None if str(raw_bgm).strip().lower() in {'', 'none'} else str(raw_bgm).strip()
    elif 'bgm_source' in edit_options:
        raw_bgm = edit_options['bgm_source']
        manifest.bgm_source = None if raw_bgm is None or str(raw_bgm).strip().lower() in {'', 'none'} else str(raw_bgm).strip()
    if manifest.speech_mode != prev_speech_mode or manifest.voice_cast != prev_voice_cast:
        for shot in manifest.shots:
            shot.assets.pop('audio', None)
    from .editing import AUTO_ASPECT_IMAGE_SIZES, default_image_size_for_aspect, validate_edit_settings
    validate_edit_settings(
        aspect_ratio=manifest.aspect_ratio,
        scale_mode=manifest.scale_mode,
        alignment_mode=manifest.alignment_mode,
        subtitle_style=manifest.subtitle_style,
        speech_mode=manifest.speech_mode,
        bgm_volume=manifest.bgm_volume,
    )
    from .character_bible import validate_consistency_settings
    cons_mode = getattr(args, 'consistency_mode', None)
    if cons_mode is None and edit_options.get('consistency_mode') is not None:
        cons_mode = edit_options['consistency_mode']
    if cons_mode is not None:
        manifest.consistency_mode = cons_mode
    style_preset = getattr(args, 'style_preset', None)
    if style_preset is None and 'style_preset' in edit_options:
        style_preset = edit_options['style_preset']
    if style_preset is not None or 'style_preset' in edit_options:
        raw_style = '' if style_preset is None else str(style_preset).strip()
        manifest.style_preset = None if raw_style.lower() in {'', 'none'} else raw_style
    manifest.consistency_mode, manifest.style_preset = validate_consistency_settings(
        manifest.consistency_mode,
        manifest.style_preset,
    )
    if getattr(args, 'mode', None): manifest.mode = args.mode
    video_route = getattr(args, 'video_provider', None)
    video_options = getattr(args, 'video_options', None)
    from .project_video import validate_video_options, effective_video_options
    from .video_router import validate_routing_settings
    if video_options is not None:
        validate_video_options(video_options)
    next_route = video_route if video_route is not None else manifest.video_provider
    next_options = video_options if video_options is not None else manifest.video_options
    next_policy = getattr(args, 'video_routing_policy', None)
    if next_policy is None and edit_options.get('video_routing_policy') is not None:
        next_policy = edit_options['video_routing_policy']
    if next_policy is None:
        next_policy = manifest.video_routing_policy
    raw_fallback = getattr(args, 'video_fallback_provider', None)
    has_fallback_update = raw_fallback is not None or 'video_fallback_provider' in edit_options
    if raw_fallback is None and 'video_fallback_provider' in edit_options:
        raw_fallback = edit_options['video_fallback_provider']
    if has_fallback_update:
        cleaned_fb = '' if raw_fallback is None else str(raw_fallback).strip()
        next_fallback = None if cleaned_fb.lower() in {'', 'none'} else cleaned_fb
    else:
        next_fallback = manifest.video_fallback_provider
    next_route, next_policy, next_fallback = validate_routing_settings(
        next_route,
        next_policy,
        next_fallback,
    )
    changed = (
        next_route != manifest.video_provider
        or next_policy != manifest.video_routing_policy
        or next_fallback != manifest.video_fallback_provider
        or effective_video_options(next_route, next_options) != effective_video_options(manifest.video_provider, manifest.video_options)
    )
    manifest.video_provider = next_route
    manifest.video_options = next_options
    manifest.video_routing_policy = next_policy
    manifest.video_fallback_provider = next_fallback
    if changed:
        for shot in manifest.shots:
            shot.assets.pop('video', None)
            shot.mode = 'preview'
            shot.error = None
    for field in ('image_source', 'image_model', 'image_size'):
        value = getattr(args, field, None)
        if value is not None:
            setattr(manifest, field, value)
    raw_budget = args.budget_limit_cny if getattr(args, 'budget_limit_cny', None) is not None else edit_options.get('budget_limit_cny')
    if raw_budget is not None:
        if float(raw_budget) <= 0:
            raise ValueError('budget-limit must be positive')
        manifest.budget_limit_cny = float(raw_budget)
    if getattr(args, 'flywheel_dir', None) is not None:
        raw_fw = str(args.flywheel_dir).strip()
        manifest.flywheel_path = None if raw_fw.lower() in {'', 'none'} else str(Path(raw_fw).resolve())
    raw_rework = args.auto_rework if getattr(args, 'auto_rework', None) is not None else edit_options.get('auto_rework')
    if raw_rework is not None:
        manifest.auto_rework = bool(raw_rework)
    raw_rounds = args.max_rework_rounds if getattr(args, 'max_rework_rounds', None) is not None else edit_options.get('max_rework_rounds')
    if raw_rounds is not None:
        if int(raw_rounds) < 0:
            raise ValueError('max-rework-rounds must be non-negative')
        manifest.max_rework_rounds = int(raw_rounds)
    raw_relay = args.relay_tail_frames if getattr(args, 'relay_tail_frames', None) is not None else edit_options.get('relay_tail_frames')
    if raw_relay is not None:
        manifest.relay_tail_frames = bool(raw_relay)
    raw_exec = getattr(args, 'execution_strategy', None) or edit_options.get('execution_strategy')
    if raw_exec is not None:
        clean_exec = str(raw_exec).strip().lower()
        if clean_exec not in {'interleaved', 'stage_batched'}:
            raise ValueError(f'unsupported execution-strategy: {raw_exec!r}')
        manifest.execution_strategy = clean_exec
    raw_lip = getattr(args, 'lipsync_provider', None) or edit_options.get('lipsync_provider')
    if raw_lip is not None:
        clean_lip = str(raw_lip).strip().lower()
        if clean_lip not in {'off', 'http', 'comfyui', 'command'}:
            raise ValueError(f'unsupported lipsync-provider: {raw_lip!r}')
        manifest.lipsync_provider = clean_lip
    raw_split = args.auto_split_shots if getattr(args, 'auto_split_shots', None) is not None else edit_options.get('auto_split_shots')
    if raw_split is not None:
        manifest.auto_split_shots = bool(raw_split)
    raw_max_sec = args.max_shot_seconds if getattr(args, 'max_shot_seconds', None) is not None else edit_options.get('max_shot_seconds')
    if raw_max_sec is not None:
        if float(raw_max_sec) <= 1.0:
            raise ValueError('max-shot-seconds must be > 1.0')
        manifest.max_shot_seconds = float(raw_max_sec)
    if manifest.aspect_ratio != prev_aspect_ratio:
        explicit_image_size = getattr(args, 'image_size', None)
        if explicit_image_size is None and manifest.image_size in AUTO_ASPECT_IMAGE_SIZES.values():
            manifest.image_size = default_image_size_for_aspect(manifest.aspect_ratio)
        for shot in manifest.shots:
            shot.assets.pop('video', None)
            shot.mode = 'preview'
            shot.error = None
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
    target_stage = str(getattr(args, 'target_stage', None) or edit_options.get('target_stage') or 'all').strip().lower()
    if target_stage not in {'all', 'keyframes', 'tts', 'video', 'assemble'}:
        raise ValueError(f'unsupported target_stage: {target_stage!r}')
    if manifest.mode == 'render' and target_stage in {'all', 'video'}:
        from .project_video import build_project_video_provider, needs_video_generation
        try:
            if needs_video_generation(manifest):
                video_provider = build_project_video_provider(manifest, project)
        except Exception as exc:
            manifest.status = 'failed'
            manifest.error = f'video provider setup failed: {type(exc).__name__}: {exc}'
            manifest.save(manifest_path)
            raise
    output = render_preview(
        manifest,
        project,
        video_provider=video_provider,
        real_preview=getattr(args, 'real_preview', None),
        enforce_review=enforce_review,
        target_stage=target_stage,
    )
    if output and (getattr(args, 'export_jianying', False) or bool(edit_options.get('export_jianying'))):
        from .jianying import export_jianying_draft
        export_jianying_draft(manifest, project)
    manifest.save(manifest_path); print(f'preview: {output}' if output else manifest.status); return 0

if __name__ == '__main__': raise SystemExit(main())
