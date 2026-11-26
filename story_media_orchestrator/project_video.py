"""Use the original model-video adapter inside a project render run."""
import inspect
import math
import os
from pathlib import Path
import re

from .adapters import StoryVideoAdapter
from .config import ModelConfig
from .registry import ArtifactRegistry
from .runtime import _import_from_root
from .video_router import (
    default_wan_size_for_aspect,
    resolve_shot_video_route,
    validate_routing_settings,
)

from .wan_video import is_wan_i2v_model, is_wan_kf2v_model, is_wan_r2v_model


def parse_optional_bool(value, *, name: str = "flag"):
    """Return True/False when set, or None when omitted.

    Explicit False must not be swallowed by ``value or True``. Non-bool
    values (including 0/1) are rejected so seed 0 cannot be confused with
    a last-frame switch.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be true, false, or omitted; got {type(value).__name__}")



def effective_video_options(provider, options):
    defaults = {
        'comfyui': {'width': 864, 'height': 480, 'length': 124, 'fps': 24, 'seed': 0, 'turbo': False},
        'dashscope': {'model': 'wanx2.1-t2v-turbo', 'size': '1280*720', 'duration_seconds': 5},
    }
    if provider == 'auto':
        target = 'dashscope'
    elif provider in defaults:
        target = provider
    else:
        raise ValueError('unsupported project video provider')
    result = {key: options.get(key, value) for key, value in defaults[target].items()}
    if 'workflow_mode' in options:
        result['workflow_mode'] = options['workflow_mode']
    if target == 'comfyui' and 'resolution' in options:
        result['resolution'] = options['resolution']
    if target == 'dashscope' and (
        result['model'] in {'wan2.2-kf2v-flash', 'wanx2.1-kf2v-plus'}
        or is_wan_i2v_model(result['model'])
        or 'resolution' in options
    ):
        result['resolution'] = options.get('resolution', '720P')
    return result


def validate_video_options(options):
    allowed = {
        'model', 'size', 'duration_seconds', 'width', 'height',
        'length', 'fps', 'seed', 'turbo', 'resolution', 'workflow_mode',
        'disable_fl2v_span_gate', 'frame_grid',
    }
    if not isinstance(options, dict) or options.keys() - allowed:
        raise ValueError('unsupported video options')
    if options.get('resolution', '720P') not in {'480P', '720P', '1080P'}:
        raise ValueError('invalid video resolution')
    if 'workflow_mode' in options and options['workflow_mode'] not in {'t2v', 'i2v'}:
        raise ValueError('invalid video workflow_mode')
    if 'frame_grid' in options and options['frame_grid'] not in {'4k+1', 'minimax_h3'}:
        raise ValueError('invalid video frame_grid')
    for key in ('model', 'size'):
        if key in options and (not isinstance(options[key], str) or not options[key].strip()):
            raise ValueError(f'{key} must be nonempty text')
    for key, low, high in [('duration_seconds', 1, 10), ('width', 64, 4096),
                           ('height', 64, 4096), ('length', 1, 4096), ('seed', 0, 2**63 - 1)]:
        if key in options and (type(options[key]) is not int or not low <= options[key] <= high):
            raise ValueError(f'invalid video {key}')
    fps = options.get('fps', 24)
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError('invalid video fps')
    if options.get('length', 124) / fps < 1:
        raise ValueError('video duration must be at least one second')
    if 'turbo' in options and type(options['turbo']) is not bool:
        raise ValueError('video turbo must be boolean')
    if 'disable_fl2v_span_gate' in options and type(options['disable_fl2v_span_gate']) is not bool:
        raise ValueError('video disable_fl2v_span_gate must be boolean')


def needs_video_generation(manifest):
    from .preview import _shot_cache_key
    total = len(manifest.shots)
    for idx, shot in enumerate(manifest.shots):
        if (shot.status not in {'generated', 'done'}
                or shot.cache_key not in {_shot_cache_key(manifest, shot), _shot_cache_key(manifest, shot, 'render')}
                or not {'image', 'audio'} <= shot.assets.keys()
                or not all(Path(value).is_file() for value in shot.assets.values())):
            return True
        route_info = resolve_shot_video_route(
            manifest,
            shot,
            has_first_frame=bool(shot.assets.get('image')),
            has_last_frame=bool(shot.assets.get('last_frame')),
            shot_index=idx,
            total_shots=total,
        )
        if route_info['route'] == 'preview' or (shot.scene_context or {}).get('_video_route') == 'preview':
            continue
        if shot.mode != 'render' or 'video' not in shot.assets or not Path(shot.assets['video']).is_file():
            return True
    return False


_H3_CAMERA_TOKENS = {
    'push_in': '[Push in]',
    'pull_out': '[Pull out]',
    'pan': '[Tracking shot]',
    'pan_left': '[Pan left]',
    'pan_up': '[Tilt up]',
    'ken_burns': '[Slow push in]',
}
_FREEZE_PHRASES_RE = re.compile(
    r'(?:神色不动|纹丝不动|原地静止|严格锁定[^，。；;]*|全程闭唇[^，。；;]*|双唇紧闭[^，。；;]*|禁止[^，。；;]*|严禁[^，。；;]*|不要[^，。；;]*)'
)
_META_NOISE_RE = re.compile(r'(?:->|→|⇒|contact[_ -]?sheet|2x2宫格|拆帧宫格|frame_grid)', re.IGNORECASE)


def _sanitize_h3_kinetic_text(raw: str, *, strip_quotes: bool = False) -> str:
    text = str(raw or '').strip()
    if strip_quotes:
        text = re.sub(r'[“"「『][^”"」』]{1,80}[”"」』]', '', text)
    text = _META_NOISE_RE.sub(' ', text)
    text = _FREEZE_PHRASES_RE.sub('', text)
    parts = [seg.strip(' ，,。；;：:') for seg in re.split(r'[；;\n]+', text)]
    cleaned = '，'.join(p for p in parts if p)
    return re.sub(r'[，,]{2,}', '，', cleaned).strip(' ，,。；;')



def resolve_video_action_text(shot, context: dict | None = None) -> tuple[str, str]:
    """Resolve the shot action text used inside a video prompt.

    `prompt_override` is explicit action/visual text for this shot, not a fully
    assembled prompt. A non-empty override takes precedence over persisted
    `scene_context.kinetic_action` / `visual_action`. Empty or whitespace-only
    values are treated as absent so default precedence is unchanged. This
    function does not mutate the shot or scene context.
    """
    override = getattr(shot, 'prompt_override', None) if shot is not None else None
    if isinstance(override, str) and override.strip():
        return override.strip(), 'prompt_override'
    ctx = context if isinstance(context, dict) else {}
    for key, source in (('kinetic_action', 'kinetic_action'), ('visual_action', 'visual_action')):
        value = ctx.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), source
    shot_text = getattr(shot, 'text', None) if shot is not None else None
    if isinstance(shot_text, str) and shot_text.strip():
        return shot_text.strip(), 'shot_text'
    return '', 'empty'


def _filter_adapter_kwargs(adapter_fn, kwargs):
    try:
        sig = inspect.signature(adapter_fn)
    except (TypeError, ValueError):
        return kwargs
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


class ProjectVideoProvider:
    def __init__(
        self,
        adapter,
        registry,
        *,
        alignment_mode: str = 'pad',
        tts_source: str = 'silent',
        manifest=None,
        adapters_by_provider=None,
        fallback_provider: str | None = None,
    ):
        self.adapter, self.registry = adapter, registry
        self.alignment_mode = alignment_mode
        self.tts_source = tts_source
        self.manifest = manifest
        self.adapters_by_provider = adapters_by_provider or {}
        self.fallback_provider = fallback_provider

    def generate(self, shot, image, output, *, use_last_frame=None, final_prompt_override=None, visual_frame_count=None, persist_timing_on_shot=True):
        frame_ref = self.registry.put_bytes(Path(image).read_bytes()) if image else None
        context = shot.scene_context
        use_last = parse_optional_bool(use_last_frame, name='use_last_frame')
        frozen_prompt = (
            final_prompt_override.strip()
            if isinstance(final_prompt_override, str) and final_prompt_override.strip()
            else None
        )
        prompt_parts = [getattr(shot, 'prompt_override', None) or context.get('visual_action') or shot.text]
        for key, label in [('location', '地点'), ('lighting', '光线'), ('time', '时间'),
                           ('framing', '构图'), ('mood', '氛围'), ('continuity', '连续性')]:
            value = context.get(key)
            if isinstance(value, str) and value.strip():
                prompt_parts.append(f'{label}：{value.strip()}')

        resolved_chars = []
        ref_image_paths = []
        if self.manifest is not None:
            from .character_bible import format_character_visuals, resolve_shot_characters
            resolved_chars = resolve_shot_characters(self.manifest, shot)
            for char in resolved_chars:
                ref = char.get('ref_image')
                if isinstance(ref, str) and ref.strip() and Path(ref.strip()).is_file():
                    ref_image_paths.append(ref.strip())

        if resolved_chars:
            prompt_parts.append('人物：' + format_character_visuals(resolved_chars, max_chars=160))
        elif context.get('characters'):
            from .providers import StoryImageProvider
            prompt_parts.append('人物：' + StoryImageProvider._visual_characters(context['characters']))

        last_asset = shot.assets.get('last_frame')
        last_asset_ok = bool(last_asset and Path(last_asset).is_file())
        # Routing still sees the on-disk asset so an explicit comfy_fl2v
        # route/model is not downgraded. Attachment is decided separately.
        if use_last is False:
            last_ref = None
        else:
            last_ref = self.registry.put_bytes(Path(last_asset).read_bytes()) if last_asset_ok else None

        route_info = None
        if self.manifest is not None:
            available = set(self.adapters_by_provider.keys()) or None
            shots_list = list(getattr(self.manifest, 'shots', []))
            try:
                shot_idx = next(i for i, item in enumerate(shots_list) if item.id == shot.id)
            except StopIteration:
                shot_idx = 0
            route_info = resolve_shot_video_route(
                self.manifest,
                shot,
                has_first_frame=frame_ref is not None,
                has_last_frame=last_asset_ok,
                has_character_refs=bool(ref_image_paths),
                available_providers=available,
                shot_index=shot_idx,
                total_shots=max(1, len(shots_list)),
            )
            shot.scene_context['_video_route'] = route_info['route']
            shot.scene_context['_video_model'] = route_info.get('model') or route_info['provider']

        if (
            route_info is not None
            and route_info.get('route') == 'comfy_fl2v'
            and last_ref is not None
            and image
            and Path(image).is_file()
            and shot.assets.get('last_frame')
            and Path(shot.assets['last_frame']).is_file()
            and not bool((getattr(self.manifest, 'video_options', None) or {}).get('disable_fl2v_span_gate'))
            and not bool((shot.scene_context or {}).get('disable_fl2v_span_gate'))
        ):
            from .codex_image import assess_fl2v_pair_compatibility

            require_delta = bool(
                (shot.scene_context or {}).get('require_pose_delta')
                or (shot.scene_context or {}).get('action_beat')
            )
            gate = assess_fl2v_pair_compatibility(
                image,
                shot.assets['last_frame'],
                require_pose_delta=require_delta,
            )
            shot.scene_context['_fl2v_span_gate'] = gate
            if not gate.get('compatible', True):
                route_info = {**route_info, 'route': 'comfy_i2v'}
                shot.scene_context['_video_route'] = 'comfy_i2v'
                last_ref = None

        scene_for_video = dict(shot.scene_context)
        mouth_guide = None
        if self.manifest is not None:
            from .editing import shot_has_character_presence
            from .lipsync import build_mouth_articulation_guidance

            mouth_guide = build_mouth_articulation_guidance(
                shot,
                getattr(self.manifest, 'character_bible', None),
                tts_source=getattr(self.manifest, 'tts_source', 'silent'),
            )
            shot.scene_context['_speech_delivery'] = mouth_guide['delivery']
            scene_for_video['_speech_delivery'] = mouth_guide['delivery']
            if (
                shot_has_character_presence(shot, getattr(self.manifest, 'character_bible', None))
                or mouth_guide['delivery'] == 'on_screen_dialogue'
            ):
                prompt_parts.append(f"口型控制：{mouth_guide['video_prompt_hint']}")
                if mouth_guide['delivery'] != 'on_screen_dialogue':
                    prompt_parts.append('纯环境声，画面角色全程双唇紧闭沉默')

        is_h3_frame_conditioned = bool(
            frame_ref is not None
            and route_info is not None
            and route_info.get('route') in {'comfy_i2v', 'comfy_fl2v'}
        )
        if frozen_prompt:
            # Request-scoped exact replay of the final H3 prompt. Skips
            # kinetic assembly and the t46u VFX suffix for this call only.
            final_video_prompt = frozen_prompt
        elif is_h3_frame_conditioned:
            delivery = (mouth_guide or {}).get('delivery', 'voiceover')
            cam_tok = str(context.get('camera_move') or '').strip() or _H3_CAMERA_TOKENS.get(
                getattr(shot, 'motion', 'none'), ''
            )
            raw_act, action_source = resolve_video_action_text(shot, context)
            clean_act = _sanitize_h3_kinetic_text(
                raw_act,
                strip_quotes=(delivery != 'on_screen_dialogue'),
            )
            if not clean_act:
                clean_act = _sanitize_h3_kinetic_text(shot.text, strip_quotes=False) or '镜头平稳推进，主体顺势完成单拍动作'
            kinetic_parts = []
            if cam_tok:
                kinetic_parts.append(f'{cam_tok} {clean_act}')
            else:
                kinetic_parts.append(clean_act)
            # An explicit prompt_override is the action text for this request.
            # Do not also append the persisted impact_result, which can contradict it.
            # Default (no override) keeps the existing impact_result suffix.
            if action_source != 'prompt_override':
                impact_res = _sanitize_h3_kinetic_text(str(context.get('impact_result') or ''))
                if impact_res:
                    kinetic_parts.append(f'可见结果：{impact_res}')
            if delivery == 'on_screen_dialogue' and mouth_guide is not None:
                kinetic_parts.append(f"口型控制：{mouth_guide['video_prompt_hint']}")
            final_video_prompt = '；'.join(kinetic_parts)
        else:
            final_video_prompt = '；'.join(prompt_parts)

        neg_override = getattr(shot, 'negative_override', None)
        if isinstance(neg_override, str) and neg_override.strip():
            existing_neg = str(scene_for_video.get('negative') or '').strip()
            scene_for_video['negative'] = f"{existing_neg}，{neg_override.strip()}" if existing_neg else neg_override.strip()

        if mouth_guide is not None and (
            shot_has_character_presence(shot, getattr(self.manifest, 'character_bible', None))
            or mouth_guide['delivery'] == 'on_screen_dialogue'
        ):
            mouth_neg = str(mouth_guide.get('negative_hint') or '').strip()
            if mouth_neg:
                existing_neg = str(scene_for_video.get('negative') or '').strip()
                scene_for_video['negative'] = f"{existing_neg}，{mouth_neg}" if existing_neg else mouth_neg

        if not frozen_prompt:
            from .vfx_spatial_standard import apply_generation_vfx_standard
            final_video_prompt, scene_for_video['negative'] = apply_generation_vfx_standard(
                shot,
                context,
                final_video_prompt,
                scene_for_video.get('negative'),
            )

        base_kwargs = {
            'first_frame_ref': frame_ref,
            'last_frame_ref': last_ref,
            'story_spans': shot.scene_context.get('source_spans') or [shot.scene_id, shot.id],
            'scene': scene_for_video,
            'prompt': final_video_prompt,
        }
        seed_override = getattr(shot, 'seed_override', None) if shot is not None else None
        explicit_length = None
        if visual_frame_count is not None:
            from .lipsync import exact_minimax_h3_duration_seconds, resolve_explicit_minimax_h3_frame_length
            explicit_length = resolve_explicit_minimax_h3_frame_length(visual_frame_count)
            shot_dur = exact_minimax_h3_duration_seconds(explicit_length)
        else:
            shot_dur = float(getattr(shot, 'duration', 0.0) or 0.0)
        extended_kwargs = {
            **base_kwargs,
            'shot_duration': shot_dur,
            **({'route': route_info} if route_info is not None else {}),
            **({'ref_images': ref_image_paths} if ref_image_paths else {}),
            **({'seed': int(seed_override)} if seed_override is not None else {}),
            **(
                {'workflow_mode': route_info['workflow_mode']}
                if route_info and route_info.get('workflow_mode') in {'t2v', 'i2v'}
                else {}
            ),
            **({'prompt_is_final': True} if frozen_prompt else {}),
            **({'length': explicit_length} if explicit_length is not None else {}),
        }

        primary_key = route_info['provider'] if route_info else None
        primary_adapter = self.adapters_by_provider.get(primary_key, self.adapter)
        try:
            call_kwargs = _filter_adapter_kwargs(primary_adapter.run, extended_kwargs)
            result = primary_adapter.run(**call_kwargs)
        except Exception:
            fb_key = self.fallback_provider
            fb_adapter = self.adapters_by_provider.get(fb_key) if fb_key else None
            if fb_adapter is None or fb_adapter is primary_adapter:
                raise
            fb_route = {
                **(route_info or {}),
                'provider': fb_key,
                'route': f'fallback:{fb_key}',
                'workflow_mode': 'i2v' if frame_ref else 't2v',
                'model': 'wan2.2-i2v-flash' if (fb_key == 'dashscope' and frame_ref) else ('wanx2.1-t2v-turbo' if fb_key == 'dashscope' else None),
            }
            fb_kwargs = {
                **base_kwargs,
                'shot_duration': shot_dur,
                'route': fb_route,
                **({'ref_images': ref_image_paths} if ref_image_paths else {}),
                **({'seed': int(seed_override)} if seed_override is not None else {}),
                'workflow_mode': fb_route['workflow_mode'],
            }
            result = fb_adapter.run(**_filter_adapter_kwargs(fb_adapter.run, fb_kwargs))
            shot.scene_context['_video_fallback_from'] = primary_key or getattr(self.manifest, 'video_provider', 'unknown')
            shot.scene_context['_video_route'] = fb_route['route']
            shot.scene_context['_video_model'] = fb_route.get('model') or fb_key

        execution = result.get('execution', {})
        if execution.get('state') != 'succeeded':
            raise RuntimeError('model video has not completed')
        content = self.registry.get_bytes(execution['artifact_ref'])
        # Check playback container before assigning a completed video asset.
        self.registry.video_preview(execution['artifact_ref'])
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix('.tmp')
        temporary.write_bytes(content)
        temporary.replace(output)
        from .editing import probe_duration, resolve_elastic_shot_duration
        from .local_media import find_ffmpeg
        video_seconds = probe_duration(find_ffmpeg(), output)
        if video_seconds is not None and persist_timing_on_shot:
            shot.scene_context['_video_duration'] = video_seconds
            shot.duration = resolve_elastic_shot_duration(
                shot.duration,
                video_seconds,
                tts_source=self.tts_source,
                alignment_mode=self.alignment_mode,
            )
        return output


def build_project_video_provider(manifest, root):
    options = manifest.video_options
    validate_video_options(options)
    provider_setting, policy, fallback_provider = validate_routing_settings(
        getattr(manifest, 'video_provider', 'comfyui'),
        getattr(manifest, 'video_routing_policy', 'manual'),
        getattr(manifest, 'video_fallback_provider', None),
    )

    if provider_setting == 'dashscope' and policy == 'manual':
        from .wan_video import validate_wan_parameters
        eff = effective_video_options('dashscope', options)
        validate_wan_parameters(**{k: v for k, v in eff.items() if k not in {'resolution', 'workflow_mode'}})
        if eff.get('model') in {'wan2.2-kf2v-flash', 'wanx2.1-kf2v-plus'} and any(
            not s.assets.get('last_frame') for s in manifest.shots
        ):
            raise ValueError('Select a last-frame candidate for every shot before first/last-frame video generation')

    from .runtime import resolve_sibling_root
    video_root = Path(os.environ.get('STORY_VIDEO_AGENT_ROOT')
                      or resolve_sibling_root('story-video-agent', 'microcodex-story-video-agent', module_name='story_video_agent'))
    package = _import_from_root(video_root, 'story_video_agent')
    registry = ArtifactRegistry(Path(root) / 'artifacts')

    def build_dashscope_adapter():
        image_root = Path(os.environ.get('STORY_IMAGE_AGENT_ROOT')
                          or resolve_sibling_root('story-image-agent', 'microcodex-story-image-agent', module_name='story_image_agent'))
        image_package = _import_from_root(image_root, 'story_image_agent')
        configured = image_package.DashScopeImageProvider.from_nanocodex_config()
        from story_video_agent.dashscope import DashScopeWanAdapter
        from .wan_video import run_wan_video
        provider = DashScopeWanAdapter(configured.api_key, base_url=configured.base_url)

        class WanProjectAdapter:
            def run(self, **kwargs):
                scene = kwargs.get('scene') or {}
                route_info = kwargs.get('route')
                if route_info is None and 'first_frame_ref' not in kwargs:
                    frames = {}
                    if options.get('model') == 'wan2.2-kf2v-flash' and kwargs.get('first_frame_ref') and kwargs.get('last_frame_ref'):
                        frames = {
                            'first_frame': registry.root / kwargs['first_frame_ref'].rsplit('/', 1)[-1],
                            'last_frame': registry.root / kwargs['last_frame_ref'].rsplit('/', 1)[-1],
                            'resolution': options.get('resolution', '720P'),
                        }
                    return run_wan_video(
                        provider,
                        registry,
                        prompt=kwargs['prompt'],
                        negative_prompt=scene.get('negative'),
                        **frames,
                        **{k: options[k] for k in ('model', 'size', 'duration_seconds') if k in options},
                    )

                model_name = (route_info or {}).get('model') or options.get('model') or 'wanx2.1-t2v-turbo'
                size = options.get('size') or default_wan_size_for_aspect(getattr(manifest, 'aspect_ratio', '16:9'))
                duration_seconds = options.get('duration_seconds', 5)
                resolution = options.get('resolution') or (
                    '1080P'
                    if (policy == 'quality_first' or (route_info or {}).get('route') == 'i2v_quality')
                    else '720P'
                )
                first_path = (
                    registry.root / kwargs['first_frame_ref'].rsplit('/', 1)[-1]
                    if kwargs.get('first_frame_ref')
                    else None
                )
                last_path = (
                    registry.root / kwargs['last_frame_ref'].rsplit('/', 1)[-1]
                    if kwargs.get('last_frame_ref')
                    else None
                )
                frames = {}
                if is_wan_kf2v_model(model_name):
                    if not first_path or not last_path:
                        raise ValueError('Select a last-frame candidate for every shot before first/last-frame video generation')
                    frames = {'first_frame': first_path, 'last_frame': last_path, 'resolution': resolution}
                elif is_wan_i2v_model(model_name) and first_path is not None:
                    frames = {'first_frame': first_path, 'resolution': resolution}
                elif is_wan_r2v_model(model_name):
                    ref_paths = [Path(p) for p in (kwargs.get('ref_images') or []) if Path(p).is_file()]
                    if not ref_paths and first_path is not None and first_path.is_file():
                        ref_paths = [first_path]
                    if ref_paths:
                        frames = {'ref_images': ref_paths}

                return run_wan_video(
                    provider,
                    registry,
                    prompt=kwargs['prompt'],
                    negative_prompt=scene.get('negative'),
                    model=model_name,
                    size=size,
                    duration_seconds=duration_seconds,
                    **frames,
                )

        return WanProjectAdapter()

    def build_comfy_adapter():
        comfy_client = package.ComfyUIAdapter.from_environment()
        if comfy_client is not None and hasattr(comfy_client, 'timeout'):
            comfy_client.timeout = float(os.environ.get('MINIMAX_H3_COMFYUI_TIMEOUT') or '180.0')
        adapter = StoryVideoAdapter(
            package.VideoPromptWorkflow(manifest.project_id),
            comfy=comfy_client,
            registry=registry,
            models=ModelConfig(video_turbo=options.get('turbo', False)),
        )

        class ComfyProjectAdapter:
            def run(self, **kwargs):
                from .editing import COMFY_ASPECT_SIZES
                from .lipsync import recommend_minimax_h3_frame_length, recommend_video_frame_length

                wf_mode = kwargs.get('workflow_mode') or options.get('workflow_mode') or 't2v'
                char_refs = [p for p in (kwargs.get('ref_images') or []) if isinstance(p, str) and p.strip()]
                shot_dur = kwargs.get('shot_duration')
                call_args = {
                    k: v
                    for k, v in kwargs.items()
                    if k not in {'route', 'ref_images', 'workflow_mode', 'seed', 'shot_duration', 'prompt_is_final', 'length'}
                }
                dim_kwargs = {k: options[k] for k in ('width', 'height', 'length', 'fps', 'seed') if k in options}
                if kwargs.get('length') is not None:
                    from .lipsync import resolve_explicit_minimax_h3_frame_length
                    dim_kwargs['length'] = resolve_explicit_minimax_h3_frame_length(int(kwargs['length']))
                elif (
                    'length' not in options
                    and getattr(manifest, 'tts_source', 'silent') != 'silent'
                    and isinstance(shot_dur, (int, float))
                    and shot_dur > 0
                ):
                    if options.get('frame_grid') == 'minimax_h3':
                        dim_kwargs['length'] = recommend_minimax_h3_frame_length(
                            float(shot_dur),
                            fps=float(dim_kwargs.get('fps', 24.0)),
                        )
                    else:
                        dim_kwargs['length'] = recommend_video_frame_length(
                            float(shot_dur),
                            fps=float(dim_kwargs.get('fps', 24.0)),
                        )
                if kwargs.get('seed') is not None:
                    dim_kwargs['seed'] = int(kwargs['seed'])
                aspect = getattr(manifest, 'aspect_ratio', '16:9')
                if 'width' not in options and 'height' not in options and aspect in COMFY_ASPECT_SIZES and aspect != '16:9':
                    dim_kwargs['width'], dim_kwargs['height'] = COMFY_ASPECT_SIZES[aspect]
                route_name = (kwargs.get('route') or {}).get('route')
                upscale = options.get('resolution')
                if not upscale:
                    if route_name == 'comfy_fl2v':
                        upscale = '1080P' if policy == 'quality_first' else '720P'
                    elif route_name == 'i2v_quality' and policy == 'quality_first':
                        upscale = '1080P'
                if upscale:
                    dim_kwargs['upscale_resolution'] = upscale
                raw_prompt = str(call_args.get('prompt') or '').strip()
                canvas_suffix = '；电影级纯净实景画面构图'
                if raw_prompt and not kwargs.get('prompt_is_final') and not raw_prompt.endswith(canvas_suffix):
                    call_args['prompt'] = f'{raw_prompt}{canvas_suffix}'
                return adapter.run(
                    **call_args,
                    workflow_mode=wf_mode,
                    **({'character_ref_images': char_refs} if char_refs else {}),
                    **dim_kwargs,
                )

        return ComfyProjectAdapter()

    adapters_by_provider = {}
    if provider_setting == 'dashscope':
        adapters_by_provider['dashscope'] = build_dashscope_adapter()
        if fallback_provider == 'comfyui' or any(s.video_route in {'comfy_i2v', 'comfy_fl2v', 'comfy_t2v'} for s in manifest.shots):
            adapters_by_provider['comfyui'] = build_comfy_adapter()
        primary_adapter = adapters_by_provider['dashscope']
    elif provider_setting == 'comfyui':
        adapters_by_provider['comfyui'] = build_comfy_adapter()
        if fallback_provider == 'dashscope' or any(
            s.video_route in {'i2v_fast', 'i2v_quality', 'kf2v', 'r2v', 't2v'} for s in manifest.shots
        ):
            adapters_by_provider['dashscope'] = build_dashscope_adapter()
        primary_adapter = adapters_by_provider['comfyui']
    else:
        # auto mode: initialize any available backends without crashing if preview fallback applies
        dash_err = None
        comfy_err = None
        try:
            adapters_by_provider['dashscope'] = build_dashscope_adapter()
        except Exception as exc:
            dash_err = exc
        if (
            os.environ.get('MINIMAX_H3_COMFYUI_BASE_URL', '').strip()
            or os.environ.get('COMFYUI_URL', '').strip()
            or os.environ.get('COMFYUI_HOST', '').strip()
            or fallback_provider == 'comfyui'
        ):
            try:
                adapters_by_provider['comfyui'] = build_comfy_adapter()
            except Exception as exc:
                comfy_err = exc
        if not adapters_by_provider:
            if fallback_provider == 'preview':
                return None
            if all(
                resolve_shot_video_route(
                    manifest, s, has_first_frame=True, available_providers=set(), shot_index=idx, total_shots=len(manifest.shots)
                )['route'] == 'preview'
                for idx, s in enumerate(manifest.shots)
            ):
                return None
            raise RuntimeError(f'no auto video backend configured ({dash_err or comfy_err or "missing credentials"})')
        primary_adapter = adapters_by_provider.get('dashscope') or adapters_by_provider['comfyui']

    return ProjectVideoProvider(
        primary_adapter,
        registry,
        alignment_mode=getattr(manifest, 'alignment_mode', 'pad'),
        tts_source=getattr(manifest, 'tts_source', 'silent'),
        manifest=manifest,
        adapters_by_provider=adapters_by_provider,
        fallback_provider=fallback_provider,
    )
