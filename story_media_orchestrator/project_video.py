"""Use the original model-video adapter inside a project render run."""
import os
import re
import subprocess
import math
from pathlib import Path

from .adapters import StoryVideoAdapter
from .config import ModelConfig
from .registry import ArtifactRegistry
from .runtime import _import_from_root


def effective_video_options(provider, options):
    defaults = {
        'comfyui': {'width': 864, 'height': 480, 'length': 124, 'fps': 24, 'seed': 0, 'turbo': False},
        'dashscope': {'model': 'wanx2.1-t2v-turbo', 'size': '1280*720', 'duration_seconds': 5},
    }
    if provider not in defaults:
        raise ValueError('unsupported project video provider')
    result = {key: options.get(key, value) for key, value in defaults[provider].items()}
    if provider == 'dashscope' and result['model'] == 'wan2.2-kf2v-flash':
        result['resolution'] = options.get('resolution', '720P')
    return result


def validate_video_options(options):
    allowed = {'model', 'size', 'duration_seconds', 'width', 'height', 'length', 'fps', 'seed', 'turbo', 'resolution'}
    if not isinstance(options, dict) or options.keys() - allowed:
        raise ValueError('unsupported video options')
    if options.get('resolution', '720P') not in {'480P', '720P', '1080P'}:
        raise ValueError('invalid video resolution')
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


def needs_video_generation(manifest):
    from .preview import _shot_cache_key
    for shot in manifest.shots:
        if (shot.mode != 'render' or shot.status not in {'generated', 'done'}
                or shot.cache_key not in {_shot_cache_key(manifest, shot), _shot_cache_key(manifest, shot, 'render')}
                or not {'image', 'audio', 'video'} <= shot.assets.keys()
                or not all(Path(value).is_file() for value in shot.assets.values())):
            return True
    return False


class ProjectVideoProvider:
    def __init__(self, adapter, registry):
        self.adapter, self.registry = adapter, registry

    def generate(self, shot, image, output):
        frame_ref = self.registry.put_bytes(Path(image).read_bytes()) if image else None
        context = shot.scene_context
        prompt_parts = [context.get('visual_action') or shot.text]
        for key, label in [('location', '地点'), ('lighting', '光线'), ('time', '时间'),
                           ('framing', '构图'), ('mood', '氛围'), ('continuity', '连续性')]:
            value = context.get(key)
            if isinstance(value, str) and value.strip():
                prompt_parts.append(f'{label}：{value.strip()}')
        if context.get('characters'):
            from .providers import StoryImageProvider
            prompt_parts.append('人物：' + StoryImageProvider._visual_characters(context['characters']))
        last_ref = self.registry.put_bytes(Path(shot.assets['last_frame']).read_bytes()) if shot.assets.get('last_frame') else None
        result = self.adapter.run(first_frame_ref=frame_ref, last_frame_ref=last_ref,
            story_spans=shot.scene_context.get('source_spans') or [shot.scene_id, shot.id],
            scene=shot.scene_context,
            prompt='；'.join(prompt_parts))
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
        from .local_media import find_ffmpeg
        probe = subprocess.run([find_ffmpeg(), '-hide_banner', '-i', str(output)], capture_output=True)
        duration = re.search(rb'Duration: (\d+):(\d+):(\d+(?:\.\d+)?)', probe.stderr)
        if duration:
            hours, minutes, seconds = map(float, duration.groups())
            shot.duration = max(shot.duration, hours * 3600 + minutes * 60 + seconds)
        return output


def build_project_video_provider(manifest, root):
    options = manifest.video_options
    validate_video_options(options)
    if manifest.video_provider == 'dashscope':
        from .wan_video import validate_wan_parameters
        validate_wan_parameters(**{k:v for k,v in effective_video_options('dashscope', options).items() if k != 'resolution'})
        if options.get('model') == 'wan2.2-kf2v-flash' and any(not s.assets.get('last_frame') for s in manifest.shots):
            raise ValueError('Select a last-frame candidate for every shot before first/last-frame video generation')
    video_root = Path(os.environ.get('STORY_VIDEO_AGENT_ROOT',
                     str(Path(__file__).resolve().parents[2] / 'story-video-agent')))
    package = _import_from_root(video_root, 'story_video_agent')
    registry = ArtifactRegistry(Path(root) / 'artifacts')
    if manifest.video_provider == 'dashscope':
        image_root = Path(os.environ.get('STORY_IMAGE_AGENT_ROOT',
                         str(Path(__file__).resolve().parents[2] / 'story-image-agent')))
        image_package = _import_from_root(image_root, 'story_image_agent')
        configured = image_package.DashScopeImageProvider.from_nanocodex_config()
        from story_video_agent.dashscope import DashScopeWanAdapter
        from .wan_video import run_wan_video
        provider = DashScopeWanAdapter(configured.api_key, base_url=configured.base_url)
        class WanProjectAdapter:
            def run(self, **kwargs):
                scene = kwargs.get('scene') or {}
                frames = {}
                if options.get('model') == 'wan2.2-kf2v-flash':
                    frames = {'first_frame':registry.root/kwargs['first_frame_ref'].rsplit('/',1)[-1],
                              'last_frame':registry.root/kwargs['last_frame_ref'].rsplit('/',1)[-1],
                              'resolution':options.get('resolution','720P')}
                return run_wan_video(provider, registry, prompt=kwargs['prompt'],
                                     negative_prompt=scene.get('negative'),
                                     **frames,
                                     **{k: options[k] for k in ('model', 'size', 'duration_seconds') if k in options})
        return ProjectVideoProvider(WanProjectAdapter(), registry)
    if manifest.video_provider != 'comfyui':
        raise ValueError('unsupported project video provider')
    adapter = StoryVideoAdapter(package.VideoPromptWorkflow(manifest.project_id),
        comfy=package.ComfyUIAdapter.from_environment(), registry=registry,
        models=ModelConfig(video_turbo=options.get('turbo', False)))
    class ComfyProjectAdapter:
        def run(self, **kwargs):
            return adapter.run(**kwargs, **{k: options[k] for k in ('width', 'height', 'length', 'fps', 'seed') if k in options})
    return ProjectVideoProvider(ComfyProjectAdapter(), registry)
