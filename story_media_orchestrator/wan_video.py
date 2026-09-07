"""Connect the original Wan adapter to durable tasks and local playback."""
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .video_checkpoint import execute_with_receipt


def download_video(url):
    parsed = urlsplit(url)
    if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("invalid video download URL")
    # A fresh request deliberately carries no model API authorization header.
    from story_video_agent.http_adapter import ComfyUIAdapter
    with urlopen(Request(url), timeout=120) as response:
        return ComfyUIAdapter._read_limited(response, 512 * 1024 * 1024)


class WanSubmission:
    def __init__(self, provider):
        self.provider = provider
        self.base_url = "wan:" + provider.base_url

    def submit(self, parameters):
        if parameters.get('first_frame_url'):
            payload = {'model': parameters['model'], 'input': {
                'prompt': parameters['prompt'], 'first_frame_url': parameters['first_frame_url'],
                'last_frame_url': parameters['last_frame_url']},
                'parameters': {'resolution': parameters['resolution'], 'prompt_extend': True}}
            if parameters.get('negative_prompt'):
                payload['input']['negative_prompt'] = parameters['negative_prompt']
            result = self.provider._call('POST', '/services/aigc/image2video/video-synthesis', payload)
            return result.get('output', {}).get('task_id')
        return self.provider.submit(**{k: v for k, v in parameters.items() if k != "batch_id"})


def validate_wan_parameters(*, model="wanx2.1-t2v-turbo", duration_seconds=5, size="1280*720"):
    if type(duration_seconds) is not int or not 1 <= duration_seconds <= 10:
        raise ValueError("duration must be an integer from 1 to 10")
    if model in {'wanx2.1-t2v-turbo', 'wan2.1-t2v-turbo'} and duration_seconds != 5:
        raise ValueError('wanx2.1-t2v-turbo 仅支持 5 秒视频，请调整时长')
    if model == 'wan2.2-kf2v-flash' and duration_seconds != 5:
        raise ValueError('wan2.2-kf2v-flash only supports 5 seconds')
    if any(not isinstance(value, str) or not value.strip() for value in (model, size)):
        raise ValueError("video model and size are required")


def run_wan_video(provider, registry, *, prompt, model="wanx2.1-t2v-turbo",
                  duration_seconds=5, size="1280*720", negative_prompt=None, batch_id=None, downloader=download_video,
                  first_frame=None, last_frame=None, resolution='720P'):
    if batch_id is not None and (not isinstance(batch_id, str) or not batch_id.strip() or len(batch_id) > 128):
        raise ValueError("invalid video batch_id")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("video prompt is required")
    validate_wan_parameters(model=model, duration_seconds=duration_seconds, size=size)
    if negative_prompt is not None and not isinstance(negative_prompt, str):
        raise ValueError("negative prompt must be text")
    parameters = {"prompt": prompt, "model": model, "duration_seconds": duration_seconds,
                  "size": size, "negative_prompt": negative_prompt}
    if model == 'wan2.2-kf2v-flash' or first_frame is not None or last_frame is not None:
        if model != 'wan2.2-kf2v-flash' or first_frame is None or last_frame is None:
            raise ValueError('first/last frame mode requires both frames and wan2.2-kf2v-flash')
        if resolution not in {'480P', '720P', '1080P'}:
            raise ValueError('unsupported keyframe resolution')
        import base64
        from pathlib import Path
        from PIL import Image
        for key, path in [('first_frame_url', first_frame), ('last_frame_url', last_frame)]:
            path = Path(path)
            if path.stat().st_size > 10 * 1024 * 1024:
                raise ValueError('keyframe exceeds 10 MiB')
            with Image.open(path) as image:
                mime = Image.MIME.get(image.format)
                if mime not in {'image/png', 'image/jpeg', 'image/webp'}:
                    raise ValueError('unsupported keyframe image format')
                image.verify()
            parameters[key] = f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode()
        parameters['resolution'] = resolution
    if batch_id is not None:
        parameters["batch_id"] = batch_id

    def execute(client):
        task_id = client.submit(parameters)
        result = provider.wait(task_id)
        if result.get("state") != "succeeded":
            raise RuntimeError(f"阿里云视频任务 {task_id} 状态：{result.get('state', 'unknown')}")
        if result.get("task_id") != task_id or not result.get("urls"):
            raise RuntimeError("阿里云视频结果与任务不匹配或缺少成片地址")
        content = downloader(result["urls"][0])
        ref = registry.put_bytes(content)
        return {"state": "succeeded", "prompt_id": task_id, "task_id": task_id, "artifact_ref": ref}

    execution = execute_with_receipt(WanSubmission(provider), parameters, registry, execute)
    return {"schema": "video-generation-result/v1", "status": "succeeded", "provider": "dashscope",
            "conditioning_mode": "first_last_frame" if first_frame is not None else "text_to_video", "review_status": "pending", "execution": execution}
