"""Connect the original Wan adapter to durable tasks and local playback."""
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .video_checkpoint import execute_with_receipt

_WAN_I2V_MODELS = {
    "wan2.2-i2v-flash",
    "wan2.2-i2v-plus",
    "wanx2.1-i2v-turbo",
    "wanx2.1-i2v-plus",
    "wan2.1-i2v-turbo",
    "wan2.1-i2v-plus",
}
_WAN_KF2V_MODELS = {"wan2.2-kf2v-flash", "wanx2.1-kf2v-plus"}
_WAN_R2V_MODELS = {"wan2.1-vace-plus", "wanx2.1-vace-plus"}
_WAN_5S_ONLY_MODELS = {
    "wanx2.1-t2v-turbo",
    "wan2.1-t2v-turbo",
    "wan2.2-kf2v-flash",
    "wanx2.1-kf2v-plus",
    "wan2.2-i2v-flash",
    "wan2.2-i2v-plus",
    "wanx2.1-i2v-plus",
    "wan2.1-i2v-plus",
    "wan2.1-vace-plus",
    "wanx2.1-vace-plus",
}
_WAN_TURBO_I2V_DURATIONS = {3, 4, 5}


def is_wan_i2v_model(model: str | None) -> bool:
    return isinstance(model, str) and model.strip() in _WAN_I2V_MODELS


def is_wan_kf2v_model(model: str | None) -> bool:
    return isinstance(model, str) and model.strip() in _WAN_KF2V_MODELS


def is_wan_r2v_model(model: str | None) -> bool:
    return isinstance(model, str) and model.strip() in _WAN_R2V_MODELS


def _encode_image_data_uri(image_path: str | Path) -> str:
    path = Path(image_path)
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("keyframe exceeds 10 MiB")
    from PIL import Image

    with Image.open(path) as image:
        mime = Image.MIME.get(image.format)
        if mime not in {"image/png", "image/jpeg", "image/webp"}:
            raise ValueError("unsupported keyframe image format")
        image.verify()
    import base64

    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


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
        if parameters.get("first_frame_url") and parameters.get("last_frame_url"):
            payload = {
                "model": parameters["model"],
                "input": {
                    "prompt": parameters["prompt"],
                    "first_frame_url": parameters["first_frame_url"],
                    "last_frame_url": parameters["last_frame_url"],
                },
                "parameters": {"resolution": parameters["resolution"], "prompt_extend": True},
            }
            if parameters.get("negative_prompt"):
                payload["input"]["negative_prompt"] = parameters["negative_prompt"]
            result = self.provider._call("POST", "/services/aigc/image2video/video-synthesis", payload)
            return result.get("output", {}).get("task_id")
        if parameters.get("img_url"):
            payload = {
                "model": parameters["model"],
                "input": {
                    "prompt": parameters["prompt"],
                    "img_url": parameters["img_url"],
                },
                "parameters": {
                    "resolution": parameters.get("resolution", "720P"),
                    "duration": parameters.get("duration_seconds", 5),
                    "prompt_extend": True,
                },
            }
            if parameters.get("negative_prompt"):
                payload["input"]["negative_prompt"] = parameters["negative_prompt"]
            result = self.provider._call("POST", "/services/aigc/video-generation/video-synthesis", payload)
            return result.get("output", {}).get("task_id")
        if parameters.get("ref_images_url"):
            payload = {
                "model": parameters["model"],
                "input": {
                    "function": parameters.get("function", "image_reference"),
                    "prompt": parameters["prompt"],
                    "ref_images_url": parameters["ref_images_url"],
                },
                "parameters": {
                    "size": parameters.get("size", "1280*720"),
                    "duration": parameters.get("duration_seconds", 5),
                    "prompt_extend": True,
                },
            }
            if parameters.get("negative_prompt"):
                payload["input"]["negative_prompt"] = parameters["negative_prompt"]
            result = self.provider._call("POST", "/services/aigc/video-generation/video-synthesis", payload)
            return result.get("output", {}).get("task_id")
        return self.provider.submit(
            **{
                k: v
                for k, v in parameters.items()
                if k not in {"batch_id", "resolution", "first_frame_url", "last_frame_url", "img_url", "ref_images_url", "function"}
            }
        )


def validate_wan_parameters(*, model="wanx2.1-t2v-turbo", duration_seconds=5, size="1280*720"):
    if type(duration_seconds) is not int or not 1 <= duration_seconds <= 10:
        raise ValueError("duration must be an integer from 1 to 10")
    if model in {"wanx2.1-t2v-turbo", "wan2.1-t2v-turbo"} and duration_seconds != 5:
        raise ValueError("wanx2.1-t2v-turbo 仅支持 5 秒视频，请调整时长")
    if model == "wan2.2-kf2v-flash" and duration_seconds != 5:
        raise ValueError("wan2.2-kf2v-flash only supports 5 seconds")
    if model in _WAN_5S_ONLY_MODELS and duration_seconds != 5:
        raise ValueError(f"{model} only supports 5 seconds")
    if model in {"wan2.1-i2v-turbo", "wanx2.1-i2v-turbo"} and duration_seconds not in _WAN_TURBO_I2V_DURATIONS:
        raise ValueError(f"{model} supports durations 3, 4, or 5 seconds")
    if any(not isinstance(value, str) or not value.strip() for value in (model, size)):
        raise ValueError("video model and size are required")


def run_wan_video(provider, registry, *, prompt, model="wanx2.1-t2v-turbo",
                  duration_seconds=5, size="1280*720", negative_prompt=None, batch_id=None, downloader=download_video,
                  first_frame=None, last_frame=None, ref_images=None, resolution="720P"):
    if batch_id is not None and (not isinstance(batch_id, str) or not batch_id.strip() or len(batch_id) > 128):
        raise ValueError("invalid video batch_id")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("video prompt is required")
    validate_wan_parameters(model=model, duration_seconds=duration_seconds, size=size)
    if negative_prompt is not None and not isinstance(negative_prompt, str):
        raise ValueError("negative prompt must be text")
    parameters = {"prompt": prompt, "model": model, "duration_seconds": duration_seconds,
                  "size": size, "negative_prompt": negative_prompt}
    conditioning_mode = "text_to_video"
    if is_wan_r2v_model(model) or ref_images is not None:
        if not is_wan_r2v_model(model) or not ref_images:
            raise ValueError("reference-to-video mode requires ref_images and wan2.1-vace-plus")
        if first_frame is not None or last_frame is not None:
            raise ValueError("reference-to-video mode cannot be combined with first/last frame")
        refs = list(ref_images)[:3]
        parameters["function"] = "image_reference"
        parameters["ref_images_url"] = [_encode_image_data_uri(item) for item in refs]
        conditioning_mode = "reference_to_video"
    elif is_wan_i2v_model(model):
        if first_frame is None or last_frame is not None:
            raise ValueError(f"single-frame I2V model {model} requires first_frame without last_frame")
        if resolution not in {"480P", "720P", "1080P"}:
            raise ValueError("unsupported keyframe resolution")
        parameters["img_url"] = _encode_image_data_uri(first_frame)
        parameters["resolution"] = resolution
        conditioning_mode = "image_to_video"
    elif is_wan_kf2v_model(model) or first_frame is not None or last_frame is not None:
        if not is_wan_kf2v_model(model) or first_frame is None or last_frame is None:
            raise ValueError("first/last frame mode requires both frames and wan2.2-kf2v-flash")
        if resolution not in {"480P", "720P", "1080P"}:
            raise ValueError("unsupported keyframe resolution")
        parameters["first_frame_url"] = _encode_image_data_uri(first_frame)
        parameters["last_frame_url"] = _encode_image_data_uri(last_frame)
        parameters["resolution"] = resolution
        conditioning_mode = "first_last_frame"
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
            "conditioning_mode": conditioning_mode, "review_status": "pending", "execution": execution}
