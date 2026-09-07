"""Persist the video result and the exact non-secret inputs that produced it."""
from copy import deepcopy


def persist_video_result(result, request, registry):
    recorded = deepcopy(result)
    recorded["source_spans"] = list(request.get("source_spans", []))
    # These are associated story assets; neither current T2V graph consumes them.
    recorded["associated_image_refs"] = list(dict.fromkeys(
        request[key] for key in ("first_frame_ref", "last_frame_ref") if request.get(key)))
    provider = request.get("video_provider", "comfyui")
    fields = ("video_model", "video_size", "duration_seconds", "negative_prompt", "batch_id") if provider == "dashscope" else ("turbo", "seed", "width", "height", "length", "fps")
    defaults = ({"video_model": "wanx2.1-t2v-turbo", "video_size": "1280*720", "duration_seconds": 5}
                if provider == "dashscope" else {"turbo": False, "seed": 0, "width": 864, "height": 480, "length": 124, "fps": 24.0})
    scene = request.get("scene") or {}
    prompt = request.get("prompt") or scene.get("action_prompt") or scene.get("description") or scene.get("summary")
    recorded["generation_parameters"] = {"video_provider": provider, "prompt": prompt,
                                         **defaults, **{key: request[key] for key in fields if key in request}}
    # The content-addressed record remains inspectable after the UI is closed.
    record_ref = registry.put_json(recorded)
    return {**recorded, "result_record_ref": record_ref}
