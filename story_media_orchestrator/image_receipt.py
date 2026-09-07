"""Resume individual DashScope candidates without resubmitting accepted tasks."""
import base64
from copy import copy
from hashlib import sha256
import json
from uuid import uuid4
from urllib.error import HTTPError

from .video_checkpoint import exclusive_receipt


def generate_candidate(provider, candidate, store, batch_id, index):
    identity = {"batch": batch_id, "index": index}
    key = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = store.root / "image-tasks"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (key + ".json")
    wrapped = copy(provider)
    transport = provider._read_json

    def save(receipt):
        temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
        try:
            temporary.write_text(json.dumps(receipt), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    with exclusive_receipt(path, "image"):
        receipt = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

        def checkpoint(request):
            if request.get_method() != "POST":
                return transport(request)
            fingerprint = sha256(request.full_url.encode() + b"\0" + (request.data or b"")).hexdigest()
            if receipt.get("request_key") and receipt["request_key"] != fingerprint:
                raise RuntimeError("恢复生图的参数已改变，请使用新的批次编号")
            if receipt.get("task_id"):
                return {"output": {"task_id": receipt["task_id"]}}
            if receipt.get("state") == "submitting":
                raise RuntimeError("生图提交结果未知，请核查云端任务后再决定是否新建批次")
            receipt.update(state="submitting", request_key=fingerprint)
            save(receipt)
            try:
                result = transport(request)
            except HTTPError as exc:
                if 400 <= exc.code < 500 and exc.code != 408:
                    receipt.update(state="rejected")
                    save(receipt)
                raise
            task_id = result.get("output", {}).get("task_id")
            if isinstance(task_id, str) and task_id.strip():
                receipt.update(state="accepted", task_id=task_id)
                save(receipt)
            return result

        # Validate input separately because a cached result skips provider.generate.
        input_key = sha256(json.dumps({"prompt": candidate.get("prompt"),
            "negative_prompt": candidate.get("negative_prompt"),
            "source_spans": candidate.get("source_spans"),
            "config": {name: getattr(provider, name, None) for name in
                       ("base_url", "model", "size", "watermark", "prompt_extend")}},
            sort_keys=True).encode()).hexdigest()
        if receipt and receipt.get("input_key") != input_key:
            raise RuntimeError("恢复生图的参数已改变，请使用新的批次编号")
        if receipt.get("state") == "done":
            content = store.get_bytes(receipt["artifact_ref"])
            return {**receipt["metadata"], "content_base64": base64.b64encode(content).decode()}
        receipt["input_key"] = input_key
        wrapped._read_json = checkpoint
        result = wrapped.generate(candidate)
        content = base64.b64decode(result["content_base64"], validate=True)
        artifact = store.put_bytes(content)
        save({**receipt, "state": "done", "artifact_ref": artifact,
              "metadata": {k: v for k, v in result.items() if k != "content_base64"}})
        return result
