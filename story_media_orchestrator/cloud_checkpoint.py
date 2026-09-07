"""Persist the existing DashScope adapter's submit receipt before it polls.

The sibling provider owns submit/poll/download and transport retries. This small
wrapper checkpoints its JSON transport seam; resume replays only the submit
receipt and lets the same provider continue polling/downloading that task.
"""
from copy import copy
from hashlib import sha256
from urllib.error import HTTPError

from .providers import StoryImageProvider


class CheckpointedStoryImageProvider(StoryImageProvider):
    def __init__(self, adapter, registry, manifest, root):
        super().__init__(adapter, registry)
        self.manifest, self.root = manifest, root

    def generate(self, shot, output):
        original = self.adapter.provider
        provider = copy(original)
        read_json = provider._read_json

        def checkpointed(request):
            if request.get_method() != "POST":
                return read_json(request)
            request_key = sha256(request.full_url.encode() + b"\0" + (request.data or b"")).hexdigest()
            previous = shot.image_task
            if previous.get("request_key") == request_key:
                if previous.get("task_id"):
                    return {"output": {"task_id": previous["task_id"]}}
                if previous.get("state") == "submitting":
                    raise RuntimeError("阿里云提交结果未知，请先核查云端任务；确认后使用重试镜头重新提交")
            shot.image_task = {"request_key": request_key, "state": "submitting"}
            self.manifest.save(self.root / "project.json")
            try:
                result = read_json(request)
            except HTTPError as exc:
                # Explicit client rejection means no accepted task to resume.
                if 400 <= exc.code < 500 and exc.code != 408:
                    shot.image_task["state"] = "rejected"
                    self.manifest.save(self.root / "project.json")
                raise
            task_id = result.get("output", {}).get("task_id")
            if isinstance(task_id, str) and task_id.strip():
                shot.image_task = {"request_key": request_key, "task_id": task_id, "state": "accepted"}
                self.manifest.save(self.root / "project.json")
            return result

        provider._read_json = checkpointed
        self.adapter.provider = provider
        try:
            return super().generate(shot, output)
        finally:
            self.adapter.provider = original
