"""Adapters for the independently versioned story, image and video agents."""
from __future__ import annotations

from copy import copy
import base64
import inspect
import json
import math
import time
from uuid import uuid4
from urllib import request as http_request, error as http_error
from typing import Any, Callable

from .registry import ArtifactRegistry


class StoryCampaignAdapter:
    """Wrap the host's story campaign runner without importing its internals."""
    def __init__(self, runner: Callable[..., Any]) -> None:
        self.runner = runner

    def run(self, story_input: dict[str, Any], **kwargs) -> dict[str, Any]:
        result = self.runner(story_input, **kwargs)
        if inspect.isawaitable(result):
            raise RuntimeError("async story runner requires an async host adapter")
        if not isinstance(result, dict) or result.get("schema") != "story-package/v1":
            result = result.get("package") if isinstance(result, dict) else None
        if not isinstance(result, dict) or result.get("schema") != "story-package/v1":
            raise ValueError("story campaign did not return story-package/v1")
        return result


class HttpStoryCampaignAdapter(StoryCampaignAdapter):
    """Call the main project's authenticated sidecar over its HTTP contract."""
    def __init__(self, base_url: str, token: str, *, timeout: float = 10.0,
                 poll_interval: float = 2.0, max_polls: int = 300) -> None:
        if not base_url.startswith(("http://", "https://")) or len(token) < 32:
            raise ValueError("invalid story sidecar configuration")
        self.base_url, self.token = base_url.rstrip("/"), token
        self.timeout, self.poll_interval, self.max_polls = timeout, poll_interval, max_polls

    def _check_terminal_events(self, run_id: str, cursor: int) -> int:
        req = http_request.Request(self.base_url + f"/v1/runs/{run_id}/events",
            headers={"Authorization": f"Bearer {self.token}", "Last-Event-ID": str(cursor)})
        with http_request.urlopen(req, timeout=self.timeout) as response:
            size = 0
            while True:
                line = response.readline(65537)
                size += len(line)
                if not line or size > 1024 * 1024 or len(line) > 65536:
                    raise RuntimeError("story event replay incomplete or oversized")
                if line.startswith(b": replay-complete"):
                    return cursor
                if not line.startswith(b"data:"):
                    continue
                event = json.loads(line[5:].strip())
                seq = event.get("seq")
                if event.get("run_id") != run_id or not isinstance(seq, int) or seq <= cursor:
                    raise ValueError("invalid story event cursor or run")
                cursor = seq
                if event.get("event_type") in {"run.failed", "run.cancelled"}:
                    payload = event.get("payload")
                    detail = payload.get("error") if isinstance(payload, dict) else None
                    detail = detail.strip()[:2000] if isinstance(detail, str) and detail.strip() else "no failure detail returned"
                    raise RuntimeError(f"story task {run_id}: {event['event_type']}; {detail}")

    def run(self, story_input: dict[str, Any], **kwargs) -> dict[str, Any]:
        budget_seconds = story_input.get("job", {}).get("budget", {}).get("deadline_seconds")
        if budget_seconds is not None and (type(budget_seconds) is not int or budget_seconds <= 0):
            raise ValueError("story deadline_seconds must be a positive integer")
        poll_limit = self.max_polls
        if budget_seconds is not None:
            poll_limit = max(poll_limit, math.ceil(budget_seconds / max(self.poll_interval, 0.1)) + 1)
        body = json.dumps(story_input, ensure_ascii=False).encode()
        req = http_request.Request(self.base_url + "/v1/runs", body,
                                   {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json",
                                    "Idempotency-Key": kwargs.get("idempotency_key") or story_input.get("job", {}).get("job_id") or f"story-media-{uuid4().hex}"})
        with http_request.urlopen(req, timeout=self.timeout) as response:
            acceptance = json.loads(response.read())
        run_id = acceptance.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError("story sidecar acceptance missing run_id")
        cursor = 0
        deadline = time.monotonic() + budget_seconds if budget_seconds is not None else None
        for _ in range(poll_limit):
            if deadline is not None and time.monotonic() >= deadline:
                break
            req = http_request.Request(self.base_url + f"/v1/runs/{run_id}/result",
                                       headers={"Authorization": f"Bearer {self.token}"})
            try:
                with http_request.urlopen(req, timeout=self.timeout) as response:
                    if response.status == 200:
                        result = json.loads(response.read())
                        package = (result if result.get("schema") == "story-package/v1" else result.get("package")) if isinstance(result, dict) else None
                        if isinstance(package, dict) and package.get("schema") == "story-package/v1":
                            return package
                        raise ValueError("story sidecar returned an invalid story package")
            except http_error.HTTPError as exc:
                if exc.code not in {404, 409, 425}:
                    raise
                cursor = self._check_terminal_events(run_id, cursor)
            time.sleep(min(self.poll_interval, max(0, deadline - time.monotonic())) if deadline is not None else self.poll_interval)
        raise TimeoutError(f"story polling timed out; run_id={run_id}; task may still be running; retry with the same job_id")


class StoryImageAdapter:
    """Bridge story-image-agent workflow/provider and persist image artifacts."""
    def __init__(self, workflow: Any, provider: Any, registry: ArtifactRegistry) -> None:
        self.workflow, self.provider, self.registry = workflow, provider, registry

    def run(self, scene: dict[str, Any], source_spans: list[str], *, quality: dict[str, Any] | None = None,
            candidate_count: int = 1, include_preview: bool = False, batch_id: str | None = None,
            prompt_revision: str | None = None) -> dict[str, Any]:
        if prompt_revision is not None and (not isinstance(prompt_revision, str) or not prompt_revision.strip()):
            raise ValueError('prompt revision must be nonempty text')
        if batch_id is not None and (not isinstance(batch_id, str) or not batch_id.strip() or len(batch_id) > 128):
            raise ValueError("invalid image batch_id")
        if type(candidate_count) is not int or not 1 <= candidate_count <= 8:
            raise ValueError("candidate_count must be an integer between 1 and 8")
        if quality is not None and candidate_count != 1:
            raise ValueError("batch candidates require individual quality evaluations")
        plan = self.workflow.build_production_plan(scene, source_spans, candidate_count=candidate_count)
        if prompt_revision is not None:
            revision = self.workflow.revise_prompt(plan['prompt_revision_id'], prompt_revision)
            plan = {**plan, 'prompt_revision_id': revision.revision_id,
                    'candidates': [self.workflow.build_generation_request(revision.revision_id)
                                   for _ in range(candidate_count)]}
        revision_record = None
        prompt_history = []
        if prompt_revision is not None:
            from dataclasses import asdict
            import json
            record = {'schema': 'image-prompt-history/v1',
                      'revisions': [asdict(item) for item in self.workflow.revisions]}
            prompt_history = record['revisions']
            revision_record = self.registry.put_bytes(json.dumps(record, ensure_ascii=False).encode('utf-8'))
        outputs = []
        for index, candidate in enumerate(plan["candidates"]):
            try:
                if batch_id is not None:
                    from .image_receipt import generate_candidate
                    generated = generate_candidate(self.provider, candidate, self.registry, batch_id, index)
                else:
                    generated = self.provider.generate(candidate)
                content = base64.b64decode(generated.get("content_base64", ""), validate=True)
                frame_ref = self.registry.put_bytes(content)
                output = {"request_id": candidate["request_id"], "artifact_ref": frame_ref,
                          "status": "succeeded"}
                mime = generated.get("mime_type")
                if include_preview and mime in {"image/png", "image/jpeg", "image/webp"}:
                    output["preview_url"] = f"data:{mime};base64,{base64.b64encode(content).decode()}"
                outputs.append(output)
            except Exception as exc:
                if candidate_count == 1:
                    raise
                from .errors import image_error_detail
                outputs.append({"request_id": candidate["request_id"], "status": "failed",
                                "error": type(exc).__name__, "error_detail": image_error_detail(exc)})
        successful = [item for item in outputs if item["status"] == "succeeded"]
        frame_ref = successful[0]["artifact_ref"] if successful else None
        candidate = plan["candidates"][0]
        if quality is not None:
            final = self.workflow.finalize_candidate(plan, candidate["request_id"], quality)
            review_status = "evaluated"
        else:
            # A generated file is not evidence of visual quality. Keep its candidate
            # and artifact for review without fabricating scores or final approval.
            final = {**plan, "evaluation": {"status": "unavailable", "passed": None,
                     "reason": "尚未提供图片质量评估", "metrics": {}}}
            review_status = "pending"
        return {"schema": "image-production-plan/v1", "plan": final,
                "source_spans": list(source_spans),
                "candidate_results": outputs,
                "prompt_history_ref": revision_record,
                "prompt_history": prompt_history,
                "selected_request_id": successful[0]["request_id"] if successful else None,
                "status": "succeeded" if len(successful) == len(outputs) else "partial" if successful else "failed",
                "review_status": review_status,
                "first_frame_ref": frame_ref, "last_frame_ref": frame_ref,
                "frame_mode": "same_image"}



class StoryVideoAdapter:
    """Build a controlled video pipeline and optionally execute it via ComfyUI."""
    def __init__(self, workflow: Any, *, comfy: Any = None, registry: ArtifactRegistry | None = None, models: Any = None) -> None:
        self.workflow, self.comfy, self.registry, self.models = workflow, comfy, registry, models

    def _execute_comfy(self, graph: dict, *, poll_interval: float = 2.0, max_polls: int = 600) -> dict:
        # Reuse the provider's submission, result parsing and download; only pace
        # its otherwise tight polling loop and retain the accepted task identity.
        client = copy(self.comfy)
        history = client.history
        task_id = None
        polled = False
        def paced_history(prompt_id):
            nonlocal task_id, polled
            task_id = prompt_id
            if polled:
                time.sleep(poll_interval)
            polled = True
            return history(prompt_id)
        client.history = paced_history
        try:
            from .video_checkpoint import execute_with_receipt
            return execute_with_receipt(client, graph, self.registry,
                lambda transport: transport.run_to_artifact(graph, self.registry, max_polls=max_polls))
        except TimeoutError as exc:
            raise TimeoutError(f"ComfyUI task still pending; prompt_id={task_id}; do not resubmit automatically") from exc

    def run(self, *, first_frame_ref: str, last_frame_ref: str | None = None,
            story_spans: list[str], shot: dict[str, Any] | None = None,
            scene: dict[str, Any] | None = None,
            action_unit: dict[str, Any] | None = None, prompt: str | None = None, seed: int = 0,
            width: int = 864, height: int = 480, length: int = 124, fps: float = 24.0,
            quality: dict[str, Any] | None = None) -> dict[str, Any]:
        if (type(length) is not int or length < 1 or isinstance(fps, bool)
                or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0
                or length / fps < 1):
            raise ValueError("video duration must be finite and at least one second")
        build = self.workflow.build_pipeline_v2 if action_unit is not None else self.workflow.build_pipeline
        pipeline = build(
            first_frame_ref, story_spans, shot or scene or {}, coarse_duration_seconds=length / fps,
            action_unit=action_unit,
            quality=quality,
        )
        result: dict[str, Any] = {"schema": pipeline["schema"], "pipeline": pipeline, "status": "planned",
                                  "review_status": "pending" if quality is None else "evaluated",
                                  "conditioning_mode": "text_to_video" if self.comfy is not None else "planned_image_reference"}
        if self.comfy is not None:
            if action_unit is not None:
                raise ValueError("当前 ComfyUI 执行图未接入动作控制；请使用支持控制计划的执行器，不能用普通文生视频替代")
            scene = scene or shot or {}
            prompt = prompt or scene.get("action_prompt") or scene.get("description") or scene.get("summary")
            if not prompt:
                raise ValueError("prompt is required for ComfyUI execution")
            from story_video_agent import build_minimax_h3_workflow
            turbo = getattr(self.models, "video_turbo", False) if self.models is not None else False
            result["execution"] = self._execute_comfy(build_minimax_h3_workflow(prompt=prompt, turbo=turbo, seed=seed, width=width, height=height, length=length, fps=fps))
            result["status"] = result["execution"]["state"]
        return result
