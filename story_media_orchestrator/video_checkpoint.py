"""Durable submission receipt around the existing ComfyUI transport."""
from contextlib import contextmanager
from copy import copy
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4
from urllib.error import HTTPError


@contextmanager
def exclusive_receipt(path, task_name="video"):
    # OS lock is released on process exit, unlike an abandoned lock-file flag.
    lock = path.with_suffix(".lock")
    with lock.open("a+b") as handle:
        # Windows byte-range locks can cover an empty file. Reading a byte
        # already locked by another worker raises before our lock handling.
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc: raise RuntimeError(f"{task_name} task already running") from exc
        else:
            import fcntl
            try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc: raise RuntimeError(f"{task_name} task already running") from exc
        try: yield
        finally:
            if os.name == "nt":
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(handle, fcntl.LOCK_UN)


def execute_with_receipt(client, graph, store, execute):
    identity = json.dumps({"endpoint":client.base_url, "graph":graph}, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(identity.encode()).hexdigest()
    directory = Path(store.root) / "video-tasks"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (key + ".json")
    def save(value):
        temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
        try:
            temporary.write_text(json.dumps(value), encoding="utf-8")
            temporary.replace(path)
        finally: temporary.unlink(missing_ok=True)
    with exclusive_receipt(path):
        receipt = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if receipt.get("state") == "done":
            store.get_bytes(receipt["result"]["artifact_ref"])
            return receipt["result"]
        if receipt.get("state") == "submitting":
            raise RuntimeError("Video submission outcome unknown; inspect provider task queue before retrying")
        wrapped = copy(client)
        submit = client.submit
        def checkpointed_submit(workflow, **kwargs):
            if receipt.get("prompt_id"):
                return receipt["prompt_id"]
            save({"state":"submitting", "request_key":key})
            try:
                prompt_id = submit(workflow, **kwargs)
            except Exception as exc:
                cause = exc
                for _ in range(8):
                    if isinstance(cause, HTTPError):
                        # An explicit client rejection has no accepted task.
                        # Timeouts and server failures remain ambiguous.
                        if 400 <= cause.code < 500 and cause.code != 408:
                            save({'state': 'rejected', 'request_key': key,
                                  'http_status': cause.code})
                        break
                    cause = cause.__cause__
                    if cause is None:
                        break
                raise
            if not isinstance(prompt_id, str) or not prompt_id:
                raise RuntimeError("ComfyUI acceptance missing prompt_id")
            receipt.update(state="accepted", prompt_id=prompt_id, request_key=key)
            save(receipt)
            return prompt_id
        wrapped.submit = checkpointed_submit
        result = execute(wrapped)
        if not isinstance(result, dict) or result.get("state") != "succeeded":
            raise RuntimeError("video execution has not completed successfully")
        if result.get("prompt_id") != receipt.get("prompt_id") or not receipt.get("prompt_id"):
            raise RuntimeError("video result does not match submitted task")
        store.get_bytes(result.get("artifact_ref"))
        save({**receipt, "state":"done", "result":result})
        return result
