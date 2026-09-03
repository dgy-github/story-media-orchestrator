"""Small content-addressed registry shared by orchestration stages."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ArtifactRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put_json(self, value: dict[str, Any]) -> str:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(payload).hexdigest()
        target = self.root / digest
        if not target.exists():
            target.write_bytes(payload)
        elif target.read_bytes() != payload:
            raise RuntimeError("artifact digest collision")
        return f"artifact://sha256/{digest}"

    def put_bytes(self, payload: bytes) -> str:
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("artifact payload must be non-empty bytes")
        digest = hashlib.sha256(payload).hexdigest()
        target = self.root / digest
        if target.exists() and target.read_bytes() != payload:
            raise RuntimeError("artifact digest collision")
        if not target.exists():
            target.write_bytes(payload)
        return f"artifact://sha256/{digest}"

    def get_json(self, ref: str) -> dict[str, Any]:
        prefix = "artifact://sha256/"
        if not isinstance(ref, str) or not ref.startswith(prefix):
            raise ValueError("invalid artifact reference")
        digest = ref[len(prefix):]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid artifact digest")
        payload = (self.root / digest).read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise RuntimeError("artifact integrity check failed")
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("artifact must contain an object")
        return value
