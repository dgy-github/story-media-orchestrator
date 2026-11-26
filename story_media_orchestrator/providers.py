"""Provider contracts used by the resumable project workflow."""
from __future__ import annotations

from pathlib import Path
import os
from urllib.error import HTTPError
from typing import Protocol

from .manifest import Shot
from .registry import ArtifactRegistry


class ImageProvider(Protocol):
    def generate(self, shot: Shot, output: Path) -> Path: ...


class VideoProvider(Protocol):
    def generate(self, shot: Shot, image: Path | None, output: Path) -> Path: ...


class TextFrameProvider:
    """Credential-free fallback that keeps a project inspectable."""

    def generate(self, shot: Shot, output: Path) -> Path:
        target = output.with_suffix(".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(shot.text, encoding="utf-8")
        return target


class StoryImageProvider:
    """Materialize images produced by the existing story-image adapter."""

    def __init__(self, adapter, registry: ArtifactRegistry, manifest=None) -> None:
        self.adapter, self.registry, self.manifest = adapter, registry, manifest

    @staticmethod
    def _visual_characters(characters):
        descriptions = []
        for character in characters or []:
            if not isinstance(character, dict):
                continue
            fields = [character.get(key) for key in ('name', 'appearance', 'clothing')]
            description = '，'.join(value.strip() for value in fields if isinstance(value, str) and value.strip())
            if description:
                descriptions.append(description)
        return '；'.join(descriptions) or '以动作描述中的人物为准'

    @staticmethod
    def _inject_transport_conditioning(provider, conditioning: dict):
        from contextlib import contextmanager
        import json

        seed = conditioning.get("seed")
        ref_img = conditioning.get("ref_img")
        original_read = getattr(provider, "_read_json", None) if provider is not None else None
        if (seed is None and not ref_img) or not callable(original_read):
            @contextmanager
            def _noop():
                yield
            return _noop()

        @contextmanager
        def _wrapped():
            def conditioned_read(request, *args, **kwargs):
                data = getattr(request, "data", None)
                if isinstance(data, (bytes, bytearray)):
                    try:
                        payload = json.loads(data.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        payload = None
                    if isinstance(payload, dict):
                        if seed is not None:
                            params = payload.get("parameters")
                            if not isinstance(params, dict):
                                params = {}
                                payload["parameters"] = params
                            params["seed"] = int(seed)
                        if ref_img:
                            inp = payload.get("input")
                            if not isinstance(inp, dict):
                                inp = {}
                                payload["input"] = inp
                            inp["ref_img"] = ref_img
                        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                        request.data = encoded
                        headers = getattr(request, "headers", None)
                        if isinstance(headers, dict):
                            for hdr in list(headers.keys()):
                                if hdr.lower() == "content-length":
                                    headers[hdr] = str(len(encoded))
                return original_read(request, *args, **kwargs)

            provider._read_json = conditioned_read
            try:
                yield
            finally:
                provider._read_json = original_read

        return _wrapped()

    def generate(self, shot: Shot, output: Path) -> Path:
        saved = self._generate_raw(shot, output)
        if self.manifest is not None:
            from .character_bible import record_auto_anchor_frame
            record_auto_anchor_frame(self.manifest, Path(output).parent.parent, shot, saved)
        return saved

    def _generate_raw(self, shot: Shot, output: Path) -> Path:
        visual_action = shot.scene_context.get("visual_action") or shot.text
        scene_payload = {
            "summary": shot.text,
            "description": visual_action,
            "action": visual_action + "。仅绘制当前动作的一张完整画面、一个时间点。不要分屏、拼贴、上下多格、漫画分镜、字幕或文字。",
            "location": shot.scene_context.get("location", "以动作描述中的地点为准"),
            "characters": self._visual_characters(shot.scene_context.get("characters")),
            "lighting": shot.scene_context.get("lighting") or shot.scene_context.get("time") or "以动作描述中的时间与光线为准",
            "shot_id": shot.id,
            **{key: shot.scene_context[key] for key in ("framing", "mood", "negative", "continuity") if key in shot.scene_context},
        }
        conditioning = {"seed": None, "ref_img": None}
        if self.manifest is not None:
            from .character_bible import build_consistent_image_scene
            project_root = Path(output).parent.parent
            scene_payload, conditioning = build_consistent_image_scene(
                self.manifest,
                shot,
                scene_payload,
                project_root=project_root if (project_root / "flywheel").is_dir() else None,
            )
        def _extract_http_error(exc: Exception) -> HTTPError | None:
            cause: Exception | None = exc
            for _ in range(8):
                if isinstance(cause, HTTPError):
                    return cause
                cause = cause.__cause__ if cause is not None else None
            return None

        def _invoke_adapter(cond: dict):
            with self._inject_transport_conditioning(getattr(self.adapter, "provider", None), cond):
                return self.adapter.run(
                    scene_payload,
                    shot.scene_context.get("source_spans") or [shot.scene_id, shot.id],
                )

        try:
            result = _invoke_adapter(conditioning)
        except Exception as exc:
            http_err = _extract_http_error(exc)
            if http_err is not None and http_err.code == 400 and conditioning.get("ref_img"):
                # Some cloud T2I models reject input.ref_img with HTTP 400; retry once with seed + DNA only.
                shot.image_task.clear()
                fallback_cond = {"seed": conditioning.get("seed"), "ref_img": None}
                try:
                    result = _invoke_adapter(fallback_cond)
                except Exception as retry_exc:
                    http_err = _extract_http_error(retry_exc)
                    if http_err is None:
                        raise
                else:
                    http_err = None
            else:
                result = None
            if result is None:
                if http_err is not None:
                    message = {
                        401: "阿里云 API Key 无效（HTTP 401），请更新 DashScope 凭据后恢复",
                        403: "阿里云拒绝访问（HTTP 403），请检查模型权限及账户状态",
                        429: "阿里云请求限流（HTTP 429），请稍后恢复",
                    }.get(http_err.code, f"阿里云请求失败（HTTP {http_err.code}）")
                    raise RuntimeError(message) from None
                raise
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(self.registry.get_bytes(result["first_frame_ref"]))
        return output


def build_project_image_provider(manifest, root: Path):
    """Reuse the sibling DashScope adapter or zero-cost Codex image_gen without requiring story/video services.

    Model and size are pinned in the manifest on first use. Credentials remain
    in the existing provider's environment/config loader, never in the project.
    """
    from .editing import default_image_size_for_aspect

    if getattr(manifest, "image_source", "local") == "codex":
        from .codex_image import CodexImageProvider

        manifest.image_model = (
            manifest.image_model
            or os.environ.get("CODEX_IMAGE_MODEL")
            or "codex-image-gen"
        )
        manifest.image_size = (
            manifest.image_size
            or os.environ.get("CODEX_IMAGE_SIZE")
            or default_image_size_for_aspect(getattr(manifest, "aspect_ratio", "16:9"))
        )
        return CodexImageProvider(manifest=manifest, root=root)

    from .runtime import _import_from_root, resolve_sibling_root
    from .adapters import StoryImageAdapter
    image_root = Path(os.environ.get("STORY_IMAGE_AGENT_ROOT")
                      or resolve_sibling_root("story-image-agent", "microcodex-story-image-agent", module_name="story_image_agent"))
    pkg = _import_from_root(image_root, "story_image_agent")
    provider = pkg.DashScopeImageProvider.from_nanocodex_config()
    provider.model = manifest.image_model or os.environ.get("STORY_IMAGE_MODEL") or provider.model
    preferred_size = (
        manifest.image_size
        or os.environ.get("STORY_IMAGE_SIZE")
        or default_image_size_for_aspect(getattr(manifest, "aspect_ratio", "16:9"))
        or provider.size
    )
    provider.size = provider._normalize_size(preferred_size)
    manifest.image_model, manifest.image_size = provider.model, provider.size
    registry = ArtifactRegistry(root / "artifacts")
    adapter = StoryImageAdapter(pkg.ImagePromptWorkflow(manifest.project_id), provider, registry)
    from .cloud_checkpoint import CheckpointedStoryImageProvider
    return CheckpointedStoryImageProvider(adapter, registry, manifest, root)
