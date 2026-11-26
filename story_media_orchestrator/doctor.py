"""Preflight environment, git closed-source lock, and remote GPU/ComfyUI readiness diagnostics."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
from urllib import error as urllib_error, parse as urllib_parse, request as urllib_request

from .local_media import find_ffmpeg
from .manifest import ProjectManifest
from .runtime import resolve_sibling_root

REQUIRED_H3_NODE_CLASSES = (
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "MiniMaxH3ImageToVideo",
    "RandomNoise",
    "KSamplerSelect",
    "BasicScheduler",
    "BasicGuider",
    "SamplerCustomAdvanced",
    "VAEDecode",
    "VAEDecodeAudio",
    "CreateVideo",
    "SaveVideo",
    "LoraLoaderModelOnly",
    "LoadImage",
    "ImageScale",
)

REQUIRED_H3_WEIGHTS = (
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "minimax_h3_video_vae_fp16.safetensors",
    "minimax_h3_audio_vae_fp32.safetensors",
    "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
)

OPTIONAL_LATENTSYNC_NODES = (
    "VHS_LoadVideo",
    "LoadAudio",
    "LatentSyncNode",
    "VHS_VideoCombine",
)

_SIBLING_SPECS = (
    (
        "microcodex-short-drama-studio",
        "STORY_CAMPAIGN_ROOT",
        "microcodex-story-campaign-adapter",
        "sidecar",
        "",
    ),
    (
        "story-image-agent",
        "STORY_IMAGE_AGENT_ROOT",
        "microcodex-story-image-agent",
        "",
        "story_image_agent",
    ),
    (
        "story-video-agent",
        "STORY_VIDEO_AGENT_ROOT",
        "microcodex-story-video-agent",
        "",
        "story_video_agent",
    ),
)

_REMOTE_LINE_RE = re.compile(r"^(\S+)\s+(\S+)\s+\((fetch|push)\)$")
_GITHUB_HTTPS_RE = re.compile(r"^https?://(?:[^@/]+@)?github\.com/([^/\s]+)/([^/\s]+?)(?:\.git)?/?$", re.IGNORECASE)
_GITHUB_SSH_RE = re.compile(r"^(?:ssh://)?git@github\.com[:/]([^/\s]+)/([^/\s]+?)(?:\.git)?/?$", re.IGNORECASE)

PUBLIC_BLOCKLIST_SLUGS = frozenset(
    {
        "dgy-github/story-media-orchestrator",
        "dgy-github/story-video-agent",
        "dgy-github/story-image-agent",
        "dgy-github/microcodex-short-drama-studio",
    }
)


def _extract_github_slug(url: str) -> str | None:
    cleaned = (url or "").strip()
    if not cleaned:
        return None
    for pat in (_GITHUB_HTTPS_RE, _GITHUB_SSH_RE):
        m = pat.match(cleaned)
        if m:
            owner, repo = m.groups()
            return f"{owner.lower()}/{repo.lower()}"
    return None


class _NoRedirect(urllib_request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _check_local_toolchain() -> dict[str, Any]:
    ffmpeg_path: str | None = None
    ffmpeg_error: str | None = None
    try:
        ffmpeg_path = str(find_ffmpeg())
    except Exception as exc:
        ffmpeg_error = str(exc)

    cv2_version: str | None = None
    try:
        import cv2  # type: ignore

        cv2_version = str(getattr(cv2, "__version__", "installed"))
    except Exception:
        cv2_version = None

    pil_version: str | None = None
    try:
        import PIL  # type: ignore

        pil_version = str(getattr(PIL, "__version__", "installed"))
    except Exception:
        pil_version = None

    return {
        "python_executable": sys.executable,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "ffmpeg_available": ffmpeg_path is not None,
        "ffmpeg_path": ffmpeg_path,
        "ffmpeg_error": ffmpeg_error,
        "cv2_version": cv2_version,
        "pillow_version": pil_version,
    }


def _resolve_repo_path(
    repo_name: str,
    env_var: str,
    dist_name: str,
    strip_suffix: str,
    module_name: str,
    *,
    repo_roots: dict[str, str | Path] | None = None,
) -> Path | None:
    if repo_roots and repo_name in repo_roots and repo_roots[repo_name]:
        cand = Path(repo_roots[repo_name]).expanduser().resolve()
        if cand.is_dir():
            return cand
        return None

    env_val = os.environ.get(env_var, "").strip()
    if env_val and Path(env_val).is_dir():
        return Path(env_val).resolve()

    resolved = resolve_sibling_root(repo_name, dist_name, strip_suffix, module_name)
    if resolved.is_dir():
        return resolved.resolve()

    fallback = Path("D:/github_dgy") / repo_name
    if fallback.is_dir():
        return fallback.resolve()
    return None


def _check_sibling_repositories(
    *,
    repo_roots: dict[str, str | Path] | None = None,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for repo_name, env_var, dist_name, strip_suffix, module_name in _SIBLING_SPECS:
        path = _resolve_repo_path(
            repo_name,
            env_var,
            dist_name,
            strip_suffix,
            module_name,
            repo_roots=repo_roots,
        )
        importable = bool(importlib.util.find_spec(module_name)) if module_name else None
        results[repo_name] = {
            "exists": path is not None and path.is_dir(),
            "path": str(path) if path is not None else None,
            "env_var": env_var,
            "importable": importable,
        }
    return results


def _inspect_git_push_lock(repo_dir: Path) -> dict[str, Any]:
    if not repo_dir.is_dir() or not (repo_dir / ".git").exists():
        return {
            "path": str(repo_dir),
            "is_git_repo": False,
            "push_locked": False,
            "reason": "not a git repository",
            "remotes": {},
            "tracking_branches": [],
        }

    remotes_proc = subprocess.run(
        ["git", "-C", str(repo_dir), "remote", "-v"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=5,
    )
    if remotes_proc.returncode != 0:
        return {
            "path": str(repo_dir),
            "is_git_repo": True,
            "push_locked": False,
            "reason": f"git remote -v failed: {remotes_proc.stderr.strip()}",
            "remotes": {},
            "tracking_branches": [],
        }

    remotes: dict[str, dict[str, str]] = {}
    for line in remotes_proc.stdout.splitlines():
        m = _REMOTE_LINE_RE.match(line.strip())
        if not m:
            continue
        name, url, direction = m.groups()
        remotes.setdefault(name, {})[direction] = url

    branches_proc = subprocess.run(
        ["git", "-C", str(repo_dir), "for-each-ref", "--format=%(refname:short)|%(upstream:short)", "refs/heads"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=5,
    )
    tracking_branches: list[dict[str, str]] = []
    if branches_proc.returncode == 0:
        for line in branches_proc.stdout.splitlines():
            branch_name, _, upstream = line.strip().partition("|")
            if branch_name and upstream:
                tracking_branches.append({"branch": branch_name, "upstream": upstream})

    blocked_slugs = set(PUBLIC_BLOCKLIST_SLUGS)
    if "public-upstream" in remotes:
        upstream_slug = _extract_github_slug(remotes["public-upstream"].get("fetch", ""))
        if upstream_slug:
            blocked_slugs.add(upstream_slug)

    unsafe_remotes: list[str] = []
    for name, dirs in remotes.items():
        push_url = dirs.get("push", "")
        push_slug = _extract_github_slug(push_url)
        if name == "public-upstream" and push_url != "DISABLED_PUBLIC_PUSH_CLOSED_SOURCE":
            unsafe_remotes.append(f"{name}:{push_url or '<unset>'}")
        elif push_slug and push_slug in blocked_slugs:
            unsafe_remotes.append(f"{name}:{push_url}")

    unsafe_upstreams = [
        item
        for item in tracking_branches
        if item["upstream"].startswith("public-upstream/")
        or any(item["upstream"].startswith(u.split(":", 1)[0] + "/") for u in unsafe_remotes)
    ]

    push_locked = (len(unsafe_remotes) == 0) and (len(unsafe_upstreams) == 0)
    return {
        "path": str(repo_dir),
        "is_git_repo": True,
        "push_locked": push_locked,
        "remotes": remotes,
        "tracking_branches": tracking_branches,
        "unsafe_remotes": unsafe_remotes,
        "unsafe_upstreams": unsafe_upstreams,
    }


def _check_closed_source_git_locks(
    sibling_info: dict[str, dict[str, Any]],
    *,
    repo_roots: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    targets: dict[str, Path] = {}
    if repo_roots:
        for name, raw in repo_roots.items():
            targets[name] = Path(raw).expanduser().resolve()
    else:
        targets["story-media-orchestrator"] = Path(__file__).resolve().parents[1]
        for name, info in sibling_info.items():
            if info.get("path"):
                targets[name] = Path(str(info["path"]))

    repos = {name: _inspect_git_push_lock(path) for name, path in targets.items()}
    closed_source_safe = bool(repos) and all(entry.get("push_locked") is True for entry in repos.values())
    return {
        "closed_source_safe": closed_source_safe,
        "repositories": repos,
    }


def _check_codex_image() -> dict[str, Any]:
    try:
        from .codex_image import resolve_codex_executable

        exe = resolve_codex_executable()
    except ImportError:
        exe = None
    mode = os.environ.get("CODEX_IMAGE_MODE", "auto").strip() or "auto"
    responses_url = (
        os.environ.get("CODEX_IMAGE_BASE_URL", "").strip()
        or os.environ.get("CLIPROXY_BASE_URL", "").strip()
    )
    api_key_set = bool(
        os.environ.get("CODEX_IMAGE_API_KEY", "").strip()
        or os.environ.get("CLIPROXY_API_KEY", "").strip()
    )
    return {
        "mode": mode,
        "cli_found": exe is not None,
        "cli_path": str(exe) if exe is not None else None,
        "responses_base_url": responses_url or None,
        "responses_api_key_configured": api_key_set,
        "ready": exe is not None or bool(responses_url and api_key_set),
    }


def _sanitize_url(raw_url: str) -> str:
    parsed = urllib_parse.urlsplit(raw_url.strip())
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}:{parsed.port}" if parsed.port else host
    return urllib_parse.urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))


def _http_get_json(url: str, *, timeout: float) -> dict[str, Any]:
    opener = urllib_request.build_opener(_NoRedirect())
    req = urllib_request.Request(url, method="GET")
    with opener.open(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8") or "{}")
    if not isinstance(data, dict):
        raise ValueError("ComfyUI response must be a JSON object")
    return data


def _collect_string_choices(obj: Any, sink: set[str]) -> None:
    if isinstance(obj, str):
        if obj.strip():
            sink.add(obj.strip())
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _collect_string_choices(item, sink)
    elif isinstance(obj, dict):
        for val in obj.values():
            _collect_string_choices(val, sink)


def _extract_available_weights(object_info: dict[str, Any]) -> set[str]:
    choices: set[str] = set()
    for node_name in ("UNETLoader", "CLIPLoader", "VAELoader", "LoraLoaderModelOnly"):
        node_spec = object_info.get(node_name)
        if not isinstance(node_spec, dict):
            continue
        input_spec = node_spec.get("input")
        if isinstance(input_spec, dict):
            _collect_string_choices(input_spec.get("required"), choices)
            _collect_string_choices(input_spec.get("optional"), choices)
    return choices


def _weight_present(expected: str, choices: set[str]) -> bool:
    for item in choices:
        normalized = item.replace("\\", "/")
        if normalized == expected or normalized.rsplit("/", 1)[-1] == expected:
            return True
    return False


def _check_comfyui_h3(*, probe_network: bool, timeout: float) -> dict[str, Any]:
    raw_base = (
        os.environ.get("MINIMAX_H3_COMFYUI_BASE_URL", "").strip()
        or os.environ.get("COMFYUI_URL", "").strip()
    )
    raw_token = os.environ.get("MINIMAX_H3_COMFYUI_TOKEN", "").strip()
    token_configured = bool(raw_token)
    if not raw_base:
        return {
            "configured": False,
            "base_url": None,
            "token_configured": token_configured,
            "probed": False,
            "reachable": False,
            "max_vram_gib": None,
            "missing_nodes": list(REQUIRED_H3_NODE_CLASSES),
            "missing_weights": list(REQUIRED_H3_WEIGHTS),
            "latentsync_nodes_available": [],
            "latentsync_ready": False,
            "ready_for_h3_production": False,
            "error": None,
        }

    safe_base = _sanitize_url(raw_base)
    if not probe_network:
        return {
            "configured": True,
            "base_url": safe_base,
            "token_configured": token_configured,
            "probed": False,
            "reachable": None,
            "max_vram_gib": None,
            "missing_nodes": [],
            "missing_weights": [],
            "latentsync_nodes_available": [],
            "latentsync_ready": None,
            "ready_for_h3_production": True,
            "error": None,
        }

    token_q = f"?token={urllib_parse.quote(raw_token, safe='')}" if raw_token else ""
    try:
        stats = _http_get_json(f"{raw_base.rstrip('/')}/system_stats{token_q}", timeout=timeout)
        object_info = _http_get_json(f"{raw_base.rstrip('/')}/object_info{token_q}", timeout=timeout)
    except (urllib_error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "configured": True,
            "base_url": safe_base,
            "token_configured": token_configured,
            "probed": True,
            "reachable": False,
            "max_vram_gib": None,
            "missing_nodes": list(REQUIRED_H3_NODE_CLASSES),
            "missing_weights": list(REQUIRED_H3_WEIGHTS),
            "latentsync_nodes_available": [],
            "latentsync_ready": False,
            "ready_for_h3_production": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    devices = stats.get("devices") if isinstance(stats.get("devices"), list) else []
    vram_values: list[float] = []
    device_names: list[str] = []
    for dev in devices:
        if not isinstance(dev, dict):
            continue
        if dev.get("name"):
            device_names.append(str(dev["name"]))
        raw_vram = dev.get("vram_total")
        if isinstance(raw_vram, (int, float)) and raw_vram > 0:
            vram_values.append(round(float(raw_vram) / (1024.0 ** 3), 2))
    max_vram_gib = max(vram_values) if vram_values else None

    missing_nodes = [cls for cls in REQUIRED_H3_NODE_CLASSES if cls not in object_info]
    weight_choices = _extract_available_weights(object_info)
    missing_weights = [w for w in REQUIRED_H3_WEIGHTS if not _weight_present(w, weight_choices)]
    latentsync_avail = [cls for cls in OPTIONAL_LATENTSYNC_NODES if cls in object_info]
    latentsync_ready = len(latentsync_avail) == len(OPTIONAL_LATENTSYNC_NODES)
    vram_ok = max_vram_gib is None or max_vram_gib >= 20.0
    ready = (not missing_nodes) and (not missing_weights) and vram_ok

    return {
        "configured": True,
        "base_url": safe_base,
        "token_configured": token_configured,
        "probed": True,
        "reachable": True,
        "devices": device_names,
        "max_vram_gib": max_vram_gib,
        "vram_sufficient": vram_ok,
        "missing_nodes": missing_nodes,
        "missing_weights": missing_weights,
        "latentsync_nodes_available": latentsync_avail,
        "latentsync_ready": latentsync_ready,
        "ready_for_h3_production": ready,
        "error": None,
    }


def _check_audio_critic_lipsync() -> dict[str, Any]:
    lipsync_url = os.environ.get("STORY_LIPSYNC_URL", "").strip()
    lipsync_cmd = os.environ.get("STORY_LIPSYNC_CMD", "").strip()
    comfy_base = (
        os.environ.get("MINIMAX_H3_COMFYUI_BASE_URL", "").strip()
        or os.environ.get("COMFYUI_URL", "").strip()
    )
    import importlib.util

    edge_tts_ok = importlib.util.find_spec("edge_tts") is not None
    return {
        "dashscope_api_key_configured": bool(os.environ.get("DASHSCOPE_API_KEY", "").strip()),
        "edge_tts_available": edge_tts_ok,
        "plot_critic_base_url": _sanitize_url(os.environ["STORY_PLOT_CRITIC_BASE_URL"])
        if os.environ.get("STORY_PLOT_CRITIC_BASE_URL", "").strip()
        else None,
        "plot_critic_model": os.environ.get("STORY_PLOT_CRITIC_MODEL", "").strip() or None,
        "quality_evaluator_url": _sanitize_url(os.environ["STORY_QUALITY_EVALUATOR_URL"])
        if os.environ.get("STORY_QUALITY_EVALUATOR_URL", "").strip()
        else None,
        "lipsync_url": _sanitize_url(lipsync_url) if lipsync_url else None,
        "lipsync_token_configured": bool(os.environ.get("STORY_LIPSYNC_TOKEN", "").strip()),
        "lipsync_command_configured": bool(lipsync_cmd),
        "lipsync_comfy_configured": bool(lipsync_url or comfy_base),
    }


def run_preflight_doctor(
    project_dir: str | Path | None = None,
    *,
    probe_network: bool = False,
    timeout: float = 5.0,
    repo_roots: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Run local toolchain, sibling repo, closed-source git lock, and H3/ComfyUI readiness checks."""
    local_toolchain = _check_local_toolchain()
    siblings = _check_sibling_repositories(repo_roots=repo_roots)
    git_locks = _check_closed_source_git_locks(siblings, repo_roots=repo_roots)
    codex_info = _check_codex_image()
    comfy_info = _check_comfyui_h3(probe_network=probe_network, timeout=timeout)
    audio_info = _check_audio_critic_lipsync()

    blockers: list[str] = []
    warnings: list[str] = []

    if not local_toolchain["ffmpeg_available"]:
        blockers.append("ffmpeg binary is not available locally")
    if local_toolchain["cv2_version"] is None or local_toolchain["pillow_version"] is None:
        warnings.append("cv2 or Pillow is missing; 2x2 video contact-sheet diagnostics or image guards are degraded")

    missing_siblings = [name for name, info in siblings.items() if not info["exists"]]
    if missing_siblings:
        warnings.append(f"sibling repositories missing: {', '.join(missing_siblings)}")

    if not git_locks["closed_source_safe"]:
        unlocked = [
            name
            for name, info in git_locks["repositories"].items()
            if not info.get("push_locked")
        ]
        warnings.append(f"closed-source git push lock incomplete in: {', '.join(unlocked)}")

    if not comfy_info["configured"]:
        warnings.append("MINIMAX_H3_COMFYUI_BASE_URL is not configured (remote 24GB+ H3 render unavailable)")
    elif probe_network and not comfy_info["ready_for_h3_production"]:
        if not comfy_info["reachable"]:
            warnings.append(f"ComfyUI unreachable at {comfy_info['base_url']}: {comfy_info['error']}")
        if comfy_info["missing_nodes"]:
            warnings.append(f"ComfyUI missing H3 nodes: {', '.join(comfy_info['missing_nodes'])}")
        if comfy_info["missing_weights"]:
            warnings.append(f"ComfyUI missing H3 weights: {', '.join(comfy_info['missing_weights'])}")
        if comfy_info.get("vram_sufficient") is False:
            warnings.append(f"ComfyUI GPU VRAM {comfy_info.get('max_vram_gib')} GiB < 20.0 GiB")

    project_report: dict[str, Any] | None = None
    if project_dir:
        proj_path = Path(project_dir).resolve()
        manifest_file = proj_path / "project.json"
        if manifest_file.is_file():
            manifest = ProjectManifest.load(manifest_file)
            project_warnings: list[str] = []
            if manifest.mode == "render" and manifest.video_provider == "comfyui" and not comfy_info["configured"]:
                project_warnings.append("project mode=render & video_provider=comfyui but MINIMAX_H3_COMFYUI_BASE_URL is unset")
            if manifest.image_source == "codex" and not codex_info["ready"]:
                project_warnings.append("project image_source=codex but neither local codex CLI nor Responses API key is configured")
            if manifest.tts_source == "cosyvoice" and not audio_info["dashscope_api_key_configured"]:
                project_warnings.append("project tts_source=cosyvoice but DASHSCOPE_API_KEY is unset")
            elif manifest.tts_source == "edge" and not audio_info.get("edge_tts_available"):
                project_warnings.append("project tts_source=edge but edge_tts package is not installed")
            if manifest.lipsync_provider == "http" and not audio_info["lipsync_url"]:
                project_warnings.append("project lipsync_provider=http but STORY_LIPSYNC_URL is unset")
            elif manifest.lipsync_provider == "command" and not audio_info["lipsync_command_configured"]:
                project_warnings.append("project lipsync_provider=command but STORY_LIPSYNC_CMD is unset")
            elif manifest.lipsync_provider == "comfyui" and not audio_info["lipsync_comfy_configured"]:
                project_warnings.append("project lipsync_provider=comfyui but MINIMAX_H3_COMFYUI_BASE_URL and STORY_LIPSYNC_URL are unset")
            warnings.extend(project_warnings)
            project_report = {
                "project_dir": str(proj_path),
                "project_id": manifest.project_id,
                "mode": manifest.mode,
                "image_source": manifest.image_source,
                "tts_source": manifest.tts_source,
                "video_provider": manifest.video_provider,
                "lipsync_provider": manifest.lipsync_provider,
                "execution_strategy": manifest.execution_strategy,
                "project_warnings": project_warnings,
            }
        else:
            warnings.append(f"project.json not found in {proj_path}")

    if blockers:
        overall_status = "blocked"
    elif warnings:
        overall_status = "warnings"
    else:
        overall_status = "ready"

    return {
        "schema": "orchestrator-doctor/v1",
        "overall_status": overall_status,
        "ready_for_local_preview": bool(local_toolchain["ffmpeg_available"]),
        "ready_for_h3_production": bool(comfy_info["ready_for_h3_production"]),
        "closed_source_safe": bool(git_locks["closed_source_safe"]),
        "blockers": blockers,
        "warnings": warnings,
        "checks": {
            "local_toolchain": local_toolchain,
            "sibling_repositories": siblings,
            "closed_source_git_locks": git_locks,
            "codex_image": codex_info,
            "comfyui_h3": comfy_info,
            "audio_critic_lipsync": audio_info,
            "project": project_report,
        },
    }
