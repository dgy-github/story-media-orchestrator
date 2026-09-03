"""Small local Tk UI for configuring and previewing a media run."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog
import json
import threading
import os
import subprocess
import secrets
from pathlib import Path
from typing import Any, Callable
from urllib import request as url_request


DEFAULTS = {
    "STORY_CAMPAIGN_ROOT": r"D:\github_dgy\microcodex-short-drama-studio",
    "STORY_IMAGE_AGENT_ROOT": r"D:\github_dgy\story-image-agent",
    "STORY_VIDEO_AGENT_ROOT": r"D:\github_dgy\story-video-agent",
    "STORY_MODEL": "configured-by-story-runtime",
    "STORY_IMAGE_MODEL": "wan2.2-t2i-flash",
    "STORY_IMAGE_SIZE": "720*1280",
    "STORY_VIDEO_MODEL": "minimax_h3_fl2va",
    "STORY_VIDEO_TURBO": "false",
    "STORY_VIDEO_STEPS": "20",
    "MINIMAX_H3_COMFYUI_BASE_URL": "http://61.157.218.59:31340",
    "STORY_SIDECAR_URL": "http://127.0.0.1:8765",
}

INTERNAL_PATH_KEYS = ("STORY_CAMPAIGN_ROOT", "STORY_IMAGE_AGENT_ROOT", "STORY_VIDEO_AGENT_ROOT")


def apply_internal_defaults(config: dict[str, str]) -> dict[str, str]:
    """Return config with sibling agent paths filled from the bundled workspace."""
    result = dict(config)
    for key in INTERNAL_PATH_KEYS:
        result[key] = result.get(key) or DEFAULTS[key]
    return result


def validate_ui_config(config: dict[str, str]) -> list[str]:
    config = apply_internal_defaults(config)
    errors = []
    for key in ("STORY_CAMPAIGN_ROOT", "STORY_IMAGE_AGENT_ROOT", "STORY_VIDEO_AGENT_ROOT"):
        if not config.get(key) or not Path(config[key]).is_dir():
            errors.append(f"路径不存在: {key}")
    if not config.get("MINIMAX_H3_COMFYUI_BASE_URL", "").startswith(("http://", "https://")):
        errors.append("ComfyUI URL 必须是 http(s) 地址")
    if config.get("STORY_VIDEO_TURBO", "false").lower() not in {"true", "false"}:
        errors.append("视频 turbo 必须是 true 或 false")
    try:
        steps = int(config.get("STORY_VIDEO_STEPS", "20"))
        if not 1 <= steps <= 100: raise ValueError
    except ValueError:
        errors.append("视频 steps 必须是 1-100 的整数")
    return errors


def launch(run: Callable[[dict[str, str]], str] | None = None, orchestrator: Any | None = None) -> None:
    root = tk.Tk()
    root.title("Story Media Orchestrator")
    root.geometry("760x520")
    fields = {
        "Story model": "STORY_MODEL",
        "Image model": "STORY_IMAGE_MODEL",
        "Image size": "STORY_IMAGE_SIZE",
        "Video model": "STORY_VIDEO_MODEL",
        "Video turbo": "STORY_VIDEO_TURBO",
        "Video steps": "STORY_VIDEO_STEPS",
        "ComfyUI URL": "MINIMAX_H3_COMFYUI_BASE_URL",
        "DashScope API key": "DASHSCOPE_API_KEY",
        "Sidecar URL": "STORY_SIDECAR_URL",
        "Sidecar token": "STORY_SIDECAR_TOKEN",
        "Capability URL": "MICROCODEX_CAPABILITY_URL",
        "Capability token": "MICROCODEX_CAPABILITY_TOKEN",
    }
    values: dict[str, tk.Entry] = {}
    form = ttk.Frame(root, padding=16); form.pack(fill="x")
    for row, (label, key) in enumerate(fields.items()):
        ttk.Label(form, text=label, width=20).grid(row=row, column=0, sticky="w", pady=4)
        entry = ttk.Entry(form, width=70, show="•" if any(word in label.lower() for word in ("token", "api key")) else "")
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        values[key] = entry
        if key.endswith("_ROOT"):
            ttk.Button(form, text="浏览", command=lambda e=entry: e.insert(0, filedialog.askdirectory())).grid(row=row, column=2, padx=4)
    form.columnconfigure(1, weight=1)
    status = tk.Text(root, height=12, state="disabled")
    status.pack(fill="both", expand=True, padx=16, pady=8)

    def write(message: str) -> None:
        status.configure(state="normal"); status.insert("end", message + "\n"); status.see("end"); status.configure(state="disabled")

    def execute() -> None:
        config = apply_internal_defaults({key: entry.get().strip() for key, entry in values.items()})
        errors = validate_ui_config(config)
        if errors:
            write("[config-error] " + "；".join(errors)); return
        write("[queued] story → image → video")
        story_path = filedialog.askopenfilename(title="选择 story-package 输入 JSON", filetypes=(("JSON", "*.json"),))
        if not story_path:
            write("[cancelled] 未选择故事输入")
            return
        def worker() -> None:
            try:
                for key, value in config.items():
                    if value:
                        os.environ[key] = value
                story_input = json.loads(open(story_path, encoding="utf-8").read())
                if orchestrator is not None:
                    result = orchestrator.run(story_input=story_input)
                    message = "[succeeded] " + json.dumps({"status": result.get("status"), "scene_index": result.get("scene_index")}, ensure_ascii=False)
                else:
                    message = run(config) if run else "[preview] fake run only; no provider request sent"
                root.after(0, write, message)
            except Exception as exc:
                root.after(0, write, f"[failed] {type(exc).__name__}: {exc}")
        threading.Thread(target=worker, daemon=True).start()

    ttk.Button(root, text="Run single-scene preview", command=execute).pack(pady=8)

    config_path = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "StoryMediaOrchestrator" / "config.json"
    def save_config() -> None:
        config = apply_internal_defaults({key: entry.get().strip() for key, entry in values.items()})
        errors = validate_ui_config(config)
        if errors:
            write("[config-error] " + "；".join(errors)); return
        config_path.parent.mkdir(parents=True, exist_ok=True)
        # Paths are persisted for runtime compatibility but are intentionally not shown as editable fields.
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        write(f"[saved] 配置已保存：{config_path}")

    def load_config() -> None:
        if not config_path.is_file():
            write("[info] 尚无已保存配置")
            return
        data = json.loads(config_path.read_text(encoding="utf-8"))
        for key, entry in values.items():
            entry.delete(0, "end"); entry.insert(0, data.get(key, ""))
        write("[loaded] 已加载本地配置")

    def generate_sidecar_token() -> None:
        token = secrets.token_urlsafe(32)
        entry = values["STORY_SIDECAR_TOKEN"]
        entry.delete(0, "end"); entry.insert(0, token)
        write("[generated] 已生成新的 sidecar token（仅保存在本机配置）")

    def start_stack() -> None:
        config = apply_internal_defaults({key: entry.get().strip() for key, entry in values.items()})
        errors = validate_ui_config(config)
        if errors:
            write("[config-error] " + "；".join(errors)); return
        save_config()
        for key, value in config.items():
            if value: os.environ[key] = value
        script = Path(__file__).resolve().parents[1] / "scripts" / "start-local-stack.ps1"
        subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                         cwd=str(script.parent.parent), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        write("[started] 正在启动本地 sidecar，请查看 sidecar.log")

    def test_connections() -> None:
        config = apply_internal_defaults({key: entry.get().strip() for key, entry in values.items()})
        errors = validate_ui_config(config)
        if errors:
            write("[config-error] " + "；".join(errors)); return
        def worker() -> None:
            try:
                with url_request.urlopen(config["MINIMAX_H3_COMFYUI_BASE_URL"] + "/system_stats?token=", timeout=8) as response:
                    root.after(0, write, f"[ok] MiniMax H3 ComfyUI HTTP {response.status}")
            except Exception as exc:
                root.after(0, write, f"[failed] ComfyUI: {type(exc).__name__}")
            if config.get("STORY_SIDECAR_URL") and config.get("STORY_SIDECAR_TOKEN"):
                try:
                    req = url_request.Request(config["STORY_SIDECAR_URL"] + "/health", headers={"Authorization": "Bearer " + config["STORY_SIDECAR_TOKEN"]})
                    with url_request.urlopen(req, timeout=8) as response: root.after(0, write, f"[ok] story sidecar HTTP {response.status}")
                except Exception as exc: root.after(0, write, f"[failed] story sidecar: {type(exc).__name__}")
        threading.Thread(target=worker, daemon=True).start()

    buttons = ttk.Frame(root); buttons.pack(pady=4)
    ttk.Button(buttons, text="保存配置", command=save_config).pack(side="left", padx=4)
    ttk.Button(buttons, text="加载配置", command=load_config).pack(side="left", padx=4)
    ttk.Button(buttons, text="生成 sidecar token", command=generate_sidecar_token).pack(side="left", padx=4)
    ttk.Button(buttons, text="启动本地栈", command=start_stack).pack(side="left", padx=4)
    ttk.Button(buttons, text="测试连接", command=test_connections).pack(side="left", padx=4)
    for key, default in DEFAULTS.items():
        if key not in values:
            continue
        values[key].insert(0, default)
    load_config()
    root.mainloop()
