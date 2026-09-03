"""Small local Tk UI for configuring and previewing a media run."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog
import json
import threading
from typing import Any, Callable


def launch(run: Callable[[dict[str, str]], str] | None = None, orchestrator: Any | None = None) -> None:
    root = tk.Tk()
    root.title("Story Media Orchestrator")
    root.geometry("760x520")
    fields = {
        "Story agent path": "STORY_CAMPAIGN_ROOT",
        "Image agent path": "STORY_IMAGE_AGENT_ROOT",
        "Video agent path": "STORY_VIDEO_AGENT_ROOT",
        "Story model": "STORY_MODEL",
        "Image model": "STORY_IMAGE_MODEL",
        "Video model": "STORY_VIDEO_MODEL",
        "ComfyUI URL": "MINIMAX_H3_COMFYUI_BASE_URL",
    }
    values: dict[str, tk.Entry] = {}
    form = ttk.Frame(root, padding=16); form.pack(fill="x")
    for row, (label, key) in enumerate(fields.items()):
        ttk.Label(form, text=label, width=20).grid(row=row, column=0, sticky="w", pady=4)
        entry = ttk.Entry(form, width=70)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        values[key] = entry
    form.columnconfigure(1, weight=1)
    status = tk.Text(root, height=12, state="disabled")
    status.pack(fill="both", expand=True, padx=16, pady=8)

    def write(message: str) -> None:
        status.configure(state="normal"); status.insert("end", message + "\n"); status.see("end"); status.configure(state="disabled")

    def execute() -> None:
        config = {key: entry.get().strip() for key, entry in values.items()}
        write("[queued] story → image → video")
        story_path = filedialog.askopenfilename(title="选择 story-package 输入 JSON", filetypes=(("JSON", "*.json"),))
        if not story_path:
            write("[cancelled] 未选择故事输入")
            return
        def worker() -> None:
            try:
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
    root.mainloop()
