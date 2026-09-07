import json
import os
from pathlib import Path
import subprocess
import sys


def test_desktop_utf8_payload_survives_windows_legacy_stdio(tmp_path):
    project = tmp_path / "shop"
    story = "清晨，信使推开钟楼的门。少女说：你好！🎬"
    environment = {**os.environ, "PYTHONIOENCODING": "gbk:surrogateescape", "PYTHONUTF8": "0"}
    result = subprocess.run([sys.executable, "-m", "story_media_orchestrator.cli", "create", str(project), "--create-stdin"],
                            input=json.dumps({"story": story}, ensure_ascii=False).encode("utf-8"),
                            capture_output=True, env=environment, cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    saved = json.loads((project / "project.json").read_text(encoding="utf-8"))
    assert saved["story"] == story
    assert not list(project.glob("*.tmp"))
