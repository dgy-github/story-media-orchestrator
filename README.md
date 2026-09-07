# Story Media Orchestrator

**Turn a story or script into a storyboarded, voiced video preview.**

The project is a resumable multimodal workflow engine. It turns prose into scenes and shots, carries structured assets between providers, and assembles a preview that can be regenerated after a failure.

```text
Story / Script → Normalize → Scene Breakdown → Shot Plan
             → Image / Video / TTS → Timeline → Preview.mp4
```

## Quick start

```powershell
python -m story_media_orchestrator.cli create demo-project "A courier arrives. The city wakes."
python -m story_media_orchestrator.cli run demo-project
python -m story_media_orchestrator.cli resume demo-project
python -m story_media_orchestrator.cli retry demo-project shot-01
```

The demo works without model credentials. If `ffmpeg` is installed it writes `preview.mp4`; otherwise it writes a text preview so the workflow remains inspectable.

## Project manifest

Each project is self-contained:

```text
project/
  project.json
  frames/ audio/ subtitles.srt
  preview.mp4
```

`project.json` tracks each shot as `planned → generating → generated → assembled → done` (or `failed`), along with asset references, review state, attempts, cache keys, and output lineage. Resume and retry operate on this manifest instead of starting from scratch. Approve a reviewed shot with `story-media review project shot-01`.

## Architecture

Providers implement stable contracts (`ImageProvider`, `VideoProvider`, `TTSProvider`) while the orchestrator owns state, caching, retries, quality gates, and timeline assembly. Preview Mode uses image motion, voice, subtitles, and FFmpeg. Render Mode can replace selected shots with image-to-video or text-to-video providers and still fall back to the preview path.

The Python package contains the orchestration contracts and adapters. `apps/desktop` is the Rust/Tauri/Svelte shell and can consume the same runtime state.

## For teams

Need a custom AI media pipeline? We can help with private deployment, provider integration, brand and style workflows, and batch content generation. Open an issue to discuss a collaboration.

## Development

```powershell
$env:PYTHONPATH='.'; pytest -q
npm run build
cargo check --manifest-path apps/desktop/src-tauri/Cargo.toml
```
