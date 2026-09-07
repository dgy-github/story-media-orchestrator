"""Content-bound approvals for storyboard and generated assets."""
import hashlib
import json
from pathlib import Path


def fingerprint(manifest, stage):
    data = {'story': manifest.story, 'shots': [
        {'id': s.id, 'text': s.text, 'context': s.scene_context} for s in manifest.shots]}
    if stage == 'assets':
        data['settings'] = [manifest.mode, manifest.tts_source, manifest.image_source,
                            manifest.image_model, manifest.image_size, manifest.video_options]
        data['assets'] = [{
            'id': s.id, 'duration': s.duration, 'subtitle': s.subtitle, 'motion': s.motion, 'transition': s.transition,
            'files': {k: hashlib.sha256(Path(v).read_bytes()).hexdigest()
                      for k, v in sorted(s.assets.items())}}
            for s in manifest.shots]
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def block_unapproved(manifest, root, stage):
    if manifest.approvals.get(stage) == fingerprint(manifest, stage):
        return False
    manifest.status = 'awaiting_storyboard_review' if stage == 'storyboard' else 'awaiting_asset_review'
    manifest.output = None
    manifest.error = None
    manifest.save(Path(root) / 'project.json')
    return True


def approve(manifest):
    stage = {'awaiting_storyboard_review': 'storyboard', 'awaiting_asset_review': 'assets'}.get(manifest.status)
    if stage is None:
        raise ValueError('No project review is pending; run the project first')
    manifest.approvals[stage] = fingerprint(manifest, stage)
