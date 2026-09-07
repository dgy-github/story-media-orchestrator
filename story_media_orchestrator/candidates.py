"""Persistent candidate selection shared by CLI and desktop."""
from pathlib import Path
from uuid import uuid4


def select_candidate(manifest, shot_id, candidate_id, role='image'):
    if role not in {'image', 'last_frame'}:
        raise ValueError('invalid frame role')
    shot = manifest.shot(shot_id)
    candidate = next((c for c in shot.candidates if c['id'] == candidate_id), None)
    if candidate is None or not Path(candidate['path']).is_file():
        raise ValueError('candidate is missing')
    from .preview import _shot_cache_key
    if candidate.get('source_key') and candidate['source_key'] != _shot_cache_key(manifest, shot):
        raise ValueError('story or image settings changed; generate current candidates')
    if candidate.get('quality', {}).get('decision') == 'failed':
        raise ValueError('candidate failed quality review; generate another candidate')
    shot.assets[role] = candidate['path']
    shot.assets.pop('video', None)
    shot.mode = 'preview'
    shot.status = 'generated'
    shot.error = None
    shot.cache_key = _shot_cache_key(manifest, shot)
    manifest.approvals.pop('assets', None)
    manifest.output = None


def generate_candidates(manifest, root, shot_id, count=2):
    if type(count) is not int or not 1 <= count <= 8:
        raise ValueError('candidate count must be 1 to 8')
    if manifest.image_source != 'dashscope':
        raise ValueError('candidate generation requires DashScope image source')
    from .providers import build_project_image_provider
    from .quality import evaluate_artifact
    from .preview import _shot_cache_key
    root = Path(root).resolve()
    shot = manifest.shot(shot_id)
    provider = build_project_image_provider(manifest, root)
    # A batch ID is saved before submitting. Repeating an interrupted batch
    # reuses the existing adapter receipts instead of billing twice.
    shot.candidate_batch = shot.candidate_batch or uuid4().hex
    manifest.save(root/'project.json')
    result = provider.adapter.run({**shot.scene_context, 'summary':shot.text,
        'description':shot.scene_context.get('visual_action') or shot.text,
        'action':shot.scene_context.get('visual_action') or shot.text},
        shot.scene_context.get('source_spans') or [shot.scene_id, shot.id],
        candidate_count=count, batch_id=shot.candidate_batch)
    outputs = result.get('candidate_results', [])
    directory = root/'candidates'/shot.id
    directory.mkdir(parents=True, exist_ok=True)
    for index, output in enumerate(outputs):
        if output.get('status') != 'succeeded':
            continue
        ref = output['artifact_ref']
        target = directory/(ref.rsplit('/',1)[-1]+'.png')
        target.write_bytes(provider.registry.get_bytes(ref))
        candidate_id = f'{shot.candidate_batch}-{index}'
        if any(c['id'] == candidate_id for c in shot.candidates):
            continue
        shot.candidates.append({'id': candidate_id, 'path':str(target), 'artifact_ref':ref, 'source_key':_shot_cache_key(manifest, shot),
                               'quality':evaluate_artifact(output, 'image')})
        manifest.save(root/'project.json')
    if not any(c.get('status') == 'succeeded' for c in outputs):
        raise RuntimeError('no successful candidates')
    if all(c.get('status') == 'succeeded' for c in outputs) and len(outputs) == count:
        shot.candidate_batch = None
    manifest.approvals.pop('assets', None)
    manifest.save(root/'project.json')
