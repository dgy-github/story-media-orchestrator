from story_media_orchestrator.manifest import ProjectManifest
from story_media_orchestrator.preview import render_preview


def test_two_reviews_block_generation_and_assembly(tmp_path):
    from story_media_orchestrator.review_gate import approve
    manifest = ProjectManifest.create('test', 'A courier arrives.')
    assert render_preview(manifest, tmp_path, enforce_review=True) is None
    assert manifest.status == 'awaiting_storyboard_review'
    assert manifest.shots[0].assets == {}
    approve(manifest)
    assert render_preview(manifest, tmp_path, enforce_review=True) is None
    assert manifest.status == 'awaiting_asset_review'
    assert manifest.shots[0].assets['audio']
    approve(manifest)
    assert render_preview(manifest, tmp_path, enforce_review=True).is_file()
    manifest.shots[0].text = 'A different story'
    assert render_preview(manifest, tmp_path, enforce_review=True) is None
    assert manifest.status == 'awaiting_storyboard_review'


def test_cli_resume_keeps_required_review(tmp_path):
    from story_media_orchestrator.cli import main
    main(['create', str(tmp_path), 'A courier arrives.'])
    main(['run', str(tmp_path), '--require-review'])
    main(['resume', str(tmp_path)])
    restored = ProjectManifest.load(tmp_path / 'project.json')
    assert restored.status == 'awaiting_storyboard_review'
    assert restored.shots[0].assets == {}


def test_changed_asset_requires_another_review(tmp_path):
    from pathlib import Path
    from story_media_orchestrator.review_gate import approve
    manifest = ProjectManifest.create('test', 'A courier arrives.')
    render_preview(manifest, tmp_path, enforce_review=True)
    approve(manifest)
    render_preview(manifest, tmp_path, enforce_review=True)
    approve(manifest)
    Path(manifest.shots[0].assets['image']).write_text('changed')
    assert render_preview(manifest, tmp_path, enforce_review=True) is None
    assert manifest.status == 'awaiting_asset_review'
