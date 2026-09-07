from unittest.mock import Mock
from story_media_orchestrator.project_video import ProjectVideoProvider
from story_media_orchestrator.registry import ArtifactRegistry
from story_media_orchestrator.manifest import Shot, ProjectManifest


def test_project_video_uses_visual_action_and_keeps_artifact(tmp_path):
    store = ArtifactRegistry(tmp_path / 'artifacts')
    image = tmp_path / 'image.png'
    image.write_bytes(b'image')
    ref = store.put_bytes(b'\x00\x00\x00\x18ftypisomvideo')
    adapter = Mock()
    adapter.run.return_value = {'execution': {'state':'succeeded', 'artifact_ref':ref}}
    shot = Shot('shot-1', '动作与对白', scene_context={'visual_action':'推开门', 'source_spans':['story-package/s/a']})
    output = ProjectVideoProvider(adapter, store).generate(shot, image, tmp_path / 'video' / 'one.mp4')
    assert output.read_bytes() == store.get_bytes(ref)
    assert adapter.run.call_args.kwargs['prompt'] == '推开门'
    assert adapter.run.call_args.kwargs['story_spans'] == ['story-package/s/a']


def test_render_cli_constructs_video_provider(monkeypatch, tmp_path):
    from story_media_orchestrator.cli import main
    ProjectManifest.create('p', 'Story.').save(tmp_path / 'project.json')
    provider = object()
    build = Mock(return_value=provider)
    render = Mock(return_value=tmp_path / 'preview.mp4')
    monkeypatch.setattr('story_media_orchestrator.project_video.build_project_video_provider', build)
    monkeypatch.setattr('story_media_orchestrator.cli.render_preview', render)
    main(['run', str(tmp_path), '--mode', 'render'])
    assert render.call_args.kwargs['video_provider'] is provider
    assert build.call_args.args[0].mode == 'render'

def test_completed_render_resumes_without_video_service(monkeypatch, tmp_path):
    from story_media_orchestrator.cli import main
    from story_media_orchestrator.preview import _shot_cache_key
    from story_media_orchestrator.project_video import needs_video_generation
    manifest = ProjectManifest.create('p', 'Story.')
    manifest.mode = 'render'
    shot = manifest.shots[0]
    shot.mode, shot.status = 'render', 'done'
    for kind in ['image', 'audio', 'video']:
        path = tmp_path / kind
        path.write_bytes(b'cached')
        shot.assets[kind] = str(path)
    shot.cache_key = _shot_cache_key(manifest, shot)
    manifest.save(tmp_path / 'project.json')
    build = Mock(side_effect=AssertionError('cached render must not need video service'))
    render = Mock(return_value=tmp_path / 'preview.mp4')
    monkeypatch.setattr('story_media_orchestrator.project_video.build_project_video_provider', build)
    monkeypatch.setattr('story_media_orchestrator.cli.render_preview', render)
    main(['resume', str(tmp_path)])
    build.assert_not_called()
    assert render.call_args.kwargs['video_provider'] is None
    shot.text = 'changed'
    assert needs_video_generation(manifest)

def test_project_video_route_switch_preserves_other_assets(monkeypatch, tmp_path):
    from story_media_orchestrator.cli import main
    manifest = ProjectManifest.create('p', 'Story.')
    manifest.mode = 'render'
    shot = manifest.shots[0]
    shot.assets = {'image':'cached-image', 'audio':'cached-audio', 'video':'old-video'}
    shot.mode = 'render'
    manifest.save(tmp_path / 'project.json')
    monkeypatch.setattr('story_media_orchestrator.project_video.build_project_video_provider', Mock())
    monkeypatch.setattr('story_media_orchestrator.cli.render_preview', Mock(return_value=tmp_path / 'preview.mp4'))
    main(['run', str(tmp_path), '--video-provider', 'dashscope'])
    restored = ProjectManifest.load(tmp_path / 'project.json')
    assert restored.video_provider == 'dashscope'
    assert restored.shots[0].assets == {'image':'cached-image', 'audio':'cached-audio'}
    assert restored.shots[0].mode == 'preview'

def test_project_video_prompt_carries_scene_and_character_context(tmp_path):
    store = ArtifactRegistry(tmp_path / 'artifacts')
    ref = store.put_bytes(b'\x00\x00\x00\x18ftypisomvideo')
    adapter = Mock()
    adapter.run.return_value = {'execution': {'state':'succeeded', 'artifact_ref':ref}}
    shot = Shot('s', 'text', scene_context={'visual_action':'推开门', 'location':'钟楼', 'lighting':'晨光',
        'continuity':'保持蓝色外套', 'characters':[{'name':'阿澄','clothing':'蓝色外套','secret':'隐藏剧情'}]})
    ProjectVideoProvider(adapter, store).generate(shot, None, tmp_path / 'video.mp4')
    prompt = adapter.run.call_args.kwargs['prompt']
    assert all(value in prompt for value in ['推开门','钟楼','晨光','阿澄','蓝色外套'])
    assert '隐藏剧情' not in prompt


def test_project_wan_preserves_scene_negative_prompt(monkeypatch, tmp_path):
    from story_media_orchestrator.project_video import build_project_video_provider
    from types import SimpleNamespace
    configured = SimpleNamespace(api_key='test-key', base_url='https://example.invalid')
    image_package = SimpleNamespace(DashScopeImageProvider=SimpleNamespace(
        from_nanocodex_config=lambda: configured))
    monkeypatch.setattr('story_media_orchestrator.project_video._import_from_root',
                        lambda root, name: image_package if name == 'story_image_agent' else SimpleNamespace())
    import sys
    monkeypatch.setitem(sys.modules, 'story_video_agent.dashscope',
                        SimpleNamespace(DashScopeWanAdapter=Mock()))
    run = Mock(return_value={'execution': {'state': 'pending'}})
    monkeypatch.setattr('story_media_orchestrator.wan_video.run_wan_video', run)
    manifest = ProjectManifest.create('p', 'Story.')
    manifest.video_provider = 'dashscope'
    manifest.video_options = {'model': 'custom', 'size': '720*1280', 'duration_seconds': 7}
    provider = build_project_video_provider(manifest, tmp_path)
    provider.adapter.run(prompt='信使走进钟楼', scene={'negative': '文字，水印'})
    assert run.call_args.kwargs == {'prompt': '信使走进钟楼', 'negative_prompt': '文字，水印', **manifest.video_options}
    provider.adapter.run(prompt='信使走进钟楼', scene={})
    assert run.call_args.kwargs['negative_prompt'] is None


def test_video_options_persist_and_invalidate_video_only(monkeypatch, tmp_path):
    import json
    from story_media_orchestrator.cli import main
    manifest = ProjectManifest.create('p', 'Story.')
    manifest.shots[0].assets = {'image': 'image', 'audio': 'audio', 'video': 'video'}
    manifest.save(tmp_path / 'project.json')
    monkeypatch.setattr('story_media_orchestrator.cli.render_preview', Mock(return_value='preview.mp4'))
    options = {'model': 'custom-model', 'size': '720*1280', 'duration_seconds': 7}
    main(['run', str(tmp_path), '--video-provider', 'dashscope', '--video-options', json.dumps(options)])
    saved = ProjectManifest.load(tmp_path / 'project.json')
    assert saved.video_options == options
    assert saved.shots[0].assets == {'image': 'image', 'audio': 'audio'}
    main(['resume', str(tmp_path)])
    assert ProjectManifest.load(tmp_path / 'project.json').video_options == options


def test_invalid_video_options_rejected_before_manifest_changes(monkeypatch, tmp_path):
    import pytest
    from story_media_orchestrator.cli import main
    ProjectManifest.create('p', 'Story.').save(tmp_path / 'project.json')
    original = (tmp_path / 'project.json').read_bytes()
    with pytest.raises(ValueError):
        main(['run', str(tmp_path), '--video-options', '{"duration_seconds":0}'])
    assert (tmp_path / 'project.json').read_bytes() == original


def test_project_comfy_options_reach_original_graph(monkeypatch, tmp_path):
    from story_media_orchestrator.project_video import build_project_video_provider
    from story_media_orchestrator.adapters import StoryVideoAdapter
    manifest = ProjectManifest.create('p', 'Story.')
    manifest.video_options = {'width': 640, 'height': 384, 'length': 96, 'fps': 16, 'seed': 42, 'turbo': True}
    monkeypatch.setenv('MINIMAX_H3_COMFYUI_BASE_URL', 'http://127.0.0.1:8188')
    execute = Mock(return_value={'state': 'pending'})
    monkeypatch.setattr(StoryVideoAdapter, '_execute_comfy', execute)
    provider = build_project_video_provider(manifest, tmp_path)
    ref = provider.registry.put_bytes(b'image')
    provider.adapter.run(first_frame_ref=ref, story_spans=['scene-01'], scene={}, prompt='Open the door')
    graph = execute.call_args.args[0]
    video = next(node['inputs'] for node in graph.values() if node['class_type'] == 'MiniMaxH3ImageToVideo')
    assert (video['width'], video['height'], video['length']) == (640, 384, 96)
    assert any(node['inputs'].get('seed') == 42 or node['inputs'].get('noise_seed') == 42 for node in graph.values())
    assert any(node['inputs'].get('frame_rate') == 16 or node['inputs'].get('fps') == 16 for node in graph.values())


def test_project_rejects_dimensions_smaller_than_original_comfy_minimum():
    import pytest
    from story_media_orchestrator.project_video import validate_video_options
    for key in ('width', 'height'):
        with pytest.raises(ValueError):
            validate_video_options({key: 32})


def test_explicit_defaults_and_inactive_route_options_preserve_video(monkeypatch, tmp_path):
    import json
    from story_media_orchestrator.cli import main
    manifest = ProjectManifest.create('p', 'Story.')
    manifest.shots[0].assets = {'image': 'image', 'audio': 'audio', 'video': 'video'}
    manifest.shots[0].mode = 'render'
    manifest.save(tmp_path / 'project.json')
    monkeypatch.setattr('story_media_orchestrator.cli.render_preview', Mock(return_value='preview.mp4'))
    options = {'width': 864, 'height': 480, 'length': 124, 'fps': 24,
               'seed': 0, 'turbo': False, 'model': 'inactive-wan-model'}
    main(['run', str(tmp_path), '--video-options', json.dumps(options)])
    saved = ProjectManifest.load(tmp_path / 'project.json')
    assert saved.shots[0].assets['video'] == 'video'
    assert saved.shots[0].mode == 'render'
    assert saved.video_options == options
    options['width'] = 640
    main(['run', str(tmp_path), '--video-options', json.dumps(options)])
    saved = ProjectManifest.load(tmp_path / 'project.json')
    assert saved.shots[0].assets == {'image': 'image', 'audio': 'audio'}


def test_invalid_wan_project_stops_before_media_generation(monkeypatch, tmp_path):
    import pytest
    from story_media_orchestrator.cli import main
    manifest = ProjectManifest.create('p', 'Story.')
    manifest.mode = 'render'
    manifest.video_provider = 'dashscope'
    manifest.video_options = {'duration_seconds': 7}
    manifest.save(tmp_path / 'project.json')
    render = Mock(side_effect=AssertionError('must not generate images'))
    monkeypatch.setattr('story_media_orchestrator.cli.render_preview', render)
    with pytest.raises(ValueError, match='5'):
        main(['run', str(tmp_path)])
    render.assert_not_called()
    saved = ProjectManifest.load(tmp_path / 'project.json')
    assert saved.status == 'failed'
    assert 'video provider setup failed' in saved.error
    assert not saved.shots[0].assets
