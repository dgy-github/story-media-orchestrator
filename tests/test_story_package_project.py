import io
import json
import sys
import pytest
from story_media_orchestrator.cli import main
from story_media_orchestrator.manifest import ProjectManifest

def test_package_roundtrip_preserves_scene_boundaries(tmp_path, monkeypatch):
    package = {"schema":"story-package/v1", "characters":[{"node_id":"ch-1","name":"阿澄"}], "scenes":[{"node_id":"scene-7","lines":[{"kind":"action","text":"他推开门。屋里很暗。"},{"kind":"dialogue","speaker":"ch-1","text":"有人吗？"}]}]}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"story":"draft", "story_package":package, "story_job":{"schema":"story-job/v1", "genre_pack_id":"family-grounded-v1"}})))
    main(["create",str(tmp_path),"--create-stdin"])
    result = ProjectManifest.load(tmp_path / "project.json")
    assert result.story_job["genre_pack_id"] == "family-grounded-v1"
    assert result.story_package == package
    assert len(result.shots) == 1
    assert result.shots[0].scene_id == "scene-7"
    assert "有人吗" in result.shots[0].text

def test_invalid_package_does_not_create_project(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"story":"draft","story_package":{"schema":"story-package/v1","scenes":[]}})))
    with pytest.raises(ValueError): main(["create",str(tmp_path),"--create-stdin"])
    assert not (tmp_path / "project.json").exists()


def test_scene_context_and_cache_follow_cast_and_location():
    from story_media_orchestrator.preview import _cache_key
    package = {"schema":"story-package/v1", "characters":[{"node_id":"ch-1","name":"阿澄"},{"node_id":"ch-2","name":"路人"}], "scenes":[{"node_id":"scene-1","location":"钟楼", "lines":[{"kind":"dialogue","speaker":"story-package/ch-1","text":"到时间了。"}]}]}
    shot = ProjectManifest.from_story_package("p", package).shots[0]
    assert shot.scene_context["location"] == "钟楼"
    assert [c["name"] for c in shot.scene_context["characters"]] == ["阿澄"]
    key = _cache_key(shot, "preview")
    shot.scene_context["location"] = "码头"
    assert key != _cache_key(shot, "preview")

def test_silent_character_named_in_action_is_preserved():
    from story_media_orchestrator.manifest import ProjectManifest
    package = {'schema':'story-package/v1', 'characters':[
        {'node_id':'silent','name':'阿澄'}, {'node_id':'speaker','name':'老人'},
        {'node_id':'absent','name':'Ann'}], 'scenes':[{'node_id':'scene', 'lines':[
            {'kind':'action','text':'阿澄扶起 Anna，老人站在门口。'},
            {'kind':'dialogue','speaker':'characters/speaker','text':'小心。'}]}]}
    shot = ProjectManifest.from_story_package('project',package).shots[0]
    assert [c['node_id'] for c in shot.scene_context['characters']] == ['silent','speaker']
    assert shot.scene_context['character_sources'] == {'silent':'action_name_match','speaker':'speaker_reference'}

def test_original_story_spans_retain_scene_and_line_hierarchy():
    from story_media_orchestrator.manifest import ProjectManifest
    package = {'schema':'story-package/v1','scenes':[{'node_id':'scene-2','lines':[
        {'node_id':'action-1','kind':'action','text':'推开门。'},
        {'node_id':'dialogue-2','kind':'dialogue','speaker':'story-package/character-1','text':'到了。'}]}]}
    shot = ProjectManifest.from_story_package('project',package).shots[0]
    assert shot.scene_context['source_spans'] == ['story-package/scene-2','story-package/scene-2/action-1','story-package/scene-2/dialogue-2']


def test_action_beats_split_shots_without_losing_dialogue_or_source():
    package = {'schema':'story-package/v1', 'scenes':[{'node_id':'scene-1', 'location':'钟楼', 'lines':[
        {'node_id':'a1', 'kind':'action', 'text':'信使推开门。'},
        {'node_id':'d1', 'kind':'dialogue', 'text':'有人吗？'},
        {'node_id':'a2', 'kind':'action', 'text':'老人从楼梯上走下来。'},
        {'node_id':'d2', 'kind':'dialogue', 'text':'我等你很久了。'}]}]}
    manifest = ProjectManifest.from_story_package('p', package)
    assert [shot.text for shot in manifest.shots] == ['信使推开门。 有人吗？', '老人从楼梯上走下来。 我等你很久了。']
    assert [shot.scene_id for shot in manifest.shots] == ['scene-1', 'scene-1']
    assert manifest.shots[1].scene_context['source_spans'] == ['story-package/scene-1', 'story-package/scene-1/a2', 'story-package/scene-1/d2']
    assert all(shot.scene_context['location'] == '钟楼' for shot in manifest.shots)
    assert len(package['scenes'][0]['lines']) == 4
    assert manifest.story_package == package


def test_image_action_excludes_dialogue_but_story_preserves_it(tmp_path):
    from unittest.mock import Mock
    from story_media_orchestrator.providers import StoryImageProvider
    from story_media_orchestrator.registry import ArtifactRegistry
    package = {'schema':'story-package/v1', 'scenes':[{'node_id':'scene', 'lines':[
        {'kind':'action', 'text':'信使推开门。'},
        {'kind':'dialogue', 'text':'我从很远的地方来。'}]}]}
    manifest = ProjectManifest.from_story_package('p', package)
    shot = manifest.shots[0]
    assert '我从很远的地方来。' in shot.text
    store = ArtifactRegistry(tmp_path / 'artifacts')
    adapter = Mock()
    adapter.run.return_value = {'first_frame_ref':store.put_bytes(b'image')}
    StoryImageProvider(adapter, store).generate(shot, tmp_path / 'frame.png')
    scene = adapter.run.call_args.args[0]
    assert '信使推开门。' in scene['action']
    assert '我从很远的地方来。' not in scene['action']
    assert '我从很远的地方来。' in scene['summary']


def test_project_image_settings_survive_manifest_and_reach_original_workflow(tmp_path):
    from unittest.mock import Mock
    from story_media_orchestrator.providers import StoryImageProvider
    from story_media_orchestrator.registry import ArtifactRegistry
    settings = {'framing':'远景', 'mood':'安静', 'negative':'拼贴', 'continuity':'蓝色外套'}
    package = {'schema':'story-package/v1', 'scenes':[{'node_id':'s', 'summary':'站在门口', **settings}]}
    manifest = ProjectManifest.from_story_package('p', package)
    manifest.save(tmp_path / 'project.json')
    shot = ProjectManifest.load(tmp_path / 'project.json').shots[0]
    store = ArtifactRegistry(tmp_path / 'artifacts')
    adapter = Mock()
    adapter.run.return_value = {'first_frame_ref':store.put_bytes(b'image')}
    StoryImageProvider(adapter, store).generate(shot, tmp_path / 'frame.png')
    scene = adapter.run.call_args.args[0]
    assert {key:scene[key] for key in settings} == settings


def test_character_backstory_does_not_overflow_visual_prompt(monkeypatch):
    from pathlib import Path
    from story_media_orchestrator.providers import StoryImageProvider
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'story-image-agent'))
    from story_image_agent import ImagePromptWorkflow
    character = {'node_id':'c', 'name':'阿澄', 'appearance':'短发', 'clothing':'蓝色外套',
                 'fear':'害怕失去家人' * 100, 'secret':'未寄出的信' * 100}
    visual = StoryImageProvider._visual_characters([character])
    plan = ImagePromptWorkflow('p').build_production_plan({'characters':visual, 'action':'推开门'}, ['scene'], candidate_count=1)
    prompt = plan['candidates'][0]['prompt']
    assert '阿澄' in prompt and '蓝色外套' in prompt and '短发' in prompt
    assert '未寄出的信' not in prompt and '害怕失去家人' not in prompt
    assert len(prompt) < 500
    assert character['secret'] == '未寄出的信' * 100
