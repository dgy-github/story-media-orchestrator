from story_media_orchestrator.registry import ArtifactRegistry
from story_media_orchestrator.video_result import persist_video_result


def test_video_record_preserves_lineage_and_parameters_without_credentials(tmp_path):
    store = ArtifactRegistry(tmp_path)
    result = {'execution': {'state':'succeeded', 'task_id':'task'}, 'conditioning_mode':'text_to_video'}
    request = {'video_provider':'dashscope', 'prompt':'walk', 'source_spans':['story-package/scene2'],
               'first_frame_ref':'artifact://sha256/image', 'last_frame_ref':'artifact://sha256/image',
               'video_size':'720*1280', 'duration_seconds':10, 'batch_id':'batch', 'api_key':'secret'}
    saved = persist_video_result(result, request, store)
    record = store.get_json(saved['result_record_ref'])
    assert record['source_spans'] == ['story-package/scene2']
    assert record['associated_image_refs'] == ['artifact://sha256/image']
    assert record['generation_parameters']['duration_seconds'] == 10
    assert record['generation_parameters']['video_model'] == 'wanx2.1-t2v-turbo'
    assert 'secret' not in str(record)
    assert result == {'execution': {'state':'succeeded', 'task_id':'task'}, 'conditioning_mode':'text_to_video'}


def test_legacy_comfy_input_records_original_defaults(tmp_path):
    saved = persist_video_result({}, {'prompt':'walk'}, ArtifactRegistry(tmp_path))
    parameters = saved['generation_parameters']
    assert parameters['video_provider'] == 'comfyui'
    assert parameters['width'] == 864 and parameters['length'] == 124
