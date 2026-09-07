#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::{Arc, Mutex}, process::{Command, Stdio}};
use tauri::State;
mod project;
mod story_host;
use story_host::{story_provider_status, save_story_credential};
use project::project_workflow;
use uuid::Uuid;

#[derive(Clone, Default)] struct AppState { runs: Arc<Mutex<HashMap<String, MediaRun>>> }
#[derive(Clone, Serialize, Deserialize)] struct Settings { story_model:String, image_model:String, image_size:String, video_model:String, comfyui_url:String, sidecar_url:String }
#[derive(Clone, Serialize, Deserialize)] struct Credentials { dashscope:String, sidecar:String, capability:String }
#[derive(Clone, Serialize, Deserialize)] struct Stage { name:String, state:String, artifact:Option<String>, retryable:bool }
#[derive(Clone, Serialize, Deserialize)] struct MediaRun { run_id:String, status:String, stages:Vec<Stage>, result:Option<serde_json::Value>, error:Option<String> }

fn credential(name:&str)->Result<keyring::Entry,String>{ keyring::Entry::new("story-media-orchestrator", name).map_err(|e|e.to_string()) }

fn credential_updates(credentials: Credentials) -> Vec<(&'static str, String)> {
  [("dashscope", credentials.dashscope), ("sidecar", credentials.sidecar), ("capability", credentials.capability)]
    .into_iter().filter(|(_, value)| !value.is_empty()).collect()
}

#[tauri::command] fn save_settings(settings:Settings, credentials:Credentials)->Result<(),String>{
  let path=std::env::var("LOCALAPPDATA").unwrap_or_else(|_|".".into()); let dir=std::path::PathBuf::from(path).join("StoryMediaOrchestrator"); std::fs::create_dir_all(&dir).map_err(|e|e.to_string())?;
  write_settings_atomically(&dir.join("settings.json"), &serde_json::to_vec_pretty(&settings).map_err(|e|e.to_string())?)?;
  for (name, value) in credential_updates(credentials) {
    credential(name)?.set_password(&value).map_err(|e|e.to_string())?;
  }
  Ok(())
}

fn write_settings_atomically(path: &std::path::Path, bytes: &[u8]) -> Result<(), String> {
  use std::io::Write;
  let temporary = path.with_file_name(format!(".settings-{}.tmp", Uuid::new_v4()));
  let result = (|| -> std::io::Result<()> {
    let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&temporary, path)
  })();
  if result.is_err() { let _ = std::fs::remove_file(&temporary); }
  result.map_err(|e| e.to_string())
}

#[tauri::command]
fn video_service_settings(url: Option<String>) -> Result<String, String> {
  let dir = std::path::PathBuf::from(std::env::var("LOCALAPPDATA").map_err(|_| "缺少本地配置目录")?).join("StoryMediaOrchestrator");
  update_video_service_settings(&dir, url)
}

fn update_video_service_settings(dir: &std::path::Path, url: Option<String>) -> Result<String, String> {
  let path = dir.join("settings.json");
  let mut settings: serde_json::Value = if path.exists() {
    serde_json::from_slice(&std::fs::read(&path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?
  } else { serde_json::json!({}) };
  if let Some(url) = url {
    let url = url.trim();
    if !url.is_empty() {
      let parsed = tauri::Url::parse(url).map_err(|_| "服务地址格式错误")?;
      if !["http", "https"].contains(&parsed.scheme()) || parsed.host_str().is_none() || !parsed.username().is_empty() || parsed.password().is_some() || parsed.query().is_some() || parsed.fragment().is_some() {
        return Err("请填写不含密钥和查询参数的 HTTP(S) 服务地址".into());
      }
    }
    let object = settings.as_object_mut().ok_or("配置文件格式错误")?;
    object.insert("comfyui_url".into(), serde_json::Value::String(url.to_string()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    write_settings_atomically(&path, &serde_json::to_vec_pretty(&settings).map_err(|e| e.to_string())?)?;
  }
  Ok(settings.get("comfyui_url").and_then(|v| v.as_str()).unwrap_or("").to_string())
}

#[tauri::command] fn generate_sidecar_token()->String { Uuid::new_v4().to_string()+&Uuid::new_v4().to_string() }

#[tauri::command]
async fn run_media_stage(app: tauri::AppHandle, stage: String, input: String) -> Result<serde_json::Value, String> {
  let result = tauri::async_runtime::spawn_blocking(move || execute_media_stage(stage, input))
    .await.map_err(|e|e.to_string())??;
  if let Some(path) = result.get("output_file").and_then(|v|v.as_str()) {
    use tauri::Manager;
    let output = std::path::PathBuf::from(path).canonicalize().map_err(|e|e.to_string())?;
    let root = std::env::var_os("STORY_MEDIA_ARTIFACT_ROOT").map(std::path::PathBuf::from)
      .unwrap_or_else(||std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../.artifacts"))
      .join("previews").canonicalize().map_err(|e|e.to_string())?;
    if !output.starts_with(root) { return Err("video preview outside artifact registry".into()); }
    app.asset_protocol_scope().allow_file(output).map_err(|e|e.to_string())?;
  }
  Ok(result)
}

fn execute_media_stage(stage:String, input:String)->Result<serde_json::Value,String>{
  execute_connected_stage(stage, input, None)
}
fn execute_connected_stage(stage:String, input:String, connection:Option<(String,String)>)->Result<serde_json::Value,String>{
  let root=std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..\\..\\..");
  let mut payload:serde_json::Value=serde_json::from_str(&input).map_err(|e|e.to_string())?;
  payload.as_object_mut().ok_or("stage input must be object".to_string())?.insert("_stage".into(),serde_json::Value::String(stage));
  let mut command=Command::new("python"); command.env("PYTHONIOENCODING", "utf-8").env("PYTHONUTF8", "1"); command.current_dir(root).args(["-m","story_media_orchestrator.cli"]).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
  let settings_path = std::env::var_os("LOCALAPPDATA").map(std::path::PathBuf::from)
    .map(|p|p.join("StoryMediaOrchestrator/settings.json"));
  let settings:serde_json::Value = settings_path.and_then(|p|std::fs::read(p).ok())
    .and_then(|bytes|serde_json::from_slice(&bytes).ok()).unwrap_or_default();
  for (key, variable) in [("sidecar_url", "STORY_SIDECAR_URL"), ("comfyui_url", "MINIMAX_H3_COMFYUI_BASE_URL")] {
    if std::env::var_os(variable).is_none() {
      if let Some(value) = settings.get(key).and_then(|v|v.as_str()).filter(|v|!v.trim().is_empty()) {
        command.env(variable, value);
      }
    }
  }
  #[cfg(windows)] {
    use std::os::windows::process::CommandExt;
    command.creation_flags(0x08000000);
  }
  if let Ok(value)=credential("dashscope").and_then(|e|e.get_password().map_err(|e|e.to_string())){command.env("DASHSCOPE_API_KEY",value);}
  if let Ok(value)=credential("sidecar").and_then(|e|e.get_password().map_err(|e|e.to_string())){command.env("STORY_SIDECAR_TOKEN",value);}
  if let Some((url, token)) = connection {
    command.env("STORY_SIDECAR_URL",url).env("STORY_SIDECAR_TOKEN",token);
  }
  let mut child=command.spawn().map_err(|e|e.to_string())?;
  use std::io::Write; child.stdin.take().ok_or("stdin unavailable".to_string())?.write_all(&serde_json::to_vec(&payload).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
  let output=child.wait_with_output().map_err(|e|e.to_string())?;
  if !output.status.success(){return Err(String::from_utf8_lossy(&output.stderr).to_string());}
  serde_json::from_slice(&output.stdout).map_err(|e|e.to_string())
}

#[tauri::command] fn start_media_run(st:State<'_,AppState>, story_input:String)->Result<MediaRun,String>{
  let id=Uuid::new_v4().to_string(); let run=MediaRun{run_id:id.clone(),status:"running".into(),result:None,error:None,stages:vec![Stage{name:"故事生成".into(),state:"running".into(),artifact:None,retryable:true},Stage{name:"首帧/尾帧生图".into(),state:"queued".into(),artifact:None,retryable:true},Stage{name:"5 秒视频生成".into(),state:"queued".into(),artifact:None,retryable:true}]}; st.runs.lock().map_err(|_|"state lock poisoned".to_string())?.insert(id.clone(),run.clone());
  let runs=st.runs.clone(); let run_id=id.clone();
  tauri::async_runtime::spawn(async move {
    let result = tauri::async_runtime::spawn_blocking(move || execute_media_stage("all".into(), story_input)).await;
    if let Ok(mut map) = runs.lock() {
      if let Some(item) = map.get_mut(&run_id) {
        match result {
          Ok(Ok(output)) => {
            item.status = output.get("status").and_then(|v|v.as_str()).unwrap_or("failed").into();
            item.stages[0].state = if output.get("story").is_some() { "succeeded" } else { "failed" }.into();
            item.stages[1].artifact = output.pointer("/image_plan/first_frame_ref").and_then(|v|v.as_str()).map(str::to_string);
            item.stages[1].state = if item.stages[1].artifact.is_some() { "succeeded" } else { "failed" }.into();
            item.stages[2].artifact = output.pointer("/video_plan/execution/artifact_ref").and_then(|v|v.as_str()).map(str::to_string);
            item.stages[2].state = output.pointer("/video_plan/execution/state").and_then(|v|v.as_str()).unwrap_or("planned").into();
            item.result = Some(output);
          },
          failure => {
            item.status = "failed".into();
            item.error = Some(match failure { Ok(Err(e)) => e, Err(e) => e.to_string(), _ => unreachable!() });
            for stage in &mut item.stages { stage.state = "unknown".into(); }
          }
        }
      }
    }
  });
  Ok(run)
}

#[tauri::command] fn get_media_run(st:State<'_,AppState>, run_id:String)->Result<MediaRun,String>{
  let runs=st.runs.lock().map_err(|_|"state lock poisoned".to_string())?;
  runs.get(&run_id).cloned().ok_or_else(||"run not found".to_string())
}

fn story_registry() -> Result<story_runtime::GenrePackRegistry, String> {
  let root = std::env::var_os("STORY_CAMPAIGN_ROOT").map(std::path::PathBuf::from)
    .unwrap_or_else(|| std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../microcodex-short-drama-studio"));
  story_runtime::GenrePackRegistry::load(&root).map_err(|e|e.to_string())
}
#[tauri::command]
fn list_story_templates() -> Result<serde_json::Value, String> {
  serde_json::to_value(story_registry()?.options()).map_err(|e|e.to_string())
}
fn prepare_story_command(job: serde_json::Value) -> Result<serde_json::Value, String> {
  let job: story_core::StoryJob = serde_json::from_value(job).map_err(|e|e.to_string())?;
  job.validate().map_err(|e|e.to_string())?;
  let context = story_registry()?.resolve_job(&job).map_err(|e|e.to_string())?;
  Ok(serde_json::json!({"schema":"start-run-command/v1", "job":job, "genre_context":context}))
}
#[tauri::command]
fn validate_template_story(job: serde_json::Value) -> Result<(), String> {
  prepare_story_command(job).map(|_| ())
}
#[tauri::command]
async fn generate_template_story(state: State<'_, story_host::StoryHostState>, job: serde_json::Value) -> Result<serde_json::Value, String> {
  let command = prepare_story_command(job)?;
  let mut session = state.0.try_lock().map_err(|_|"故事任务正在运行，请等待当前任务完成")?;
  if session.is_none() {
    let repository = std::env::var_os("STORY_CAMPAIGN_ROOT").map(std::path::PathBuf::from)
      .unwrap_or_else(||std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../microcodex-short-drama-studio"));
    *session = Some(story_host::HostedStory::launch(&repository,
      command["job"]["budget"]["deadline_seconds"].as_u64().unwrap_or(900)).await?);
  }
  session.as_ref().unwrap().check_health().await?;
  let connection = session.as_ref().unwrap().connection();
  tauri::async_runtime::spawn_blocking(move || {
    execute_connected_stage("story".into(), command.to_string(), Some(connection))
  }).await.map_err(|e|e.to_string())?
}
#[tauri::command]
async fn choose_project_directory() -> Result<Option<String>, String> {
  tauri::async_runtime::spawn_blocking(|| rfd::FileDialog::new().set_title("选择项目文件夹").pick_folder().map(|p| p.to_string_lossy().into_owned())).await.map_err(|e| e.to_string())
}
fn main(){ tauri::Builder::default().manage(AppState::default()).manage(story_host::StoryHostState::default()).invoke_handler(tauri::generate_handler![video_service_settings,save_story_credential,story_provider_status,validate_template_story,list_story_templates,generate_template_story,choose_project_directory,save_settings,generate_sidecar_token,project_workflow,run_media_stage,start_media_run,get_media_run]).run(tauri::generate_context!()).expect("error while running tauri application"); }

#[cfg(test)]
mod story_template_tests {
  use super::*;
  fn job() -> serde_json::Value {
    serde_json::json!({"schema":"story-job/v1","job_id":"job_test_templates","content_form":"scripted_short_drama","input":"维修工和父亲在商场修复电梯","genre_mode":"fixed","allowed_genres":["family"],"genre_pack_id":"family-grounded-v1","constraint_profile_id":"short-vertical-v1","audience":"25-45","format":{"episodes":6,"minutes_per_episode":2},"content_limits":[],"budget":{"max_tokens":180000,"max_cny_fen":1200,"deadline_seconds":900}})
  }
  #[test]
  fn original_registry_resolves_both_profiles_and_rejects_mismatch() {
    let mut input = job();
    let command = prepare_story_command(input.clone()).unwrap();
    assert_eq!(command["genre_context"]["pack_id"], "family-grounded-v1");
    assert!(command["genre_context"]["architect_directives"].as_array().unwrap().len() > 0);
    input["constraint_profile_id"] = "long-serial-v1".into();
    assert!(prepare_story_command(input.clone()).is_err());
    input["format"]["episodes"] = 40.into();
    assert!(prepare_story_command(input.clone()).is_ok());
    input["genre_pack_id"] = "missing-template".into();
    assert!(prepare_story_command(input).is_err());
  }
  #[test]
  fn blank_story_is_rejected_before_generation() {
    let mut input = job(); input["input"] = " ".into();
    assert!(prepare_story_command(input).is_err());
  }
}

#[cfg(test)]
mod credential_update_tests {
  use super::*;
  #[test]
  fn blank_fields_preserve_all_existing_credentials() {
    assert!(credential_updates(Credentials{dashscope:String::new(),sidecar:String::new(),capability:String::new()}).is_empty());
  }
  #[test]
  fn only_explicit_credential_is_changed() {
    let updates = credential_updates(Credentials{dashscope:"test-only".into(),sidecar:String::new(),capability:String::new()});
    assert_eq!(updates, vec![("dashscope", "test-only".to_string())]);
  }
}


#[cfg(test)]
mod video_service_settings_tests {
  use super::*;
  #[test]
  fn saves_only_video_address_and_reloads_it() {
    let dir = std::env::temp_dir().join(format!("story-settings-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("settings.json");
    std::fs::write(&file, r#"{"image_model":"existing","sidecar_url":"http://story","custom":{"keep":true}}"#).unwrap();
    update_video_service_settings(&dir, Some("http://127.0.0.1:8188".into())).unwrap();
    assert_eq!(update_video_service_settings(&dir, None).unwrap(), "http://127.0.0.1:8188");
    let saved: serde_json::Value = serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
    assert_eq!(saved["image_model"], "existing");
    assert_eq!(saved["sidecar_url"], "http://story");
    assert_eq!(saved["custom"]["keep"], true);
    std::fs::remove_file(file).unwrap();
    std::fs::remove_dir(dir).unwrap();
  }
  #[test]
  fn invalid_address_does_not_modify_settings() {
    let dir = std::env::temp_dir().join(format!("story-settings-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("settings.json");
    let original = br#"{"image_model":"existing"}"#;
    std::fs::write(&file, original).unwrap();
    for value in ["file:///tmp/service", "http://user:secret@localhost", "https://localhost?token=secret", "not-url"] {
      assert!(update_video_service_settings(&dir, Some(value.into())).is_err());
      assert_eq!(std::fs::read(&file).unwrap(), original);
    }
    std::fs::remove_file(file).unwrap();
    std::fs::remove_dir(dir).unwrap();
  }
}


#[cfg(test)]
mod live_story_acceptance {
  #[test]
  #[ignore = "explicit live model acceptance; requires STORY_ACCEPTANCE_REQUEST and may incur model charges"]
  fn original_template_generates_real_story() {
    let request = std::path::PathBuf::from(std::env::var("STORY_ACCEPTANCE_REQUEST").expect("explicit acceptance request required"));
    let output = request.with_file_name("story-result.json");
    if output.exists() { panic!("result already exists; inspect it instead of generating again"); }
    let job = serde_json::from_slice(&std::fs::read(&request).unwrap()).unwrap();
    let command = super::prepare_story_command(job).expect("template validation");
    tauri::async_runtime::block_on(async {
      let repository = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../microcodex-short-drama-studio");
      let host = crate::story_host::HostedStory::launch(&repository, 900).await.expect("original host startup");
      let connection = host.connection();
      let result = tauri::async_runtime::spawn_blocking(move || super::execute_connected_stage("story".into(), command.to_string(), Some(connection))).await.unwrap();
      match result {
        Ok(value) => { std::fs::write(&output, serde_json::to_vec_pretty(&value).unwrap()).unwrap(); },
        Err(error) => { std::fs::write(request.with_file_name("story-error.txt"), &error).unwrap(); panic!("story generation failed; see story-error.txt"); }
      }
    });
  }
}
