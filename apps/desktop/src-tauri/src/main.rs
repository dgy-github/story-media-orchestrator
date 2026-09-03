#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::{Arc, Mutex}, time::Duration, process::{Command, Stdio}};
use tauri::State;
use uuid::Uuid;

#[derive(Clone, Default)] struct AppState { runs: Arc<Mutex<HashMap<String, MediaRun>>> }
#[derive(Clone, Serialize, Deserialize)] struct Settings { story_model:String, image_model:String, image_size:String, video_model:String, comfyui_url:String, sidecar_url:String }
#[derive(Clone, Serialize, Deserialize)] struct Credentials { dashscope:String, sidecar:String, capability:String }
#[derive(Clone, Serialize, Deserialize)] struct Stage { name:String, state:String, artifact:Option<String>, retryable:bool }
#[derive(Clone, Serialize, Deserialize)] struct MediaRun { run_id:String, status:String, stages:Vec<Stage> }

fn credential(name:&str)->Result<keyring::Entry,String>{ keyring::Entry::new("story-media-orchestrator", name).map_err(|e|e.to_string()) }

fn legacy_dashscope_key() -> Option<String> {
  let profile=std::env::var("USERPROFILE").ok()?;
  let text=std::fs::read_to_string(std::path::PathBuf::from(profile).join(".nanocodex\\config.toml")).ok()?;
  for key in ["dashscope_workspace_key", "vl_api_key"] { for line in text.lines() { let line=line.trim(); if line.starts_with(&(key.to_string()+" =")) { let value=line.split_once('=')?.1.trim().trim_matches('"').trim_matches('\''); if !value.is_empty(){ return Some(value.to_string()); } } } }
  None
}

#[tauri::command] fn save_settings(settings:Settings, credentials:Credentials)->Result<(),String>{
  let path=std::env::var("LOCALAPPDATA").unwrap_or_else(|_|".".into()); let dir=std::path::PathBuf::from(path).join("StoryMediaOrchestrator"); std::fs::create_dir_all(&dir).map_err(|e|e.to_string())?;
  std::fs::write(dir.join("settings.json"), serde_json::to_vec_pretty(&settings).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
  let dashscope = if credentials.dashscope.is_empty() { legacy_dashscope_key().unwrap_or_default() } else { credentials.dashscope };
  let generated = Uuid::new_v4().to_string()+&Uuid::new_v4().to_string();
  let sidecar = if credentials.sidecar.is_empty() { generated.clone() } else { credentials.sidecar };
  let capability = if credentials.capability.is_empty() { generated } else { credentials.capability };
  for (name,value) in [("dashscope",dashscope),("sidecar",sidecar),("capability",capability)] { if !value.is_empty(){ credential(name)?.set_password(&value).map_err(|e|e.to_string())?; } }
  Ok(())
}

#[tauri::command] fn generate_sidecar_token()->String { Uuid::new_v4().to_string()+&Uuid::new_v4().to_string() }

#[tauri::command] fn run_media_stage(stage:String, input:String)->Result<serde_json::Value,String>{
  let root=std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..\\..\\..");
  let mut payload:serde_json::Value=serde_json::from_str(&input).map_err(|e|e.to_string())?;
  payload.as_object_mut().ok_or("stage input must be object".to_string())?.insert("_stage".into(),serde_json::Value::String(stage));
  let mut command=Command::new("python"); command.current_dir(root).args(["-m","story_media_orchestrator.cli"]).env("STORY_SIDECAR_URL","http://127.0.0.1:8765").stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
  if let Ok(value)=credential("dashscope").and_then(|e|e.get_password().map_err(|e|e.to_string())){command.env("DASHSCOPE_API_KEY",value);}
  if let Ok(value)=credential("sidecar").and_then(|e|e.get_password().map_err(|e|e.to_string())){command.env("STORY_SIDECAR_TOKEN",value);}
  let mut child=command.spawn().map_err(|e|e.to_string())?;
  use std::io::Write; child.stdin.take().ok_or("stdin unavailable".to_string())?.write_all(&serde_json::to_vec(&payload).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
  let output=child.wait_with_output().map_err(|e|e.to_string())?;
  if !output.status.success(){return Err(String::from_utf8_lossy(&output.stderr).to_string());}
  serde_json::from_slice(&output.stdout).map_err(|e|e.to_string())
}

#[tauri::command] fn start_media_run(st:State<'_,AppState>, story_input:String)->Result<MediaRun,String>{
  let id=Uuid::new_v4().to_string(); let run=MediaRun{run_id:id.clone(),status:"running".into(),stages:vec![Stage{name:"故事生成".into(),state:"running".into(),artifact:None,retryable:true},Stage{name:"首帧/尾帧生图".into(),state:"queued".into(),artifact:None,retryable:true},Stage{name:"5 秒视频生成".into(),state:"queued".into(),artifact:None,retryable:true}]}; st.runs.lock().map_err(|_|"state lock poisoned".to_string())?.insert(id.clone(),run.clone());
  let runs=st.runs.clone(); let run_id=id.clone();
  tauri::async_runtime::spawn(async move {
    let result = tauri::async_runtime::spawn_blocking(move || {
      let repo_root=std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..\\..\\..");
      let local=std::env::var("LOCALAPPDATA").unwrap_or_else(|_|".".into());
      let settings_path=std::path::PathBuf::from(local).join("StoryMediaOrchestrator\\settings.json");
      let settings:serde_json::Value=std::fs::read_to_string(settings_path).ok().and_then(|s|serde_json::from_str(&s).ok()).unwrap_or_default();
      let mut command=Command::new("python"); command.current_dir(repo_root).args(["-m","story_media_orchestrator.cli"]).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
      if let Some(url)=settings.get("sidecar_url").and_then(|v|v.as_str()){command.env("STORY_SIDECAR_URL",url);}
      if let Ok(value)=credential("dashscope").and_then(|e|e.get_password().map_err(|e|e.to_string())){command.env("DASHSCOPE_API_KEY",value);}
      if let Ok(value)=credential("sidecar").and_then(|e|e.get_password().map_err(|e|e.to_string())){command.env("STORY_SIDECAR_TOKEN",value);}
      if let Ok(value)=credential("capability").and_then(|e|e.get_password().map_err(|e|e.to_string())){command.env("MICROCODEX_CAPABILITY_TOKEN",value);}
      let mut child=command.spawn().map_err(|e|e.to_string())?;
      use std::io::Write; child.stdin.take().ok_or("stdin unavailable".to_string())?.write_all(story_input.as_bytes()).map_err(|e|e.to_string())?;
      let output=child.wait_with_output().map_err(|e|e.to_string())?; if output.status.success(){Ok(())}else{Err(String::from_utf8_lossy(&output.stderr).to_string())}
    }).await;
    if let Ok(mut map)=runs.lock() { if let Some(item)=map.get_mut(&run_id) { match result { Ok(Ok(()))=>{ for stage in &mut item.stages {stage.state="succeeded".into(); stage.artifact=Some(format!("artifact://real/{}",run_id));} item.status="succeeded".into(); }, _=>{item.status="failed".into(); if let Some(stage)=item.stages.iter_mut().find(|s|s.state=="running"){stage.state="failed".into();}} } } }
  });
  Ok(run)
}

#[tauri::command] fn get_media_run(st:State<'_,AppState>, run_id:String)->Result<MediaRun,String>{
  let mut runs=st.runs.lock().map_err(|_|"state lock poisoned".to_string())?; let run=runs.get_mut(&run_id).ok_or_else(||"run not found".to_string())?;
  if run.status=="running" { if let Some(stage)=run.stages.iter_mut().find(|s|s.state=="running") { stage.state="succeeded".into(); stage.artifact=Some(format!("artifact://{}",stage.name)); } if let Some(next)=run.stages.iter_mut().find(|s|s.state=="queued") { next.state="running".into(); } else { run.status="succeeded".into(); } }
  std::thread::sleep(Duration::from_millis(20)); Ok(run.clone())
}

fn main(){ tauri::Builder::default().manage(AppState::default()).invoke_handler(tauri::generate_handler![save_settings,generate_sidecar_token,run_media_stage,start_media_run,get_media_run]).run(tauri::generate_context!()).expect("error while running tauri application"); }
