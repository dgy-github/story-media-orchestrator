#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Mutex, time::Duration};
use tauri::State;
use uuid::Uuid;

#[derive(Default)] struct AppState { runs: Mutex<HashMap<String, MediaRun>> }
#[derive(Clone, Serialize, Deserialize)] struct Settings { story_model:String, image_model:String, image_size:String, video_model:String, comfyui_url:String, sidecar_url:String }
#[derive(Clone, Serialize, Deserialize)] struct Credentials { dashscope:String, sidecar:String, capability:String }
#[derive(Clone, Serialize, Deserialize)] struct Stage { name:String, state:String, artifact:Option<String>, retryable:bool }
#[derive(Clone, Serialize, Deserialize)] struct MediaRun { run_id:String, status:String, stages:Vec<Stage> }

fn credential(name:&str)->Result<keyring::Entry,String>{ keyring::Entry::new("story-media-orchestrator", name).map_err(|e|e.to_string()) }

#[tauri::command] fn save_settings(settings:Settings, credentials:Credentials)->Result<(),String>{
  let path=std::env::var("LOCALAPPDATA").unwrap_or_else(|_|".".into()); let dir=std::path::PathBuf::from(path).join("StoryMediaOrchestrator"); std::fs::create_dir_all(&dir).map_err(|e|e.to_string())?;
  std::fs::write(dir.join("settings.json"), serde_json::to_vec_pretty(&settings).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
  for (name,value) in [("dashscope",credentials.dashscope),("sidecar",credentials.sidecar),("capability",credentials.capability)] { if !value.is_empty(){ credential(name)?.set_password(&value).map_err(|e|e.to_string())?; } }
  Ok(())
}

#[tauri::command] fn generate_sidecar_token()->String { Uuid::new_v4().to_string()+&Uuid::new_v4().to_string() }

#[tauri::command] fn start_media_run(st:State<'_,AppState>, story_input:String)->Result<MediaRun,String>{
  let _=story_input; let id=Uuid::new_v4().to_string(); let run=MediaRun{run_id:id.clone(),status:"running".into(),stages:vec![Stage{name:"故事生成".into(),state:"running".into(),artifact:None,retryable:true},Stage{name:"首帧/尾帧生图".into(),state:"queued".into(),artifact:None,retryable:true},Stage{name:"5 秒视频生成".into(),state:"queued".into(),artifact:None,retryable:true}]}; st.runs.lock().map_err(|_|"state lock poisoned".to_string())?.insert(id,run.clone()); Ok(run)
}

#[tauri::command] fn get_media_run(st:State<'_,AppState>, run_id:String)->Result<MediaRun,String>{
  let mut runs=st.runs.lock().map_err(|_|"state lock poisoned".to_string())?; let run=runs.get_mut(&run_id).ok_or_else(||"run not found".to_string())?;
  if run.status=="running" { if let Some(stage)=run.stages.iter_mut().find(|s|s.state=="running") { stage.state="succeeded".into(); stage.artifact=Some(format!("artifact://{}",stage.name)); } if let Some(next)=run.stages.iter_mut().find(|s|s.state=="queued") { next.state="running".into(); } else { run.status="succeeded".into(); } }
  std::thread::sleep(Duration::from_millis(20)); Ok(run.clone())
}

fn main(){ tauri::Builder::default().manage(AppState::default()).invoke_handler(tauri::generate_handler![save_settings,generate_sidecar_token,start_media_run,get_media_run]).run(tauri::generate_context!()).expect("error while running tauri application"); }
