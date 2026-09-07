use std::{path::{Path, PathBuf}, process::{Command, Stdio}};
use serde_json::{json, Value};
use tauri::Manager;

fn project_path(root: &Path, project: &str) -> Result<PathBuf, String> {
    if project.trim().is_empty() { return Err("项目目录不能为空".into()); }
    let path = PathBuf::from(project);
    Ok(if path.is_absolute() { path } else { root.join(path) })
}

fn execute(root: &Path, command: &str, project: &str, story: Option<String>,
           shot: Option<String>, mode: Option<String>, image_source: Option<String>,
           image_model: Option<String>, image_size: Option<String>, story_package: Option<Value>, story_job: Option<Value>, video_provider: Option<String>, video_options: Option<Value>, tts_source: Option<String>, auto_run: Option<bool>, edit_options: Option<Value>) -> Result<Value, String> {
    if !["create", "load", "run", "resume", "retry", "review", "candidates", "select", "effects"].contains(&command) {
        return Err("unsupported project command".into());
    }
    let path = project_path(root, project)?;
    let mut error = None;
    if command != "load" {
        let mut process = Command::new("python");
        process.env("PYTHONIOENCODING", "utf-8").env("PYTHONUTF8", "1");
        if std::env::var_os("MINIMAX_H3_COMFYUI_BASE_URL").is_none() {
            let settings = std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
                .and_then(|p| std::fs::read(p.join("StoryMediaOrchestrator/settings.json")).ok())
                .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());
            if let Some(url) = settings.as_ref().and_then(|v| v.get("comfyui_url")).and_then(Value::as_str).filter(|v| !v.trim().is_empty()) {
                process.env("MINIMAX_H3_COMFYUI_BASE_URL", url);
            }
        }
        process.current_dir(root).args(["-m", "story_media_orchestrator.cli", command]).arg(&path);
        if command == "create" { process.arg("--create-stdin").stdin(Stdio::piped()); }
        if ["retry", "review", "candidates", "select", "effects"].contains(&command) { process.arg(shot.ok_or("shot id is required")?); }
        if ["candidates", "select", "effects"].contains(&command) {
            if let Some(options) = edit_options {
                for (key, flag) in [("candidate_count", "--candidate-count"), ("candidate_id", "--candidate-id"), ("frame_role", "--frame-role"), ("motion", "--motion"), ("transition", "--transition")] {
                    if let Some(value) = options.get(key) { process.arg(flag).arg(value.as_str().map(str::to_owned).unwrap_or_else(||value.to_string())); }
                }
            }
        }
        if ["run", "resume"].contains(&command) {
            if let Some(value) = mode { process.args(["--mode", &value]); }
        }
        if ["run", "resume", "retry"].contains(&command) {
            process.arg("--real-preview");
            process.arg(if auto_run.unwrap_or(false) { "--auto" } else { "--require-review" });
            if let Some(value) = tts_source { process.args(["--tts-source", &value]); }
            if let Some(value) = video_provider { process.args(["--video-provider", &value]); }
            if let Some(value) = video_options { process.args(["--video-options", &value.to_string()]); }
            for (flag, value) in [("--image-source", image_source), ("--image-model", image_model), ("--image-size", image_size)] {
                if let Some(value) = value.filter(|v| !v.trim().is_empty()) { process.args([flag, &value]); }
            }
        }
        if ["run", "resume", "retry", "candidates"].contains(&command) {
            if std::env::var_os("DASHSCOPE_API_KEY").is_none() && std::env::var_os("DASHSCOPE_WORKSPACE_KEY").is_none() {
                if let Ok(value) = crate::credential("dashscope").and_then(|e| e.get_password().map_err(|e| e.to_string())) {
                    process.env("DASHSCOPE_API_KEY", value);
                }
            }
        }
        #[cfg(windows)] {
            use std::os::windows::process::CommandExt;
            process.creation_flags(0x08000000);
        }
        let output = if command == "create" {
            use std::io::Write;
            let mut child = process.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().map_err(|e|e.to_string())?;
            let payload = json!({"story":story.ok_or("story is required")?, "story_package":story_package, "story_job":story_job});
            child.stdin.take().ok_or("stdin unavailable")?.write_all(payload.to_string().as_bytes()).map_err(|e|e.to_string())?;
            child.wait_with_output().map_err(|e|e.to_string())?
        } else { process.output().map_err(|e| e.to_string())? };
        if !output.status.success() {
            error = Some(String::from_utf8_lossy(&output.stderr).trim().to_string());
        }
    }
    let bytes = std::fs::read(path.join("project.json"))
        .map_err(|e| error.clone().unwrap_or_else(|| e.to_string()))?;
    let manifest: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let canonical = path.canonicalize().map_err(|e| e.to_string())?;
    let output = manifest.get("output").and_then(Value::as_str)
        .and_then(|p| PathBuf::from(p).canonicalize().ok())
        .filter(|p| p.starts_with(&canonical) && p.extension().is_some_and(|e| e == "mp4"));
    Ok(json!({"manifest": manifest, "error": error, "project": canonical, "output": output}))
}

#[tauri::command]
pub async fn project_workflow(app: tauri::AppHandle, command: String, project: String,
    story: Option<String>, shot: Option<String>, mode: Option<String>, image_source: Option<String>,
    image_model: Option<String>, image_size: Option<String>, story_package: Option<Value>, story_job: Option<Value>, video_provider: Option<String>, video_options: Option<Value>, tts_source: Option<String>, auto_run: Option<bool>, edit_options: Option<Value>) -> Result<Value, String> {
    let result = tauri::async_runtime::spawn_blocking(move || {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
        execute(&root, &command, &project, story, shot, mode, image_source, image_model, image_size, story_package, story_job, video_provider, video_options, tts_source, auto_run, edit_options)
    }).await.map_err(|e| e.to_string())??;
    if let Some(output) = result.get("output").and_then(Value::as_str) {
        app.asset_protocol_scope().allow_file(output).map_err(|e| e.to_string())?;
    }
    if let Some(shots) = result.pointer("/manifest/shots").and_then(Value::as_array) {
        let root = result.get("project").and_then(Value::as_str).map(PathBuf::from).and_then(|p|p.canonicalize().ok());
        for shot in shots {
            if let Some(candidates) = shot.get("candidates").and_then(Value::as_array) {
                for candidate in candidates {
                    if let Some(path) = candidate.get("path").and_then(Value::as_str).and_then(|p|PathBuf::from(p).canonicalize().ok()) {
                        if root.as_ref().is_some_and(|r|path.starts_with(r)) { app.asset_protocol_scope().allow_file(path).map_err(|e|e.to_string())?; }
                    }
                }
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn relative_project_is_resolved_against_repository() {
        assert_eq!(project_path(Path::new("repo"), "demo").unwrap(), Path::new("repo/demo"));
        assert!(project_path(Path::new("repo"), " ").is_err());
    }
}
