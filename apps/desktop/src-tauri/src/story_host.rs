//! Read the original desktop's append-only route records without copying secrets.
use std::path::Path;
use serde_json::{json, Value};
use story_provider::{ProviderRoute, ProviderCredentialId, ProviderCredentialStore, WindowsCredentialStore};

pub fn route_settings(root: &Path, provider: &str) -> Result<Value, String> {
    let (endpoint, model) = match provider {
        "deepseek" => ("https://api.deepseek.com/chat/completions", "deepseek-v4-pro"),
        "aliyun_bailian" => ("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", "qwen3-vl-plus"),
        _ => return Err("unsupported story provider".into()),
    };
    let mut latest: Option<Value> = None;
    let directory = root.join(provider);
    if directory.exists() {
        for entry in std::fs::read_dir(directory).map_err(|_|"无法读取原故事模型配置")? {
            let path = entry.map_err(|_|"无法读取模型记录")?.path();
            let name = path.file_name().and_then(|v|v.to_str()).unwrap_or("");
            if !name.starts_with("route_") || !name.ends_with(".json") { continue; }
            let record: Value = serde_json::from_slice(&std::fs::read(&path).map_err(|_|"无法读取模型记录")?).map_err(|_|"模型记录格式错误")?;
            if record["schema"] != "desktop-provider-route/v1" || record["provider"] != provider
                || record["profile"] != "default" || record["source"] != "user"
                || record["thinking_disabled"] != (provider == "aliyun_bailian")
                || record["updated_at_unix_ms"].as_u64().is_none()
                || record["record_id"].as_str().map(|id|format!("{id}.json")) != Some(name.to_string()) {
                return Err("原故事模型记录校验失败".into());
            }
            ProviderRoute::validate(record["endpoint"].as_str().ok_or("缺少模型地址")?, record["model"].as_str().ok_or("缺少模型名称")?).map_err(|_|"原故事模型地址无效")?;
            let key = |v: &Value| (v["updated_at_unix_ms"].as_u64().unwrap_or(0), v["record_id"].as_str().unwrap_or("").to_string());
            if latest.as_ref().map(|v|key(&record)>key(v)).unwrap_or(true) { latest = Some(record); }
        }
    }
    Ok(latest.unwrap_or_else(||json!({"provider":provider,"endpoint":endpoint,"model":model,"thinking_disabled":provider=="aliyun_bailian","source":"default"})))
}

#[tauri::command]
pub async fn story_provider_status(state: tauri::State<'_, StoryHostState>) -> Result<Value, String> {
    let root = std::env::var_os("LOCALAPPDATA").map(std::path::PathBuf::from).unwrap_or_else(std::env::temp_dir)
        .join("MicrocodeX/ShortDramaStudio/provider-routes");
    let store = WindowsCredentialStore::new();
    let mut routes = Vec::new();
    for provider in ["deepseek", "aliyun_bailian"] {
        let mut route = route_settings(&root, provider)?;
        let id = ProviderCredentialId::new(provider, "default").map_err(|_|"凭据标识错误")?;
        route["configured"] = store.get(&id).map_err(|_|"无法读取原故事凭据存储")?.is_some().into();
        routes.push(route);
    }
    let session = state.0.try_lock();
    let connected = match session.as_ref() {
        Ok(guard) => match guard.as_ref() { Some(host) => host.sidecar.health().await.is_ok(), None => false },
        Err(_) => false,
    };
    Ok(json!({"routes":routes,"host_connected":connected,"busy":session.is_err()}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reads_latest_original_record_and_rejects_invalid_route() {
        let root = std::env::temp_dir().join(format!("story-route-test-{}", uuid::Uuid::new_v4()));
        let dir = root.join("deepseek"); std::fs::create_dir_all(&dir).unwrap();
        let record = |id: &str, time: u64, model: &str| json!({"schema":"desktop-provider-route/v1","provider":"deepseek","profile":"default","source":"user","thinking_disabled":false,"record_id":id,"updated_at_unix_ms":time,"endpoint":"https://api.deepseek.com/chat/completions","model":model});
        std::fs::write(dir.join("route_a.json"), record("route_a", 1, "older").to_string()).unwrap();
        std::fs::write(dir.join("route_b.json"), record("route_b", 2, "newer").to_string()).unwrap();
        assert_eq!(route_settings(&root,"deepseek").unwrap()["model"], "newer");
        let mut bad = record("route_c",3,"bad"); bad["endpoint"] = "http://invalid".into();
        std::fs::write(dir.join("route_c.json"),bad.to_string()).unwrap();
        assert!(route_settings(&root,"deepseek").is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}

pub struct HostedStory {
    _host: story_provider::CapabilityHost,
    #[cfg(test)]
    capability_token: story_provider::CapabilityToken,
    sidecar: story_runtime::SidecarProcess,
    token: story_provider::CapabilityToken,
}
#[derive(Default)]
pub struct StoryHostState(pub tokio::sync::Mutex<Option<HostedStory>>);

impl HostedStory {
    pub async fn launch(repository: &Path, deadline: u64) -> Result<Self, String> {
        use story_provider::{CapabilityHost, CapabilityHostConfig, CapabilityToken, PricingCatalog};
        use story_runtime::{SidecarAuthToken, SidecarLaunchConfig, SidecarProcess};
        use std::time::Duration;
        let local = std::env::var_os("LOCALAPPDATA").map(std::path::PathBuf::from).unwrap_or_else(std::env::temp_dir);
        let routes = local.join("MicrocodeX/ShortDramaStudio/provider-routes");
        let store = WindowsCredentialStore::new();
        let route = |provider: &str| -> Result<ProviderRoute, String> {
            let settings = route_settings(&routes, provider)?;
            let id = ProviderCredentialId::new(provider,"default").map_err(|_|"invalid provider")?;
            let secret = store.get(&id).map_err(|_|"无法读取原故事凭据")?
                .ok_or_else(||format!("缺少 {provider} 凭据，请在原故事项目中配置"))?;
            let route = ProviderRoute::new(settings["endpoint"].as_str().ok_or("missing endpoint")?, settings["model"].as_str().ok_or("missing model")?, secret).map_err(|e|e.to_string())?;
            Ok(if provider == "aliyun_bailian" { route.with_thinking_disabled() } else { route })
        };
        let generation = route("deepseek")?;
        let review = route("aliyun_bailian")?;
        let pricing = PricingCatalog::from_json(&std::fs::read_to_string(repository.join("config/provider-pricing-v1.json")).map_err(|_|"无法读取原模型价格表")?).map_err(|e|e.to_string())?;
        let material = || format!("{}{}", uuid::Uuid::new_v4().simple(), uuid::Uuid::new_v4().simple());
        let capability_token = CapabilityToken::new(material()).map_err(|e|e.to_string())?;
        let host = CapabilityHost::start(CapabilityHostConfig {
            generation, review, pricing,
            package_schema_path: repository.join("schemas/story-package-v1.json"),
            retained_store_root: local.join("StoryMediaOrchestrator/story/retained"),
            media_project_store_root: local.join("StoryMediaOrchestrator/story/media-projects"),
            token: CapabilityToken::new(capability_token.expose().to_string()).map_err(|e|e.to_string())?,
            request_timeout: Duration::from_secs(deadline),
        }).await.map_err(|e|e.to_string())?;
        let python = std::env::var_os("MICROCODEX_PYTHON").map(std::path::PathBuf::from)
            .unwrap_or_else(||repository.join(".venv/Scripts/python.exe"));
        let config = SidecarLaunchConfig::new(python,repository.join("sidecar"),Duration::from_secs(15)).map_err(|e|e.to_string())?;
        let token = CapabilityToken::new(material()).map_err(|e|e.to_string())?;
        let sidecar = SidecarProcess::launch_with_capability(config,
            SidecarAuthToken::new(token.expose().to_string()).map_err(|e|e.to_string())?,
            &host.endpoint(), &capability_token).await.map_err(|e|e.to_string())?;
        Ok(Self { _host: host, sidecar, token, #[cfg(test)] capability_token })
    }
    pub async fn check_health(&self) -> Result<(), String> {
        self.sidecar.health().await.map_err(|_|"故事服务暂不可达；保留原任务，请检查服务后再恢复，避免重复提交".into())
    }
    pub fn connection(&self) -> (String, String) {
        (self.sidecar.base_url().to_string(), self.token.expose().to_string())
    }
}

#[cfg(test)]
mod local_host_smoke {
    #[test]
    #[ignore = "requires the original local story runtime and Windows credentials; no model requests"]
    fn original_host_starts_without_generation() {
        let repository = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../microcodex-short-drama-studio");
        tauri::async_runtime::block_on(async {
            let host = super::HostedStory::launch(&repository, 900).await.expect("original host startup");
            host.check_health().await.expect("original sidecar health");
        });
    }
}

#[tauri::command]
pub async fn save_story_credential(state: tauri::State<'_, StoryHostState>, provider: String, secret: String) -> Result<(), String> {
    let _guard = state.0.try_lock().map_err(|_|"故事任务运行中，请完成后再修改凭据")?;
    let secret = story_provider::ProviderSecret::new(secret.into_bytes()).map_err(|_|"密钥不能为空")?;
    if !["deepseek", "aliyun_bailian"].contains(&provider.as_str()) {
        return Err("不支持的故事模型通道".into());
    }
    let bytes = secret.expose_secret();
    if bytes.len() > 4096 || bytes.iter().any(|b| b.is_ascii_whitespace() || b.is_ascii_control()) {
        return Err("密钥包含空白字符或长度无效".into());
    }
    let id = ProviderCredentialId::new(provider, "default").map_err(|_|"凭据标识无效")?;
    WindowsCredentialStore::new().set(&id, bytes).map_err(|_|"无法保存到 Windows 凭据存储")?;
    Ok(())
}

#[cfg(test)]
mod live_recovery_acceptance {
    #[test]
    #[ignore = "explicit recovery acceptance; calls real providers using retained run budget"]
    fn resume_retained_story_acceptance() {
        let script = std::env::var("STORY_RECOVERY_SCRIPT").expect("explicit script required");
        let repository = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../microcodex-short-drama-studio");
        tauri::async_runtime::block_on(async {
            let host = super::HostedStory::launch(&repository, 900).await.unwrap();
            let status = tokio::process::Command::new(repository.join(".venv/Scripts/python.exe"))
                .args(["-X", "utf8", &script])
                .env("STORY_RECOVERY_CAPABILITY_URL", host._host.endpoint())
                .env("STORY_RECOVERY_CAPABILITY_TOKEN", host.capability_token.expose())
                .status().await.unwrap();
            assert!(status.success(), "inspect retained recovery result");
        });
    }
}
