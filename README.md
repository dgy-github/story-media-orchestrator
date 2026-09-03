# Story Media Orchestrator

顶层调度层，串联三个独立项目：故事生成、故事生图、故事视频。

第一阶段范围：单故事、单场景、首帧/尾帧图片、单条 5 秒视频计划。

本目录只依赖契约（`story-package/v1`、`image-production-plan/v1`、
`video-generation-pipeline/v2`），通过注入 callable 连接实际 agent；不复制三个项目代码。
默认测试使用 fake agent，不调用真实模型、服务器或密钥。

当前真实接入仍需由宿主注入三个 agent callable；故事 campaign 的 Rust capability、图片
DashScope provider 和视频 MiniMax H3 ComfyUI provider 不会在导入本包时自动启动。

正式装配入口为 `build_runtime(story_runner=...)`。它读取
`STORY_IMAGE_AGENT_ROOT`、`STORY_VIDEO_AGENT_ROOT` 和可选的
`STORY_MEDIA_ARTIFACT_ROOT`，图片 provider 使用 `DashScopeImageProvider` 的
`from_nanocodex_config()`，视频 provider 使用 `ComfyUIAdapter.from_environment()`。
故事 runner 必须由主项目 Rust capability-backed runtime 注入；本包不会自行创建
Rust 进程、读取其凭据或在 import 时发起网络请求。

统一模型配置入口为 `OrchestratorConfig.from_environment()` 或
`build_runtime_from_environment(story_runner=...)`。支持 `STORY_MODEL`、
`STORY_IMAGE_MODEL`、`STORY_IMAGE_SIZE`、`STORY_VIDEO_MODEL`、
`STORY_VIDEO_TURBO`、`STORY_VIDEO_STEPS`；视频连接使用
`MINIMAX_H3_COMFYUI_BASE_URL` 与 `MINIMAX_H3_COMFYUI_TOKEN`。密钥仍由下游
provider 按各自安全配置读取。

本地界面：

```powershell
python -c "from story_media_orchestrator.ui import launch; launch()"
```

界面默认只显示并运行 fake preview；真实 runner 由宿主注入。`workspace.toml` 记录三个
独立 agent 的路径环境变量和契约，当前采用 external-workspaces 模式，不复制或改写三个仓库。

一键配置并启动本地 sidecar：

```powershell
.\scripts\start-local-stack.ps1
```

脚本会生成或复用用户级 `MICROCODEX_SIDECAR_TOKEN`，配置 sibling 路径和 MiniMax H3
ComfyUI 地址，启动 sidecar 并执行健康检查；不会打印 token。Rust capability 已启动时，
可传入 `-CapabilityUrl` 和 `-CapabilityToken`。

也可以直接使用 UI 的“保存配置 / 加载配置 / 启动本地栈”按钮。配置保存在当前用户的
`%LOCALAPPDATA%\StoryMediaOrchestrator\config.json`，不进入 Git；token 输入框默认隐藏。

## Rust/Tauri 桌面端

正式桌面端位于 `apps/desktop`，复用 Rust/Tauri 技术路线。它提供：

- Rust 保存模型设置，API key/token 通过系统凭据存储；
- Svelte 配置页和自动生成 sidecar token；
- Rust 调度入口及故事→图片→视频阶段状态；
- 每阶段 artifact、失败可重试状态展示。

开发运行：

```powershell
cd apps/desktop
npm install
npm run tauri dev
```

当前桌面端的阶段推进命令已用本地状态机验证；接入真实 sibling runtime 时，将
`start_media_run` 的阶段执行替换为现有 Python/Rust adapter 调用，界面契约保持不变。
