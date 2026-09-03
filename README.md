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
