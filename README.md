# Story Media Orchestrator

顶层调度层，串联三个独立项目：故事生成、故事生图、故事视频。

第一阶段范围：单故事、单场景、首帧/尾帧图片、单条 5 秒视频计划。

本目录只依赖契约（`story-package/v1`、`image-production-plan/v1`、
`video-generation-pipeline/v2`），通过注入 callable 连接实际 agent；不复制三个项目代码。
默认测试使用 fake agent，不调用真实模型、服务器或密钥。
