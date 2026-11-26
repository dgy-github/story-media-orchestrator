# 手把手教程：用 AI 客户端（Antigravity / Codex / Cursor）通过 MCP 协同做片与审片

本项目的核心设计之一是**通过标准 MCP（Model Context Protocol）把工程状态、视频诊断图和素材注入能力暴露给 AI 客户端**。你可以直接复用客户端里已有的多模态大模型，让它充当「分镜美术」或「成片质检监督」，而不用自己写胶水脚本。

---

## 一、两种接入模式怎么选？

| 模式 | 适用场景 | 推荐参数 |
| :--- | :--- | :--- |
| **模式 A：单窗口直连（`stdio`）** | 个人新手在一个客户端窗口里完成看图、改图、审片、出片 | `--transport stdio --profile core` |
| **模式 B：多窗口并发（`streamable-http`）** | 进阶提效：一个窗口专职画图注入，另一个窗口专职看片审片 | `--transport streamable-http --host 127.0.0.1 --port 8780` |

### 模式 A：单窗口 `stdio` 配置示例

在你的 AI 客户端 MCP 配置文件中加入（参考根目录 `mcp_config.example.json`）：

```json
{
  "mcpServers": {
    "story-media-supervisor": {
      "command": "python",
      "args": [
        "-m",
        "story_media_orchestrator.mcp_server",
        "--transport",
        "stdio",
        "--profile",
        "core",
        "--flywheel-dir",
        "/path/to/story-media-orchestrator/.flywheel"
      ],
      "env": {
        "PYTHONPATH": "/path/to/story-media-orchestrator"
      }
    }
  }
}
```

### 模式 B：多窗口并发（本机 HTTP 共享服务）

在终端启动共享 MCP 服务：

```bash
python -m story_media_orchestrator.cli mcp --transport streamable-http --host 127.0.0.1 --port 8780 --profile core
```

也可以按职责拆成两个端口，让每个窗口只看到自己需要的精简工具列表，减少模型误调用：

```bash
# 生图/素材窗口（仅暴露 9 个素材生产与注入工具）
python -m story_media_orchestrator.cli mcp --transport streamable-http --host 127.0.0.1 --port 8781 --profile producer

# 审片/返工窗口（含剧本预演、多模态诊断、裁决、改剧本/插桥镜与验收工具）
python -m story_media_orchestrator.cli mcp --transport streamable-http --host 127.0.0.1 --port 8782 --profile critic
```

---

## 二、角色并发租约（防止多窗口互相覆盖）

多个 AI 客户端同时操作同一个工程时，使用 `story_claim_takeover` 的 `role` 参数隔离写权限：

- **素材生产侧（Producer）**：`role="asset_producer"` → 调用 `story_get_keyframe_tasks` 拉取待画清单，画好后带 `lease_token` 调用 `story_inject_external_asset` 注入 `image` / `last_frame` / `video` / `audio`。
- **质检监督侧（Critic）**：`role="critic"` → 调用 `story_inspect_shot_multimodal` 查看 2×2 视频宫格、8 帧密集条和 ROI 人物放大图，审完带 `lease_token` 调用 `story_submit_quality_verdict` 提交裁决。
- `asset_producer` 与 `critic` **可以并发持有、互不阻塞**；需要独占总控或终版装配时，再使用默认 `role="supervisor"`。

---

## 三、新手可以直接复制的 AI 协作指令

### 1. 让 AI 帮你看片审片并定位问题
> 请连接 `story-media-supervisor`，先调用 `story_get_pipeline_snapshot` 查看工程 `./demo-ep01` 的总览，再用 `story_claim_takeover(role="critic")` 获取审片租约。逐个对未通过镜头调用 `story_inspect_shot_multimodal` 查看首帧与视频诊断格，按角色一致性、动作连贯性、画面文字与音画同步给出评估，最后用 `story_submit_quality_verdict` 记录结果并释放租约。

### 2. 让 AI 帮你替换/注入单镜素材
> 请获取 `./demo-ep01` 的 `role="asset_producer"` 租约，调用 `story_get_keyframe_tasks` 查看待补齐的关键帧，把准备好的 16:9 干净首帧通过 `story_inject_external_asset` 注入到对应镜头，完成后释放租约。

### 3. 单镜低成本外科返工
> 请检查 `./demo-ep01` 的 `shot-02`，如果只是首帧构图或人物服饰漂移，优先调用 `story_intervene_and_rework_shot(strategy="rewrite_image_prompt")` 仅重绘首帧并保留已生成的配音，不要整集重跑。

### 4. 前置「成片质量剧本预演」与视频失败回退改剧本
> 请对 `./demo-ep01` 调用 `story_rehearse_script(genre="仙侠")` 执行成片质量剧本预演，结合历史 Demo 飞轮教训与题材缺失项，按 P0 → P1 顺序调用 `story_patch_script_beat` 补齐单镜可见动作，或调用 `story_insert_bridge_shot` 插入因果桥/反应镜；若后期某镜视频连续返工仍不过关，请直接回退调用 `story_patch_script_beat` 修改上游剧本（只清除该镜过期视觉缓存，不影响其他已通过镜头）。
