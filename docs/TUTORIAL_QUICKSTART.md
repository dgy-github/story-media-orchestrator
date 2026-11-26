# 10 分钟零门槛上手：从一句话故事到成片与剪映草稿

这份教程面向**第一次接触本项目的新手**。哪怕你暂时没有申请任何商业大模型 API Key、本机也没手动配置过系统 FFmpeg，也可以在 10 分钟内跑通第一条完整视频流水线。

---

## 第一步：安装环境（推荐 Python 3.11+）

克隆仓库后，在项目根目录执行：

```bash
pip install -e ".[preview,mcp]"
```

> **为什么装 `.[preview,mcp]`？**
> - `preview` 自带 `Pillow` 和 `imageio-ffmpeg`（内置 FFmpeg 二进制，无需手动配置系统 PATH）。
> - `mcp` 安装 MCP 协议与 `opencv-python`，用于视频拆帧诊断和接入 AI 客户端。
> - 如果后续需要使用对白 VAD + ASR 审计（`audit-speech`），再按需追加 `pip install -e ".[asr]"`。

安装完成后，跑一次环境自检：

```bash
python -m story_media_orchestrator.cli doctor
```

看到 `"ffmpeg_available": true` 且 `install_hints` 为空，说明本地基础环境已经全部就绪（`ffprobe` 非必需，无 `ffprobe` 也能正常运行）。

---

## 第二步：创建你的第一个短剧工程

用一句简短的故事创建一个工程目录（例如 `demo-ep01`）：

```bash
python -m story_media_orchestrator.cli create demo-ep01 "林雾撑伞走过雨夜长街。她停在钟楼门前，抬头望向亮着暖光的阁楼。"
```

执行后会在 `demo-ep01/` 下生成 `project.json`，系统会自动把故事拆分成多个分镜（`shot-01`、`shot-02`...），每个镜头独立记录状态、资产路径与质检结论。

---

## 第二步（进阶推荐）：生图前跑一次「成片质量剧本预演」

在正式花钱生图和出视频之前，先让系统结合过往成片 Demo 经验（`.flywheel`）与题材要素，预演一遍剧本有没有缺血肉、动作过载或抽象特效漏洞：

```bash
python -m story_media_orchestrator.cli rehearse-script demo-ep01 --genre urban_suspense
```

系统会返回就绪分（0–100）、题材缺失项与优先级修复建议（`P0/P1`）；如果有问题，可直接用 `patch-script` 修单镜或 `insert-bridge` 插因果桥镜。

---

## 第三步：零成本跑通预览版成片（Preview 模式）

先不调用任何付费模型，用内置的本地预览模式把整条链路（分镜卡 → 本地静音/系统语音 → 画面动效 → 字幕 → MP4 合成）跑通：

```bash
python -m story_media_orchestrator.cli run demo-ep01 --mode preview --real-preview --tts-source silent --auto
```

如果你在 Windows 上，也可以把 `--tts-source silent` 换成 `--tts-source windows`，直接使用系统自带语音合成配音。

跑完后在 `demo-ep01/` 目录下就能看到合成好的 MP4 视频。

---

## 第四步：导出剪映草稿（方便二次精剪）

如果你想把生成的分镜、配音和字幕直接拖进剪映继续加特效、配乐或微调切点：

```bash
python -m story_media_orchestrator.cli export-jianying demo-ep01
```

导出完成后，把生成的草稿目录复制到剪映草稿文件夹即可直接打开编辑。

---

## 第五步：下一步该学什么？

跑通预览版后，你可以按需要继续进阶：

1. **接入 AI 客户端自动画图、看片、修镜头**：阅读 [MCP 多 Agent 协同接入指南](TUTORIAL_MCP_AGENT.md)
2. **用专业质检门禁控画质、做单镜返工**：阅读 [AI 短剧质量验收与低成本返工指南](TUTORIAL_QA_AND_REWORK.md)
3. **查看各模型与外部服务对接方式**：阅读 [安装、运行与常见问题](GETTING_STARTED.md) 与 [已验证能力与当前限制](VERIFICATION.md)
