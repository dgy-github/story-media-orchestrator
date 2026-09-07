# 安装与使用

## 先跑本地预览

本地故事卡预览只需 Python 3.11+ 和本仓库。它用于验证流程，不生成 AI 场景图片。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[preview]"
python -m story_media_orchestrator.cli create my-story "信使来到钟楼。城市渐渐苏醒。"
python -m story_media_orchestrator.cli run my-story --real-preview --image-source local
```

结果保存在 `my-story/preview.mp4`。FFmpeg 优先从 `STORY_MEDIA_FFMPEG`、PATH 或 `imageio-ffmpeg` 查找。中文字体可通过 `STORY_MEDIA_FONT` 指定，Windows 默认使用可用的中文系统字体。

## 完整桌面与 AI 工作流

**配套版本发布状态：** 本次验收所用的依赖版本尚未全部公开，默认分支不保证能编译完整桌面。请先阅读[配套版本说明](DEPENDENCIES.md)；下列克隆命令用于准备目录，不代表完整集成版本已发布。

当前采用相邻仓库方式，尚未将全部运行时打成独立安装包。请把以下项目放在同一个父目录；各依赖的安装说明和许可以其仓库为准。

```text
workspace/
  story-media-orchestrator/
  microcodex-short-drama-studio/
  story-image-agent/
  story-video-agent/
```

在父目录执行：

```powershell
git clone https://github.com/dgy-github/microcodex-short-drama-studio.git
git clone https://github.com/dgy-github/story-image-agent.git
git clone https://github.com/dgy-github/story-video-agent.git
```

故事仓库需要自己的 Python 环境：

```powershell
cd microcodex-short-drama-studio
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e sidecar
cd ..\story-media-orchestrator
.venv\Scripts\Activate.ps1
python -m pip install -e ".[preview]"
python -m pip install -e ../story-image-agent -e ../story-video-agent
```

桌面执行媒体任务时使用 PATH 上的 `python`，因此请从激活本项目环境的终端启动。原故事服务默认使用相邻故事仓库的 `.venv/Scripts/python.exe`，可通过 `MICROCODEX_PYTHON` 指定解释器。

构建还需要 Node.js/npm、Rust/Cargo，以及 Windows Tauri 开发环境（包括 Microsoft C++ 构建工具与 WebView2）。

```powershell
npm ci --prefix apps/desktop
npm run desktop:build --prefix apps/desktop
```

输出：`apps/desktop/src-tauri/target/release/story-media-orchestrator-desktop.exe`。使用该构建命令可内置前端，不依赖 Vite。运行中的同名程序会阻止覆盖，请先保存工作并关闭旧程序。

开发界面：

```powershell
npm run dev --prefix apps/desktop
# 另开一个已激活 Python 环境的终端
npm run tauri:dev --prefix apps/desktop
```

浏览器直接访问 Vite 仅适合界面开发；项目文件操作与模型调用需要 Tauri 桥接。

## 配置模型服务

- **故事**：在“故事”节点选择原有模板。故事服务配置中分别设置 DeepSeek 生成与阿里云审核凭据；凭据保存到 Windows 凭据存储。当前六集默认建议 30 万 Token，实际消耗随模型和内容变化。已有项目的显式预算会保留。
- **图片**：选择 DashScope。适配器沿用原图像项目的 `~/.nanocodex/config.toml` 配置；请按照 [story-image-agent](https://github.com/dgy-github/story-image-agent) 的说明配置密钥、区域与模型。高级参数可以覆盖图片模型和尺寸。
- **视频**：在“合成”中选择模型视频与 DashScope。北京路线默认 `wanx2.1-t2v-turbo`，按当前代码校验模型、尺寸和时长；账号和区域应与实际服务一致。其他模型是否可用，以供应商与适配器支持为准。
- **ComfyUI**：在折叠的视频服务设置中填写自己的服务地址，并准备兼容的工作流与模型。不能仅凭填写 URL 就认为 GPU 环境已就绪。
- **配音**：可选静音或 Windows 本地语音。中文需要已安装的 zh-CN 声音，缺失时明确报错；云端语音未接入。

付费模型调用使用你配置的账号。不要把密钥写进故事、Issue、示例配置或提交到 Git。

## 操作顺序

1. 顶部选择项目目录，创建或打开项目。
2. 点击“故事”，手动输入或按模板生成。
3. 查看分镜，确认内容、候选图片和需要的生成参数。
4. 点击“开始作业”；启用人工审核时，按节点批准后继续。
5. 在右侧播放结果，底部查看日志。失败后根据具体原因使用恢复或重试。

故事、分镜和模型视频是不同阶段。阿里云文生视频路线不会自动保证人物一致性；首尾帧路线需要正确选择两张图片，且应单独验证效果。

## 恢复、重试与审核

```powershell
story-media resume my-story
story-media retry my-story shot-01
story-media run my-story --real-preview --require-review
story-media review my-story all
story-media resume my-story
```

`resume` 复用有效素材；`retry` 是明确重做指定镜头，可能触发新的付费请求。`--auto` 对本次调用跳过人工审核。审核绑定分镜或素材内容，内容变化会使对应批准失效。

生成服务返回的任务编号会保存，用于继续轮询和下载。如果提交结果未知，系统不会把再次付费提交伪装成“恢复”。故事界面的“取回原任务结果”也不等于重新启动已经终止的原服务任务。

```powershell
story-media run my-story --real-preview --tts-source windows
story-media candidates my-story shot-01 --candidate-count 2
story-media effects my-story shot-01 --motion push_in --transition fade
```

候选生图使用已保存的 DashScope 设置。镜头运动支持 none/push_in/pan；转场 fade 为经黑场淡入淡出，不是重叠交叉溶解。

## 项目文件

```text
my-story/
  project.json
  frames/  audio/  video/  artifacts/
  subtitles.srt
  preview.mp4
```

项目清单保存分镜状态、参数、缓存、来源和审核信息。编码先写临时文件，成功后替换成片；失败时保留旧文件，并清除当前输出引用以免误认为新结果。

[验证范围](VERIFICATION.md) · [返回首页](../README.md)
