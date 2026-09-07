# 配套仓库版本

更新：2026-09-08。下列配套版本均已发布到 GitHub。完整桌面请使用这个组合；故事仓库使用指定功能分支，不是默认 main。

- [microcodex-short-drama-studio](https://github.com/dgy-github/microcodex-short-drama-studio/tree/0f2e677ba07a77a642432501037e2b01ad3dc880)：分支 `feature/eval-p3a-unlock`，版本 `0f2e677ba07a77a642432501037e2b01ad3dc880`。
- [story-image-agent](https://github.com/dgy-github/story-image-agent/tree/d697d455c52d70c1056cf426a38b575e5f4c780e)：分支 `main`，版本 `d697d455c52d70c1056cf426a38b575e5f4c780e`。
- [story-video-agent](https://github.com/dgy-github/story-video-agent/tree/b6c12259f498b5b5f093e17bd4fe4adc322f641f)：分支 `main`，版本 `b6c12259f498b5b5f093e17bd4fe4adc322f641f`。

## 获取配套代码

在这些仓库的共同父目录执行：

```powershell
git clone --branch feature/eval-p3a-unlock https://github.com/dgy-github/microcodex-short-drama-studio.git
git -C microcodex-short-drama-studio checkout 0f2e677ba07a77a642432501037e2b01ad3dc880
git clone --branch main https://github.com/dgy-github/story-image-agent.git
git -C story-image-agent checkout d697d455c52d70c1056cf426a38b575e5f4c780e
git clone --branch main https://github.com/dgy-github/story-video-agent.git
git -C story-video-agent checkout b6c12259f498b5b5f093e17bd4fe4adc322f641f
```

已有仓库请先保存自己的改动，再 fetch 并切换到对应版本。使用固定提交可以避免各仓库继续开发后接口不同步。

这些提交不包含本机其他未提交的实验性修改。完整 AI 功能仍需 Python 环境、模型凭据及相应服务；可按[安装说明](GETTING_STARTED.md)继续配置。本仓库的本地故事卡 CLI 预览可独立运行。
