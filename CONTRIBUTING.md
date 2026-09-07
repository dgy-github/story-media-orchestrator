# 开发与贡献

欢迎提交可复现的问题、界面改进建议，以及带有验证记录的修复。项目定制与商务沟通请联系 [2537676793@qq.com](mailto:2537676793@qq.com)。

## 本地检查

```powershell
python -m pip install -e ".[preview]" pytest
python -m pytest -q
npm ci --prefix apps/desktop
npm run build --prefix apps/desktop
cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml
```

Rust 与部分集成测试依赖相邻原项目，请先按[安装说明](docs/GETTING_STARTED.md)准备目录与环境。显式标记为 ignored 的真实模型验收会访问服务或产生费用，不应在普通 CI 中自动启用。

## 提交内容

- 说明问题、变更后的行为和验证方式。
- 避免把模型调用测试写成默认付费请求；离线测试和真实验收分开记录。
- 不提交密钥、账号配置、项目缓存、个人目录、运行日志或未经授权的素材。
- 示例媒体应说明来源与验证范围，不能把模拟测试当作实际生成效果。
- 请保留任务回执、缓存失效和失败恢复的行为，不用静默降级掩盖故障。

涉及商用授权、源码交付或第三方素材，请先明确适用协议与交付范围。
