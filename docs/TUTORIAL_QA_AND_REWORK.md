# AI 短剧质量验收与低成本返工实战指南

做 AI 短剧最容易踩的坑有两个：**一是单镜有问题就整集重跑（成本爆炸），二是靠肉眼抽查导致带字幕乱码、人物串脸或黑帧的废片混进成片**。本项目把成熟的制片质检纪律做成了代码硬门禁，新手照着下面的步骤就能稳定出片。

---

## 一、四态验收原则（每镜独立追踪）

每个镜头都按四个阶段独立验收，**文件存在不等于内容合格**：

1. **设计通过**：分镜动作、景别、台词与角色出场明确。
2. **静帧通过**：独立干净的 16:9（或目标画幅）起手首帧通过角色、服饰与无乱码文字核查。
3. **单镜视频通过**：通过 2×2 宫格 / 8 帧条带 / ROI 放大图复核，动作连贯、无严重穿模、无伪字幕、无旁白张嘴。
4. **入母带**：通过成片完整性、黑帧/静帧与响度验证，进入最终交付。

> 凡是改动了首帧、提示词或音轨，系统会自动变更输入指纹并使旧视频缓存失效，防止旧废片被误用。

---

## 二、发现问题先分诊，不要盲目整集重出

遇到不合格镜头时，按最早出错的环节对症下药：

| 现象 | 根因环节 | 正确修法 |
| :--- | :--- | :--- |
| 静帧阶段脸型、发色、服饰或画面文字就错了 | 静帧阶段 | 只重绘该镜首帧（`rewrite_image_prompt` / `patch_character_dna`），保留配音和其他镜头 |
| 首帧干净，但视频出现伪字幕、旁白张嘴或末尾僵住 | 视频清理 | 清理提示词里的画面台词、补充闭口/静音词约束，或裁剪僵死尾段 |
| 首帧摆拍导致视频人物站桩不动、动作锁死 | 动作拆拍 | 把首帧改回动作发生前的起势态，拆成一镜一拍，不要拿动作高潮峰值图当首帧 |
| 单镜都还行，但拼起来跳轴或节奏断层 | 剪辑衔接 | 调整切点、补反应镜或导进剪映做环境声跨切 |

---

## 三、生图前必做：成片质量剧本预演 + 上游回退闭环

很多视频“怎么改提示词都救不回来”，根因往往是**前置剧本缺少血肉、因果或把峰值态当首帧**。系统内置了 6 大剧本硬伤扫描与分题材预演：

```bash
# 1. 成片质量剧本预演（结合历史 Demo 飞轮教训 + 题材要素缺口 + 单镜硬伤）
python -m story_media_orchestrator.cli rehearse-script ./demo-ep01 --genre xianxia

# 2. 逐镜剧本细节审核（识别抽象心理、多动作过载、全屏亮光、峰值首帧、台词超长等）
python -m story_media_orchestrator.cli audit-script ./demo-ep01

# 3. 视频不过关时回退修改上游单镜剧本（只失效该镜视觉缓存，保留其他镜头与配音）
python -m story_media_orchestrator.cli patch-script ./demo-ep01 shot-01 \
  --text "陆南攥紧布囊，将最后半块干粮塞进阿石手里。" \
  --visual-action "陆南低头将粗饼塞入阿石衣襟" \
  --framing "近景" --location "雨夜石寨土墙边"

# 4. 缺动机/反应/余波时，在指定锚点后插入因果桥镜（不打乱原有镜号与缓存）
python -m story_media_orchestrator.cli insert-bridge ./demo-ep01 shot-01 \
  --text "阿石眼眶发红，按住陆南手腕又缓缓松开。" --story-role reaction_handoff
```

---

## 四、4 条开箱即用的硬门禁命令

在合成本集最终母带之前，建议依次跑这几个确定性代码校验（零大模型费用）：

```bash
# 1. 校验交接清单与四态字段完整性
python -m story_media_orchestrator.cli lint-handoff <manifest.json> --check-files

# 2. 校验 H3 视频包（17k+5 帧长网格、静音镜语音触发词清洗）
python -m story_media_orchestrator.cli lint-h3-bundle <bundle_dir>

# 3. 校验对白音轨 VAD + ASR 与字幕对齐（需安装 .[asr]）
python -m story_media_orchestrator.cli audit-speech <chapter_dir> --bundle <review.json>

# 4. 校验最终成片 MP4（分辨率、帧率、黑帧/静帧与响度）
python -m story_media_orchestrator.cli verify-master <final.mp4>
```

---

## 五、进阶：无人值守多模型自动审片（CLI）

如果你有自己的视觉模型 API Key（例如 Gemini / Grok / DashScope 等 OpenAI 兼容接口），可以不打开客户端窗口，直接用命令行批量自动审片：

```bash
# 通过环境变量注入密钥（严禁写入代码或项目文件）
export GEMINI_API_KEY="你的主审模型 Key"
export STORY_GEMINI_CRITIC_MODEL="gemini-2.5-pro"

# 可选：配置二审仲裁模型（仅在主审给 warning/unavailable 或返工 >= 2 轮时才触发，省费用）
export XAI_API_KEY="你的二审模型 Key"
export STORY_GROK_CRITIC_MODEL="grok-2-vision-1212"

# 批量审核工程中所有未通过的镜头，并将裁决持久化写回 project.json
python -m story_media_orchestrator.cli critic-panel ./demo-ep01 --all --only-pending --stage video --policy escalate --persist
```

- 零 token 客观硬门禁会最先拦截缺文件、黑帧和分屏拼贴废图；
- 只有通过客观硬门的镜头才会发给主审模型，主审明确 `passed` 时不会浪费二审 token；
- 所有核验通过的修复案例都会自动沉淀到 `.flywheel/playbook.json`，越用越准。
