# AI 动漫 / 漫剧生成质量质检标准手册与 VLM 审核 Prompt 模板

本手册用于统一人工审片、自动 VLM 裁判（`plot_critic.py`）与外部多模态监督 Agent（MCP Supervisor）对 **AI 动漫 / 漫剧** 的抽帧检查流程、6 大质检维度、分题材权重阈值和一票否决红线。

> 与代码实现一一对应：
> - 题材配置与 Rubric 生成：[qa_profiles.py](../story_media_orchestrator/qa_profiles.py)
> - 抽帧拼图、ROI 局部放大、色温/光线漂移与音画同步检测：[frame_decomposer.py](../story_media_orchestrator/frame_decomposer.py)
> - 自动多模态质检裁判：[plot_critic.py](../story_media_orchestrator/plot_critic.py)
> - MCP 监督工具与资源：[mcp_server.py](../story_media_orchestrator/mcp_server.py)（`story_get_qa_rubric`、`story://qa/profiles`、`story_inspect_shot_multimodal`）

---

## 一、标准抽帧与诊断图组合策略

单靠 2~4 张缩小后的关键帧容易漏掉瞬间穿模、微表情崩坏、口型错拍与中段冻结。本系统每个视频镜头默认生成 **三图 + 一套音画遥测**：

1. **2×2 核心锚点图（`*_contact_sheet.jpg`）**
   - 采样位置：首（6%）、中前（36%）、中后（66%）、尾（92%），并在 ±2 帧窗口自动避开纯黑帧。
   - **异常帧自动顶替机制**：先在中段 [20%, 80%] 扫描 10 帧，计算局部清晰度保留率、刚性平移残差（Phase Correlation）与边缘撕裂连通域变化；若发现中段疑似穿模/撕裂帧（`sharp_topology_tear` / `soft_mush`），自动替换进 2×2 宫格并用红框标注 `!CLIP?`。
2. **8 帧密集时序条（`*_dense_strip.jpg`）**
   - 采样位置：`4% -> 17% -> 30% -> 43% -> 56% -> 69% -> 82% -> 95%`。
   - 用来直观核查：**画面是否变 PPT/冻结**（`temporal_freeze_ratio`）、**亮度闪烁**（`flicker_max_luma_jump`）、**色温/色相漂移**（`color_drift_max_delta`、`color_temp_max_jump`）、**画风饱和度跳变**（`saturation_max_jump`）以及**运动峰值时间点**（`video_motion_peak_seconds`）。
3. **4 格 ROI 局部放大对比条（`*_roi_zoom.jpg`）**
   - 自动检测人脸区域（Haar 检测优先，动漫风格化人脸自动回退到上半身显著边缘重心裁剪），把 4 个锚点帧的头部/五官/嘴部放大拼成横向对比条。
   - 专门解决缩略图看不清**变脸、眼神飘忽、苦笑变惊悚笑、对白口型错拍、旁白乱张嘴**的问题。
4. **音频包络与音画同步遥测（`audio` + `av_sync`）**
   - 从 `.wav` 音轨提取响度（`rms_db`）、削波爆音占比（`clipping_ratio`）、静音占比与首尾/内部最长空白（`silence_ratio` / `leading_silence_s` / `trailing_silence_s` / `max_internal_silence_s`）、能量突变峰值点（`onset_times_s`）。
   - 与视频做跨模态对齐：核查音画时长差（`duration_delta_seconds`）以及动作最高潮帧与最近音效峰值的时间差（`nearest_onset_delta_seconds`）。

> **动漫风格特别说明**：动漫常见刻意使用“一拍二 / 一拍三”（12fps/8fps）的节奏顿挫或 1~2 帧冲击定格帧（Impact Frames），质检时严禁直接按实拍电影 60fps 的平滑度标准误杀正常动漫韵律。

---

## 二、6 大核心质检维度（通用语言规范）

所有镜头统一围绕以下 6 个维度进行描述与打分（`0.0 ~ 1.0`）：

| 维度键 | 中文名称 | 核心检查项（质检员怎么看） | 对应客观指标（`diagnostics`） | 典型废片红线 | 推荐 `failure_code` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `character` | 人物一致性与解剖结构 | 脸型、瞳色、发型/发饰、服饰花纹跨帧一致；四肢与手指解剖合理；赛璐璐轮廓线清晰稳定不“沸腾”；关键道具不私自变形 | `is_clipping_suspect`、`clipping_mode`（`sharp_topology_tear`） | 中途变脸/换角、发色服饰突变、多指/融手、反关节、武器/头发穿进身体 | `CHARACTER_INCONSISTENT` / `HAND_FACE_DEFECT` |
| `motion` | 动作连贯性与物理交互 | 动作具备起势→发力→收势完整轨迹；手物/双人接触点真实；布料/发丝随惯性与风向自然摆动 | `motion_mean_diff`、`temporal_freeze_ratio`、`pose_locked_vfx_suspect` | 站桩假动（仅粒子/背景动、主体僵死）、动作走一半卡死、接触面融化粘连 | `VIDEO_MOTION_FAIL` / `VIDEO_STATIC_OR_WARP` |
| `expression` | 微表情与眼神视线 | 表情贴合当前台词与心理；眼神有高光且有明确焦点（双人对话视线对得上）；眨眼与情绪过渡自然 | `roi_zoom_path`（局部放大对比条复核） | 情绪错位（苦笑变惊悚笑）、翻白眼、脸部像橡皮泥变形、五官飘移 | `HAND_FACE_DEFECT` / `PLOT_DRIFT` |
| `camera_displacement` | 位移、运镜与空间轴线 | 推/拉/摇/移/跟拍意图清晰平滑；背景建筑/地平线刚性稳定；人物位移脚踏实地；镜头切换遵循 180° 空间轴线 | `pan_shift`、`motion_blur_likely`、`nonrigid_center_mad` | 无故晕镜抖动、推镜时背景扭曲融化、原地滑步/瞬移、双人对话莫名越轴 | `VIDEO_STATIC_OR_WARP` / `VIDEO_MOTION_FAIL` |
| `lighting_style` | 光线、色温与画风稳定 | 主光源方向固定、阴影随运动合理移动；色温与影调前后统一；2D/3D 风格稳定不串台；无黑屏死黑或惨白过曝 | `flicker_max_luma_jump`、`color_drift_max_delta`、`color_temp_max_jump`、`saturation_max_jump`、`dark_clip_ratio_avg`、`overexposed_ratio_avg` | 高频闪屏、阴影在脸上乱飘、夕阳秒变冷夜、2D 赛璐璐突然油腻 3D 化 | `COMPOSITION_LIGHTING` |
| `audio_sync` | 声音质量、口型与节奏同步 | 对白嘴唇开合贴合语速与停顿；旁白/画外音人物自然闭口；打击/开门等动作峰值与音效对齐；无爆音/异常长静音 | `audio.clipping_ratio`、`audio.silence_ratio`、`av_sync.duration_delta_seconds`、`av_sync.nearest_onset_delta_seconds` | 闭嘴发声、嚼口香糖式乱动嘴、旁白镜头角色张嘴、爆音破音、话音早停画面发呆 | `AUDIO_PACING_MISMATCH` / `IMAGE_PROMPT_DRIFT` |

---

## 三、分题材质检配置表（6 套内置 Profile）

不同题材的审美与容忍边界差异极大，系统按 `ProjectManifest.genre`（或单镜 `scene_context.genre`、未指定时自动按剧情关键词推断）加载对应配置：

### 1. 权重矩阵（6 维度权重和 = 100%）

| 题材 ID | 题材名称 | 人物 `character` | 动作 `motion` | 表情 `expression` | 运镜位移 `camera` | 光线画风 `lighting` | 声画口型 `audio` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `general` | 通用动漫 / 剧情均衡 | 24% | 18% | 16% | 14% | 14% | 14% |
| `action` | 热血打斗 / 动作番 | 18% | **30%** | 8% | **18%** | 10% | **16%** |
| `romance` | 言情 / 日常 / 情感对话 | **26%** | 10% | **26%** | 8% | **16%** | 14% |
| `suspense` | 悬疑 / 惊悚 / 暗黑推理 | 20% | 14% | **18%** | **18%** | **18%** | 12% |
| `fantasy` | 玄幻 / 仙侠 / 科幻机甲 | **22%** | **22%** | 10% | **18%** | **16%** | 12% |
| `comedy` | 搞笑 / 萌系 / Q版轻喜剧 | 22% | 16% | **24%** | 10% | 12% | **16%** |

### 2. 客观抽帧与音画阈值矩阵

| 指标 | `general` | `action`（打斗） | `romance`（言情） | `suspense`（悬疑） | `fantasy`（玄幻/科幻） | `comedy`（搞笑） |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 静态警告下限 `static_motion_min` | `1.20` | **`1.80`**（严防站桩） | `0.65`（允许轻呼吸微动） | `0.70`（允许缓慢压迫） | `1.30` | `0.90` |
| 冻结占比上限 `max_freeze_ratio` | `0.45` | **`0.30`** | `0.65` | `0.62` | `0.40` | `0.52` |
| 亮度跳变上限 `max_luma_jump` | `30.0` | `46.0`（容忍冲击闪帧） | **`22.0`**（严防氛围闪烁） | **`22.0`** | `40.0`（容忍能量光效） | `34.0` |
| 色度漂移上限 `max_color_drift` | `18.0` | `26.0` | **`13.5`**（严控柔光色温） | **`14.0`** | `24.0` | `18.0` |
| 清晰度保留下限 `clipping_retain_ratio` | `0.45` | `0.34`（放宽动态模糊） | **`0.56`**（面部必须清晰） | `0.50` | `0.42` | `0.45` |
| 暗帧亮度底线 `dark_luma_floor` | `8.0` | `7.5` | `9.0` | **`4.5`**（允许暗黑影调） | `7.5` | `10.0` |
| 音画时长差上限 `audio_duration_delta_max_s` | `0.85s` | **`0.65s`** | `0.70s` | `0.75s` | `0.80s` | **`0.65s`** |
| 峰值同步窗 `audio_peak_sync_window_s` | `0.45s` | **`0.25s`**（强打击卡点） | `0.50s` | `0.35s` | `0.35s` | `0.30s` |

### 3. 各题材放宽项与一票否决红线速查

* **`action`（热血打斗）**
  - **放宽**：高速挥击动态模糊、速度线、夸张透视、1~2 帧冲击闪白帧。
  - **一票否决**：反关节/多肢体、兵器穿模或变弯消失、光效狂闪人站桩、中段糊成无法分辨攻防的色块。
* **`romance`（言情日常）**
  - **放宽**：情绪特写中的低幅度微动（眨眼、发丝轻拂、缓慢推镜）。
  - **一票否决**：微表情错位（苦笑变惊悚笑）、主角变脸、拥抱/牵手身体手指融化粘连、色温氛围突变。
* **`suspense`（悬疑惊悚）**
  - **放宽**：低照度暗黑影调、强对比阴影、缓慢推进与静默留白。
  - **一票否决**：关键线索道具（信件、伤口、凶器）变形消失、暗部死黑糊成一团不可辨认、高频闪烁、空间瞬移越轴。
* **`fantasy`（玄幻仙侠 / 科幻机甲）**
  - **放宽**：大招/法阵/推进器引起的局部光效与饱和度抬升；大远景弱化微距五官。
  - **一票否决**：严重画风穿帮（古风混现代/西式铠甲）、机甲/法宝对称结构融化、远景宫殿山体塌陷成抽象涂鸦、特效糊死人脸。
* **`comedy`（搞笑萌系）**
  - **放宽**：夸张表情符号（汗滴、青筋、星星眼）、短暂吐槽定格与弹性形变。
  - **一票否决**：夸张颜艺失控变恐怖谷畸形脸、台词说完长时间尴尬发呆、画面冒出乱码对白气泡或分格边框。

---

## 四、可直接复用的 VLM 审核 Prompt 模板

### 模板 1：动态视频质检 Prompt（配合 2×2 宫格 + 8 帧时序条 + ROI 局部放大条）

```text
你是 AI 动漫短剧质检监督员。请基于以下【题材质检模板】与多模态诊断图，对镜头 {shot_id} 的生成视频进行专业验收。

{rubric_block}

【镜头剧本与约束】
- 剧情动作：{visual_action}
- 场景与景别：{location} | {framing}
- 出场角色设定（Visual DNA）：{character_summary}
- 音频与口型要求：{dialogue_or_voiceover_rule}
- 客观抽帧/音画遥测：{diagnostics_json}

【多模态证据核查顺序】
1. 看【角色母版参考图】+【ROI 局部放大对比条】：查五官、瞳色、发饰是否漂移；微表情是否贴合情绪；对白口型是否自然（旁白必须闭口）。
2. 看【2x2 核心锚点图】：重点查带 `!CLIP?` 红框帧是否存在肢体穿模、多手、兵器变形、文字水印或分屏边框。
3. 看【8帧密集时序条】：查动作起势-发力-收势是否连贯、有无中后段僵死或站桩假动、背景透视是否随运镜融化、光线与色温是否跳变闪烁。
4. 核对【音画遥测】：检查时长差、削波爆音、异常长静音以及动作高潮与音效峰值是否脱节。

【输出要求】
严格输出单个 JSON 对象，不要附加额外解释：
{
  "decision": "passed | warning | failed",
  "failure_code": "PASS | PLOT_DRIFT | IMAGE_PROMPT_DRIFT | CHARACTER_INCONSISTENT | HAND_FACE_DEFECT | LOCAL_DEFECT | SPLIT_PANEL_TEXT | VIDEO_STATIC_OR_WARP | VIDEO_MOTION_FAIL | AUDIO_PACING_MISMATCH | COMPOSITION_LIGHTING",
  "scores": {
    "character": 0.0,
    "motion": 0.0,
    "expression": 0.0,
    "camera_displacement": 0.0,
    "lighting_style": 0.0,
    "audio_sync": 0.0
  },
  "defects": [
    {
      "dimension": "character | motion | expression | camera_displacement | lighting_style | audio_sync",
      "severity": "high | medium | low",
      "frame_hint": "如 #2 t=1.20s 或全程",
      "description": "具体可执行的缺陷描述"
    }
  ],
  "reason": "一句话总结通过或打回的根因",
  "rework_action": {
    "strategy": "none | regenerate_first_frame | patch_character_dna | reroute_video | fallback_25d_motion",
    "revised_prompt": null,
    "negative_addition": null,
    "target_video_route": null
  }
}
```

### 模板 2：静态首帧 / 尾帧预审 Prompt（在烧视频算力前拦截废图）

```text
你是 AI 动漫关键帧预审员，目标是在进入付费图生视频前拦截构图、人物与逻辑废图。

{rubric_block}

【镜头信息】
- 画面动作：{visual_action}
- 出场角色 DNA：{character_summary}
- 客观图像指标：{image_diagnostics_json}

【必查硬红线】
1. 严禁分屏、上下/左右多格漫画拼贴、白边/黑边分隔条、乱码汉字、气泡字幕或水印。
2. 人物面部五官、瞳色、发型与服饰必须严格匹配角色设定与母版；手部手指数量与握持姿势自然无融化。
3. 光源方向、景深与氛围色温符合题材与场景时间；如为旁白/画外音镜头，人物必须自然闭口。
4. 如本帧作为动作起始/结束锚点，人物姿态必须留有合理的运动空间且不与背景道具穿模。

严格输出单个 JSON 对象（结构同模板 1）。
```

---

## 五、在 `story-media-orchestrator` 中的接入方式

1. **项目级配置题材**：在 `project.json` 中设置 `"genre": "action"`（支持中英文别名，如 `"打斗"` / `"言情"` / `"悬疑"` / `"玄幻"` / `"搞笑"`）；也可在单镜 `shot.scene_context["genre"]` 中按分镜覆盖（例如悬疑剧中插一段打斗戏）。
2. **自动抽帧与诊断**：运行 `refresh_shot_diagnostics(manifest, project_root, shot_id)` 或调用 MCP `story_inspect_shot_multimodal` 时，会自动生成：
   - `*_contact_sheet.jpg`（2×2 锚点 + 异常顶替）
   - `*_dense_strip.jpg`（8 帧密集时序条）
   - `*_roi_zoom.jpg`（4 格局部人物放大条）
   - `diagnostics.video` 中的色温/色相/饱和度漂移指标与动作峰值时间戳
   - `diagnostics.audio` & `diagnostics.av_sync` 音画同步指标
   - `diagnostics.objective_qa` 题材化客观预警报告
3. **MCP 获取标准与审核**：
   - 调用 `story_get_qa_rubric(genre="romance", project_dir=..., shot_id=...)` 可直接拿到结构化配置与组装好的 VLM Prompt。
   - 读取资源 `story://qa/profiles` 可查看全部 6 套内置题材质检模板。
   - 提交裁决 `submit_quality_verdict(..., scores={...}, genre=...)` 会按当前题材权重自动计算加权 `overall_score`。
