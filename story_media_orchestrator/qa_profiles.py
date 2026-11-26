"""Genre-aware Anime QA Profiles, objective diagnostic evaluators, and VLM rubric builders."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

QA_DIMENSIONS: tuple[str, ...] = (
    "character",
    "motion",
    "expression",
    "camera_displacement",
    "lighting_style",
    "vfx_quality",
    "audio_sync",
)

QA_DIMENSION_LABELS: dict[str, str] = {
    "character": "人物一致性与解剖结构",
    "motion": "动作连贯性与物理交互",
    "expression": "微表情与眼神视线",
    "camera_displacement": "位移、运镜与空间轴线",
    "lighting_style": "光线、色温与画风稳定",
    "vfx_quality": "特效质感、体积光影与空间透视",
    "audio_sync": "声音质量、口型与节奏同步",
}

# Motion defects that are hard-fails only when the shot is explicitly
# classified as action. Inferred-only action (keyword heuristic) stays warning
# so dialogue/static scenes are not promoted by accident.
ACTION_MOTION_HARD_CODES = frozenset({
    "POSE_LOCKED_VFX",
    "TEMPORAL_FREEZE",
    "TAIL_MOTION_STALL",
    "LOW_MOTION_STATIC",
})


def should_enforce_action_motion_fails(profile: dict[str, Any] | None) -> bool:
    """True only for an explicit action QA profile, not keyword inference."""
    if not isinstance(profile, dict) or str(profile.get("id") or "") != "action":
        return False
    return profile.get("resolved_from") == "explicit"


_GENRE_ALIASES: dict[str, str] = {
    # general
    "general": "general",
    "default": "general",
    "anime": "general",
    "通用": "general",
    "剧情": "general",
    "综合": "general",
    # action
    "action": "action",
    "battle": "action",
    "shonen": "action",
    "fight": "action",
    "热血": "action",
    "打斗": "action",
    "动作": "action",
    "格斗": "action",
    "武侠": "action",
    "燃向": "action",
    # romance
    "romance": "romance",
    "slice_of_life": "romance",
    "daily": "romance",
    "drama": "romance",
    "dialogue": "romance",
    "言情": "romance",
    "恋爱": "romance",
    "甜宠": "romance",
    "虐恋": "romance",
    "日常": "romance",
    "都市": "romance",
    "情感": "romance",
    "对话": "romance",
    # suspense
    "suspense": "suspense",
    "thriller": "suspense",
    "horror": "suspense",
    "mystery": "suspense",
    "noir": "suspense",
    "悬疑": "suspense",
    "惊悚": "suspense",
    "恐怖": "suspense",
    "暗黑": "suspense",
    "推理": "suspense",
    "刑侦": "suspense",
    "怪谈": "suspense",
    # fantasy
    "fantasy": "fantasy",
    "xianxia": "fantasy",
    "scifi": "fantasy",
    "sci-fi": "fantasy",
    "mecha": "fantasy",
    "玄幻": "fantasy",
    "仙侠": "fantasy",
    "修真": "fantasy",
    "奇幻": "fantasy",
    "科幻": "fantasy",
    "机甲": "fantasy",
    "异界": "fantasy",
    "赛博": "fantasy",
    # comedy
    "comedy": "comedy",
    "chibi": "comedy",
    "moe": "comedy",
    "搞笑": "comedy",
    "喜剧": "comedy",
    "日常搞笑": "comedy",
    "萌系": "comedy",
    "沙雕": "comedy",
    "q版": "comedy",
}

_GENRE_INFERENCE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("action", ("拔剑", "挥剑", "打斗", "格斗", "踢击", "重拳", "爆破", "追逐", "狂奔", "冲锋", "厮杀", "刀光", "剑气", "爆炸")),
    ("suspense", ("凶案", "密室", "尸体", "血迹", "侦探", "凶手", "悬疑", "诡异", "阴森", "线索", "监控", "证物", "潜行", "惊悚")),
    ("fantasy", ("修仙", "仙侠", "御剑", "法阵", "宗门", "灵气", "法宝", "巨龙", "星舰", "机甲", "魔法", "异界", "渡劫", "秘境")),
    ("romance", ("对视", "告白", "拥抱", "牵手", "落泪", "心碎", "夕阳", "咖啡馆", "暗恋", "初恋", "眼眶", "微笑", "柔光")),
    ("comedy", ("吐槽", "夸张", "颜艺", "摔了个狗吃屎", "呆毛", "Q版", "沙雕", "大笑", "滑稽", "搞笑")),
)


ANIME_QA_PROFILES: dict[str, dict[str, Any]] = {
    "general": {
        "id": "general",
        "label": "通用动漫 / 剧情均衡",
        "description": "适用于常规多场景剧情漫剧，7大维度均衡把关（含特效质感），兼顾动漫一拍二/一拍三节奏习惯。",
        "weights": {
            "character": 0.22,
            "motion": 0.20,
            "expression": 0.16,
            "camera_displacement": 0.14,
            "lighting_style": 0.10,
            "vfx_quality": 0.08,
            "audio_sync": 0.10,
        },
        "thresholds": {
            "static_motion_min": 1.2,
            "max_freeze_ratio": 0.45,
            "max_luma_jump": 30.0,
            "max_color_drift": 18.0,
            "max_saturation_jump": 26.0,
            "clipping_retain_ratio": 0.45,
            "allow_impact_motion_blur": False,
            "dark_luma_floor": 8.0,
            "audio_duration_delta_max_s": 0.85,
            "audio_clipping_ratio_max": 0.01,
            "audio_silence_ratio_max": 0.75,
            "audio_peak_sync_window_s": 0.45,
            "require_roi_zoom": False,
        },
        "focus_checks": {
            "character": "主角脸型、瞳色、发型、服饰关键配饰跨帧保持一致；四肢与手指解剖合理，无穿模与线条沸腾。",
            "motion": "动作具备起势-发力-收势连贯轨迹，避免站桩只有特效飘动或后半段突然僵死。",
            "expression": "面部表情自然贴合台词情绪，视线焦点明确，无诡异扭曲或翻白眼。",
            "camera_displacement": "推拉摇移平滑且背景透视稳定，人物不滑步、不瞬移，双人镜头不越轴。",
            "lighting_style": "主光源方向和阴影逻辑前后统一，无高频闪屏、无2D/3D画风突变、无色温跳变。",
            "vfx_quality": "如含光效/烟火/雨雪/粒子，需有体积感、空间前后遮挡与环境受光反馈；无特效镜头按自然环境物理打分。",
            "audio_sync": "对白口型与人声节奏匹配，旁白/画外音人物闭口不乱张嘴；无爆音、异常静音与声画脱节。",
        },
        "tolerances": [
            "允许动漫惯用的低帧率顿挫（一拍二/一拍三）与短暂静止停顿（Hold），不以实拍60fps标准误杀。",
            "远景/大远景镜头允许弱化微距五官细节，重点核查大轮廓、配色与空间稳定性。",
        ],
        "veto_rules": [
            "画面内出现乱码汉字、多余字幕条、分屏拼贴白边/黑边或水印。",
            "主角中段明显变脸、发色/服装突变，或出现多余肢体/反关节/严重穿模。",
            "画外音或旁白镜头中人物明显张嘴说话，或正脸近景对白镜头全程闭嘴发声。",
            "出现廉价扁平2D荧光油漆光条、无受光静态贴片光斑或离地假棉花团烟雾。",
        ],
    },
    "action": {
        "id": "action",
        "label": "热血打斗 / 动作番",
        "description": "强调发力轨迹、打击可读性、3D体积特效质感、肢体结构与音效卡点；适度放宽高速运动模糊与冲击闪帧。",
        "weights": {
            "character": 0.16,
            "motion": 0.26,
            "expression": 0.07,
            "camera_displacement": 0.15,
            "lighting_style": 0.08,
            "vfx_quality": 0.16,
            "audio_sync": 0.12,
        },
        "thresholds": {
            "static_motion_min": 1.8,
            "max_freeze_ratio": 0.30,
            "max_luma_jump": 46.0,
            "max_color_drift": 26.0,
            "max_saturation_jump": 36.0,
            "clipping_retain_ratio": 0.34,
            "allow_impact_motion_blur": True,
            "dark_luma_floor": 7.5,
            "audio_duration_delta_max_s": 0.65,
            "audio_clipping_ratio_max": 0.02,
            "audio_silence_ratio_max": 0.70,
            "audio_peak_sync_window_s": 0.25,
            "require_roi_zoom": False,
        },
        "focus_checks": {
            "character": "高速动作下主角辨识度与兵器形态保持稳定，严禁武器中途变弯、分裂或长在肉里。",
            "motion": "出拳、挥剑、闪避、受击轨迹清晰有力；严禁前半段动作做完后段2-3秒定格停滞，或光效乱飘人站桩。",
            "expression": "怒吼、咬牙、瞳孔震颤等强张力表情到位，不因剧烈运动融化成糊块。",
            "camera_displacement": "快切、跟拍、环绕运镜方向明确不晕镜；高速推拉时背景建筑透视不塌陷融化。",
            "lighting_style": "剑气、爆炸、闪光等局部高光需自然照亮人物轮廓，不能像廉价平面贴图。",
            "vfx_quality": "打击火花、剑气、音爆环、烟尘与能量波必须具备UE5级体积层次、粒子微细节与真实受击物理反馈（碎甲/扬尘/边缘光），严禁扁平2D荧光油漆刀光条、平面线框或糊块爆光遮挡攻防主体。",
            "audio_sync": "挥击、碰撞、爆炸等视觉动作峰值与音频能量峰值紧凑对齐，节奏有冲击力。",
        },
        "tolerances": [
            "允许高速挥击时的合理动态模糊（Motion Blur）、速度线、夸张透视和1-2帧冲击闪白帧（Impact Frame）。",
            "允许大招释放瞬间出现局部亮度峰值跳变，不直接按普通闪烁判死刑。",
        ],
        "veto_rules": [
            "关节反向折断、多长手臂/手指、兵器穿模或中途凭空消失。",
            "假动站桩（只有背景速度线/粒子狂闪，人物躯干完全定格）或尾段连续2秒以上明显动作停滞。",
            "高动态中段整张画面糊成无法分辨攻防双方的色块，或爆光/火焰骤然膨胀糊死人物与兵器。",
            "廉价2D荧光油漆刀光条、塑料高饱和平面激光带、平面几何线框/斑马条纹特效或无空间透视的静态贴片光效。",
        ],
    },
    "romance": {
        "id": "romance",
        "label": "言情 / 日常 / 情感对话",
        "description": "极度看重微表情、眼神光、口型同步、氛围光影与肢体轻接触自然度；允许克制的微动镜头。",
        "weights": {
            "character": 0.24,
            "motion": 0.11,
            "expression": 0.26,
            "camera_displacement": 0.10,
            "lighting_style": 0.11,
            "vfx_quality": 0.04,
            "audio_sync": 0.14,
        },
        "thresholds": {
            "static_motion_min": 0.65,
            "max_freeze_ratio": 0.65,
            "max_luma_jump": 22.0,
            "max_color_drift": 13.5,
            "max_saturation_jump": 20.0,
            "clipping_retain_ratio": 0.56,
            "allow_impact_motion_blur": False,
            "dark_luma_floor": 9.0,
            "audio_duration_delta_max_s": 0.70,
            "audio_clipping_ratio_max": 0.008,
            "audio_silence_ratio_max": 0.80,
            "audio_peak_sync_window_s": 0.50,
            "require_roi_zoom": True,
        },
        "focus_checks": {
            "character": "男女主五官比例、眼神光、发丝与服装质感始终精致稳定，绝不允许变脸或路人化。",
            "motion": "牵手、拥抱、递物、拭泪等轻接触动作物理自然，严禁身体/手指融化粘连。",
            "expression": "微表情精准传达情绪（含笑、委屈、泛泪、心动），对视镜头双方视线高度与方向对齐。",
            "camera_displacement": "镜头推进/轻微摇移柔和克制，无突兀抖动和景深乱跳。",
            "lighting_style": "逆光、夕阳、柔光滤镜与色温跨帧高度稳定，无频闪和油腻3D异变高光。",
            "vfx_quality": "氛围光斑、花瓣、雪花与丁达尔光束柔和通透并融入景深，无生硬贴图或塑料感滤镜。",
            "audio_sync": "近景/特写对白口型开合节奏贴合台词停顿，无嚼口香糖式乱动嘴；旁白内心独白闭口。",
        },
        "tolerances": [
            "允许情绪特写或日常静谧镜头保持低幅度呼吸感微动（眨眼、发丝微飘、慢推镜头）。",
        ],
        "veto_rules": [
            "情绪彻底错位（例如苦笑/悲伤渲染成诡异惊悚笑、眼神空洞翻白眼）。",
            "双人拥抱/牵手时脸部或手指融化穿模粘连。",
            "色温或光影氛围剧烈跳变（如上一秒暖黄夕阳下一秒突然变冷青暗室）。",
        ],
    },
    "suspense": {
        "id": "suspense",
        "label": "悬疑 / 惊悚 / 暗黑推理",
        "description": "重点守护关键线索道具一致、暗部细节可读、空间轴线严谨与压迫节奏稳定。",
        "weights": {
            "character": 0.20,
            "motion": 0.14,
            "expression": 0.18,
            "camera_displacement": 0.18,
            "lighting_style": 0.12,
            "vfx_quality": 0.06,
            "audio_sync": 0.12,
        },
        "thresholds": {
            "static_motion_min": 0.70,
            "max_freeze_ratio": 0.62,
            "max_luma_jump": 22.0,
            "max_color_drift": 14.0,
            "max_saturation_jump": 22.0,
            "clipping_retain_ratio": 0.50,
            "allow_impact_motion_blur": False,
            "dark_luma_floor": 4.5,
            "audio_duration_delta_max_s": 0.75,
            "audio_clipping_ratio_max": 0.01,
            "audio_silence_ratio_max": 0.85,
            "audio_peak_sync_window_s": 0.35,
            "require_roi_zoom": True,
        },
        "focus_checks": {
            "character": "角色外观与关键线索道具（信件、伤口、凶器、证物、符号）跨帧绝对一致，严禁AI私自篡改细节。",
            "motion": "潜行、回头、推门、翻找等悬疑动作节奏稳重，不能忽快忽慢或出现滑步瞬移。",
            "expression": "警觉、疑虑、恐惧、冷峻等微表情克制真实，暗光下眼睛与五官依然清晰可辨。",
            "camera_displacement": "空间布局、门窗方位与180度轴线严格连贯，缓慢推镜/摇镜平稳无畸变。",
            "lighting_style": "允许低照度与强明暗对比，但暗部须保留层次与主体轮廓，禁止塑料降噪和无故闪光。",
            "vfx_quality": "暗雾、咒纹、尘埃与冷光需具空间层次与暗部融合感，禁止浮在黑幕表层的廉价荧光线条或糊块。",
            "audio_sync": "环境静音留白合理，脚步/开门/惊雷等突发音效点与视觉反应帧精确呼应。",
        },
        "tolerances": [
            "允许低照度暗黑影调、大面积阴影留白与高对比冷峻光效，不因平均亮度偏低误判为黑屏。",
            "允许极缓慢的压迫感推进镜头与蓄力静默段。",
        ],
        "veto_rules": [
            "关键道具/证物/伤痕在同镜头内变形、移位或凭空消失。",
            "暗部糊成死黑一团导致主体不可辨认，或出现高频明暗闪烁破坏紧张气氛。",
            "空间方位逻辑冲突（无转场瞬移、门窗结构融化变形、对话越轴）。",
        ],
    },
    "fantasy": {
        "id": "fantasy",
        "label": "玄幻 / 仙侠 / 科幻机甲",
        "description": "强化世界观资产统一、UE5级3D体积特效质感、宏大远景几何稳定与遮挡层次；容忍合理能量光效波动。",
        "weights": {
            "character": 0.18,
            "motion": 0.19,
            "expression": 0.07,
            "camera_displacement": 0.14,
            "lighting_style": 0.10,
            "vfx_quality": 0.22,
            "audio_sync": 0.10,
        },
        "thresholds": {
            "static_motion_min": 1.3,
            "max_freeze_ratio": 0.40,
            "max_luma_jump": 40.0,
            "max_color_drift": 24.0,
            "max_saturation_jump": 32.0,
            "clipping_retain_ratio": 0.42,
            "allow_impact_motion_blur": True,
            "dark_luma_floor": 7.5,
            "audio_duration_delta_max_s": 0.80,
            "audio_clipping_ratio_max": 0.015,
            "audio_silence_ratio_max": 0.75,
            "audio_peak_sync_window_s": 0.35,
            "require_roi_zoom": False,
        },
        "focus_checks": {
            "character": "角色道袍/盔甲纹路、本命法宝、机甲对称结构严格统一，严禁古今/中西画风串台。",
            "motion": "御剑、结印、施法、机甲变形或巨兽移动需有重量感与连贯路径，粒子/法阵与人物遮挡关系正确，无尾段停滞定格。",
            "expression": "威严、决绝、惊骇等大开大合表情清晰，不受华丽光效过曝吞噬。",
            "camera_displacement": "航拍仙山、宫殿、星舰等宏大场景在运镜中保持建筑与山体刚性，严禁抽象塌陷融化。",
            "lighting_style": "能量光效（法阵、激光、灵气）在角色与环境上产生一致受光，不过曝成惨白死区。",
            "vfx_quality": "云海、雷霆、剑阵、灵焰、星尘与冲击波必须具备电影级3D体积层次、多尺度粒子细节、合理空气透视与地面/角色环境反光；严禁扁平2D荧光油漆刀光条、平面线框斑马柱、离地两米假棉花团云层或静态贴片光效。",
            "audio_sync": "法术蓄力/爆发或机甲机械音效与画面高潮同步，史诗配音气势连贯无断句吞字。",
        },
        "tolerances": [
            "允许大招、灵光、雷电或推进器焰光引起阶段性局部亮度上升与色彩饱和波动。",
            "宏大远景镜头允许弱化微小人物面部细节，重点审核世界观美学、特效体积感与建筑几何稳定。",
        ],
        "veto_rules": [
            "严重画风穿帮（如仙侠场景混入现代建筑/西式奇幻铠甲，或科幻机甲结构左右融化崩塌）。",
            "法宝/武器/机甲核心部件在镜头中随意变样或脱离手部乱飞。",
            "廉价2D荧光油漆能量条、平面CAD几何线框/斑马条纹光柱、或离地极低破坏比例尺的假棉花团云层。",
            "无体积感/无环境受光/无前后遮挡关系的静态贴片特效，或中段灵焰/爆光骤然膨胀糊死人物脸部与兵器。",
            "远景建筑/山峰扭曲成抽象涂鸦，或法阵范围与目标空间关系严重穿帮。",
        ],
    },
    "comedy": {
        "id": "comedy",
        "label": "搞笑 / 萌系 / Q版轻喜剧",
        "description": "强调夸张表情可读性、明快色彩稳定与吐槽节奏卡点；包容动漫夸张形变与Q版切画。",
        "weights": {
            "character": 0.21,
            "motion": 0.15,
            "expression": 0.23,
            "camera_displacement": 0.10,
            "lighting_style": 0.10,
            "vfx_quality": 0.06,
            "audio_sync": 0.15,
        },
        "thresholds": {
            "static_motion_min": 0.90,
            "max_freeze_ratio": 0.52,
            "max_luma_jump": 34.0,
            "max_color_drift": 18.0,
            "max_saturation_jump": 26.0,
            "clipping_retain_ratio": 0.45,
            "allow_impact_motion_blur": False,
            "dark_luma_floor": 10.0,
            "audio_duration_delta_max_s": 0.65,
            "audio_clipping_ratio_max": 0.012,
            "audio_silence_ratio_max": 0.72,
            "audio_peak_sync_window_s": 0.30,
            "require_roi_zoom": True,
        },
        "focus_checks": {
            "character": "角色标志性特征（呆毛、发色、服饰配色）清晰讨喜，手脚不出现恐怖谷式畸形。",
            "motion": "吐槽挥手、跌倒、惊跳等喜剧动作干脆利落有弹性，不拖泥带水。",
            "expression": "颜艺、呆滞、得意、慌张等喜剧表情夸张但清晰可爱，线条干净不脏糊。",
            "camera_displacement": "快切、震屏或定格吐槽符合喜剧节奏，不出现非预期的背景融化。",
            "lighting_style": "画面色彩明快干净、平涂/赛璐璐边界清晰，无莫名发灰发脏或油腻阴影。",
            "vfx_quality": "喜剧漫画特效（速度线、星光、汗滴、爆炸烟球）线条利落与画风统一，不出现脏糊贴图。",
            "audio_sync": "快节奏吐槽台词与口型/动作包袱落点精准对齐，尾段不留冗长尴尬空白。",
        },
        "tolerances": [
            "允许动漫喜剧惯用的夸张表情符号（汗滴、青筋、星星眼）、短暂定格吐槽与弹性变形。",
        ],
        "veto_rules": [
            "夸张表情失控变成恐怖谷畸形脸或多余五官。",
            "吐槽台词已说完画面却长时间僵死发呆，或快节奏对白明显口型错拍。",
            "画面出现乱码气泡文字或漫画分格边框。",
        ],
    },
}


def normalize_genre_id(genre: str | None) -> str:
    """Normalize Chinese/English genre labels or aliases to a canonical profile id."""
    if not genre or not isinstance(genre, str):
        return "general"
    raw = genre.strip().lower()
    if raw in _GENRE_ALIASES:
        return _GENRE_ALIASES[raw]
    for alias, target in _GENRE_ALIASES.items():
        if alias and alias in raw:
            return target
    return "general"


def get_qa_profile(genre: str | None = None) -> dict[str, Any]:
    """Return a deep copy of the canonical Anime QA Profile for `genre`."""
    gid = normalize_genre_id(genre)
    return deepcopy(ANIME_QA_PROFILES[gid])


def list_qa_profiles() -> list[dict[str, Any]]:
    """List all built-in genre QA profiles."""
    return [deepcopy(profile) for profile in ANIME_QA_PROFILES.values()]


def infer_genre_from_text(*texts: str | None) -> str:
    """Heuristically infer a genre id from story/shot text when none is explicitly set."""
    haystack = " ".join(str(t).strip() for t in texts if isinstance(t, str) and t.strip())
    if not haystack:
        return "general"
    best_id = "general"
    best_hits = 0
    for gid, keywords in _GENRE_INFERENCE_KEYWORDS:
        hits = sum(1 for kw in keywords if kw in haystack)
        if hits > best_hits:
            best_hits = hits
            best_id = gid
    return best_id if best_hits >= 1 else "general"


def resolve_shot_qa_profile(
    manifest: Any = None,
    shot: Any = None,
    *,
    genre: str | None = None,
) -> dict[str, Any]:
    """Resolve the active QA profile from explicit override -> shot -> manifest -> policy -> heuristic inference."""
    candidates: list[str] = []
    if isinstance(genre, str) and genre.strip():
        candidates.append(genre.strip())

    shot_ctx = getattr(shot, "scene_context", None) if shot is not None else None
    if isinstance(shot_ctx, dict):
        shot_genre = shot_ctx.get("genre") or shot_ctx.get("qa_genre")
        if isinstance(shot_genre, str) and shot_genre.strip():
            candidates.append(shot_genre.strip())

    if manifest is not None:
        m_genre = getattr(manifest, "genre", None)
        if isinstance(m_genre, str) and m_genre.strip() and normalize_genre_id(m_genre) != "general":
            candidates.append(m_genre.strip())
        policy = getattr(manifest, "supervisor_policy", None)
        if isinstance(policy, dict):
            p_genre = policy.get("genre") or policy.get("qa_genre")
            if isinstance(p_genre, str) and p_genre.strip():
                candidates.append(p_genre.strip())

    for cand in candidates:
        gid = normalize_genre_id(cand)
        if gid != "general" or cand.strip().lower() in {"general", "default", "通用", "剧情", "综合"}:
            profile = get_qa_profile(gid)
            profile["resolved_from"] = "explicit"
            return profile

    inferred = infer_genre_from_text(
        getattr(shot, "text", None) if shot is not None else None,
        shot_ctx.get("visual_action") if isinstance(shot_ctx, dict) else None,
        shot_ctx.get("mood") if isinstance(shot_ctx, dict) else None,
        getattr(manifest, "story", None) if manifest is not None else None,
        getattr(manifest, "style_preset", None) if manifest is not None else None,
    )
    profile = get_qa_profile(inferred)
    profile["resolved_from"] = "inferred" if inferred != "general" else "default"
    return profile


def compute_weighted_score(
    scores: dict[str, float],
    profile: dict[str, Any] | None = None,
) -> float:
    """Compute a genre-weighted overall score across available dimensions, falling back to mean."""
    if not scores:
        return 0.0
    weights = (profile or {}).get("weights") if isinstance(profile, dict) else None
    if not isinstance(weights, dict) or not weights:
        return round(sum(scores.values()) / len(scores), 4)

    # Map legacy/alias score keys onto canonical 7 dimensions when appropriate.
    alias_to_dim = {
        "character": "character",
        "character_consistency": "character",
        "motion": "motion",
        "motion_stability": "motion",
        "plot_fidelity": "motion",
        "expression": "expression",
        "camera_displacement": "camera_displacement",
        "camera": "camera_displacement",
        "lighting_style": "lighting_style",
        "visual_h3_grade": "lighting_style",
        "visual_quality": "lighting_style",
        "vfx_quality": "vfx_quality",
        "vfx": "vfx_quality",
        "effects": "vfx_quality",
        "audio_sync": "audio_sync",
        "audio": "audio_sync",
    }

    weighted_sum = 0.0
    weight_total = 0.0
    for key, val in scores.items():
        dim = alias_to_dim.get(str(key).strip().lower())
        w = float(weights.get(dim, 0.0)) if dim else 0.0
        if w > 0.0:
            weighted_sum += float(val) * w
            weight_total += w

    if weight_total <= 0.0:
        return round(sum(scores.values()) / len(scores), 4)
    return round(weighted_sum / weight_total, 4)


def evaluate_objective_qa_flags(
    diagnostics: dict[str, Any] | None,
    profile: dict[str, Any] | None = None,
    *,
    stage: str = "video",
    has_dialogue: bool = False,
) -> dict[str, Any]:
    """Evaluate objective CV/lighting/audio telemetry against genre-calibrated thresholds."""
    prof = profile if isinstance(profile, dict) else get_qa_profile("general")
    thresholds = prof.get("thresholds") or {}
    diag = diagnostics if isinstance(diagnostics, dict) else {}

    # Accept either full shot diagnostics bundle or flat video/image diagnostics dict.
    vid_diag = diag.get("video") if isinstance(diag.get("video"), dict) else (diag if stage == "video" else {})
    img_diag = diag.get("image") if isinstance(diag.get("image"), dict) else (diag if stage == "image" else {})
    aud_diag = diag.get("audio") if isinstance(diag.get("audio"), dict) else {}
    av_sync = diag.get("av_sync") if isinstance(diag.get("av_sync"), dict) else {}

    warnings: list[dict[str, Any]] = []
    hard_fails: list[dict[str, Any]] = []

    def add_flag(target: list[dict[str, Any]], code: str, dimension: str, message: str, value: Any, limit: Any) -> None:
        target.append(
            {
                "code": code,
                "dimension": dimension,
                "message": message,
                "value": value,
                "threshold": limit,
            }
        )

    dark_floor = float(thresholds.get("dark_luma_floor", 8.0))

    if img_diag.get("available") is True:
        img_luma = float(img_diag.get("mean_luma", 100.0))
        img_sharp = float(img_diag.get("sharpness", 100.0))
        if img_luma < dark_floor and img_sharp < 3.0:
            add_flag(
                hard_fails,
                "DARK_OR_BLANK_IMAGE",
                "lighting_style",
                f"首帧平均亮度 {img_luma:.1f} 低于题材底线 {dark_floor:.1f} 且清晰度过低，疑似黑帧废图",
                img_luma,
                dark_floor,
            )

    if vid_diag.get("available") is True:
        motion_diff = float(vid_diag.get("motion_mean_diff", 0.0))
        static_min = float(thresholds.get("static_motion_min", 1.2))
        if motion_diff < static_min:
            add_flag(
                warnings,
                "LOW_MOTION_STATIC",
                "motion",
                f"画面平均帧差 {motion_diff:.2f} 低于 {prof['label']} 建议下限 {static_min:.2f}，疑似静态/PPT",
                motion_diff,
                static_min,
            )

        if vid_diag.get("pose_locked_vfx_suspect") is True:
            add_flag(
                warnings,
                "POSE_LOCKED_VFX",
                "motion",
                "检测到边缘特效剧烈变化但画面中心主体几乎静止，疑似站桩假动",
                {
                    "center_motion": vid_diag.get("center_motion_mean_diff"),
                    "border_motion": vid_diag.get("border_motion_mean_diff"),
                },
                "center/border >= 0.35",
            )

        freeze_ratio = float(vid_diag.get("temporal_freeze_ratio", 0.0))
        max_freeze = float(thresholds.get("max_freeze_ratio", 0.45))
        if freeze_ratio > max_freeze:
            add_flag(
                warnings,
                "TEMPORAL_FREEZE",
                "motion",
                f"8帧密集时序冻结占比 {freeze_ratio:.2f} 高于题材上限 {max_freeze:.2f}，存在明显卡顿或中后段停滞",
                freeze_ratio,
                max_freeze,
            )

        if vid_diag.get("tail_stall_suspect") is True:
            tail_ratio = float(vid_diag.get("tail_stall_ratio", 0.0))
            add_flag(
                warnings,
                "TAIL_MOTION_STALL",
                "motion",
                f"检测到前段有动作但尾段连续停滞（尾段停滞占比 {tail_ratio:.2f}），疑似H3过长帧数导致的后段定格拖尾",
                tail_ratio,
                "tail_stall_ratio < 0.28",
            )

        vfx_metrics = vid_diag.get("vfx_metrics") if isinstance(vid_diag.get("vfx_metrics"), dict) else {}
        if vfx_metrics.get("flat_vfx_blob_suspect") is True:
            add_flag(
                warnings,
                "FLAT_2D_VFX_BLOB",
                "vfx_quality",
                f"检测到高饱和扁平发光区域（neon={vfx_metrics.get('neon_clip_ratio_max')}, detail={vfx_metrics.get('glow_micro_detail_min')}），疑似廉价2D荧光油漆光条或平面色块特效",
                {
                    "neon_clip_ratio_max": vfx_metrics.get("neon_clip_ratio_max"),
                    "glow_area_ratio_max": vfx_metrics.get("glow_area_ratio_max"),
                    "glow_micro_detail_min": vfx_metrics.get("glow_micro_detail_min"),
                },
                "volumetric_detail",
            )
        if vfx_metrics.get("vfx_static_sticker_suspect") is True:
            add_flag(
                warnings,
                "STATIC_VFX_STICKER",
                "vfx_quality",
                f"检测到亮部特效掩码跨帧几乎静止（IoU={vfx_metrics.get('glow_mask_iou_mean')}），疑似浮在画面表层的静态贴片光效",
                vfx_metrics.get("glow_mask_iou_mean"),
                "< 0.86",
            )
        if vfx_metrics.get("vfx_bloat_collapse_suspect") is True:
            add_flag(
                warnings,
                "VFX_BLOAT_COLLAPSE",
                "vfx_quality",
                "检测到中段高光/灵焰面积骤然膨胀且画面清晰度跌落，疑似特效糊块遮挡主体",
                vfx_metrics.get("glow_area_ratio_max"),
                "no_midshot_bloat_collapse",
            )

        luma_jump = float(vid_diag.get("flicker_max_luma_jump", 0.0))
        max_luma = float(thresholds.get("max_luma_jump", 30.0))
        if luma_jump > max_luma:
            add_flag(
                warnings,
                "LUMA_FLICKER",
                "lighting_style",
                f"帧间最大亮度跳变 {luma_jump:.1f} 超过题材上限 {max_luma:.1f}，疑似闪烁或光照突变",
                luma_jump,
                max_luma,
            )

        color_drift = float(vid_diag.get("color_drift_max_delta", 0.0))
        max_color = float(thresholds.get("max_color_drift", 18.0))
        if color_drift > max_color:
            add_flag(
                warnings,
                "COLOR_TEMP_DRIFT",
                "lighting_style",
                f"帧间最大色度/色温漂移 {color_drift:.1f} 超过题材上限 {max_color:.1f}，疑似氛围变天或画风漂移",
                color_drift,
                max_color,
            )

        sat_jump = float(vid_diag.get("saturation_max_jump", 0.0))
        max_sat = float(thresholds.get("max_saturation_jump", 26.0))
        if sat_jump > max_sat:
            add_flag(
                warnings,
                "STYLE_SATURATION_JUMP",
                "lighting_style",
                f"帧间饱和度跳变 {sat_jump:.1f} 超过题材上限 {max_sat:.1f}，疑似画风/上色风格失稳",
                sat_jump,
                max_sat,
            )

        if vid_diag.get("is_clipping_suspect") is True and not vid_diag.get("motion_blur_likely"):
            clip_info = vid_diag.get("clipping_suspect_frame") or {}
            retain = float(clip_info.get("sharpness_retain_ratio", 1.0))
            clip_limit = float(thresholds.get("clipping_retain_ratio", 0.45))
            clip_mode = vid_diag.get("clipping_mode") or clip_info.get("clipping_mode")
            allow_blur = bool(thresholds.get("allow_impact_motion_blur", False))
            if clip_mode == "sharp_topology_tear":
                add_flag(
                    hard_fails if not allow_blur else warnings,
                    "SHARP_TOPOLOGY_TEAR",
                    "character",
                    "视频中段检测到锐利边缘撕裂与非刚性拓扑突变，高度疑似穿模或肢体错位",
                    clip_info.get("nonrigid_center_mad"),
                    "nonrigid_tear",
                )
            elif retain < clip_limit:
                add_flag(
                    warnings,
                    "MIDSHOT_BLUR_CLIPPING",
                    "motion",
                    f"视频中段清晰度保留率 {retain:.2f} 低于题材阈值 {clip_limit:.2f}，疑似重影或糊帧崩坏",
                    retain,
                    clip_limit,
                )

    if aud_diag.get("available") is True:
        clip_ratio = float(aud_diag.get("clipping_ratio", 0.0))
        clip_max = float(thresholds.get("audio_clipping_ratio_max", 0.01))
        if clip_ratio > clip_max:
            add_flag(
                warnings,
                "AUDIO_CLIPPING",
                "audio_sync",
                f"音频削波爆音占比 {clip_ratio:.3f} 超过上限 {clip_max:.3f}",
                clip_ratio,
                clip_max,
            )

        sil_ratio = float(aud_diag.get("silence_ratio", 0.0))
        sil_max = float(thresholds.get("audio_silence_ratio_max", 0.75))
        if has_dialogue and sil_ratio > sil_max:
            add_flag(
                warnings,
                "AUDIO_EXCESSIVE_SILENCE",
                "audio_sync",
                f"对白镜头音频静音占比 {sil_ratio:.2f} 高于上限 {sil_max:.2f}，疑似漏配音或尾段过长空白",
                sil_ratio,
                sil_max,
            )

    if av_sync.get("available") is True:
        dur_delta = float(av_sync.get("duration_delta_seconds", 0.0))
        dur_max = float(thresholds.get("audio_duration_delta_max_s", 0.85))
        if dur_delta > dur_max:
            add_flag(
                warnings,
                "AV_DURATION_MISMATCH",
                "audio_sync",
                f"音画时长差 {dur_delta:.2f}s 超过题材上限 {dur_max:.2f}s，容易导致声画脱节或尾帧发呆",
                dur_delta,
                dur_max,
            )

        nearest_onset = av_sync.get("nearest_onset_delta_seconds")
        peak_window = float(thresholds.get("audio_peak_sync_window_s", 0.45))
        if (
            prof["id"] == "action"
            and isinstance(nearest_onset, (int, float))
            and float(nearest_onset) > peak_window
        ):
            add_flag(
                warnings,
                "ACTION_IMPACT_AUDIO_OFFSET",
                "audio_sync",
                f"动作高潮帧与最近音频能量峰值偏差 {float(nearest_onset):.2f}s（动作题材建议 <= {peak_window:.2f}s）",
                float(nearest_onset),
                peak_window,
            )

    action_hard = should_enforce_action_motion_fails(prof)
    if action_hard:
        remaining_warnings: list[dict[str, Any]] = []
        for item in warnings:
            if item.get("code") in ACTION_MOTION_HARD_CODES:
                hard_fails.append(item)
            else:
                remaining_warnings.append(item)
        warnings = remaining_warnings

    status = "failed" if hard_fails else ("warning" if warnings else "passed")
    return {
        "schema": "objective-qa/v1",
        "genre": prof["id"],
        "genre_label": prof["label"],
        "status": status,
        "hard_fail_count": len(hard_fails),
        "warning_count": len(warnings),
        "hard_fails": hard_fails,
        "warnings": warnings,
        "action_motion_hard_gates": action_hard,
    }


def compute_objective_audio_sync_score(
    diagnostics: dict[str, Any] | None,
    profile: dict[str, Any] | None = None,
    *,
    has_dialogue: bool = False,
) -> float | None:
    """Compute a deterministic [0.15, 0.96] audio_sync baseline from objective audio & av_sync telemetry."""
    if not isinstance(diagnostics, dict):
        return None
    prof = profile if isinstance(profile, dict) else get_qa_profile("general")
    thresholds = prof.get("thresholds") or {}
    aud_diag = diagnostics.get("audio") if isinstance(diagnostics.get("audio"), dict) else {}
    av_sync = diagnostics.get("av_sync") if isinstance(diagnostics.get("av_sync"), dict) else {}

    has_aud = aud_diag.get("available") is True
    has_av = av_sync.get("available") is True
    if not has_aud and not has_av:
        return None

    score = 0.94 if (has_aud and has_av) else 0.86

    if has_av:
        dur_delta = float(av_sync.get("duration_delta_seconds", 0.0))
        dur_max = float(thresholds.get("audio_duration_delta_max_s", 0.80))
        if dur_delta > dur_max:
            excess = dur_delta - dur_max
            score -= min(0.55, 0.22 + 0.16 * excess)
        elif dur_delta > dur_max * 0.65:
            score -= 0.05

        nearest_onset = av_sync.get("nearest_onset_delta_seconds")
        peak_window = float(thresholds.get("audio_peak_sync_window_s", 0.45))
        if prof.get("id") == "action" and isinstance(nearest_onset, (int, float)):
            if float(nearest_onset) > peak_window:
                score -= min(0.16, 0.08 + 0.10 * (float(nearest_onset) - peak_window))

    if has_aud:
        clip_ratio = float(aud_diag.get("clipping_ratio", 0.0))
        clip_max = float(thresholds.get("audio_clipping_ratio_max", 0.01))
        if clip_ratio > clip_max:
            score -= min(0.30, 0.15 + clip_ratio * 2.0)

        sil_ratio = float(aud_diag.get("silence_ratio", 0.0))
        sil_max = float(thresholds.get("audio_silence_ratio_max", 0.75))
        if has_dialogue and sil_ratio > sil_max:
            score -= min(0.35, 0.18 + (sil_ratio - sil_max))

    return round(max(0.15, min(0.96, score)), 4)


def build_vlm_qa_rubric_prompt(
    profile: dict[str, Any],
    *,
    stage: str = "video",
    shot: Any = None,
    diagnostics: dict[str, Any] | None = None,
    character_bible: dict[str, Any] | None = None,
) -> str:
    """Build a concise, genre-tailored Chinese QA rubric block for VLM/Supervisor prompting."""
    prof = profile if isinstance(profile, dict) else get_qa_profile("general")
    weights = prof.get("weights") or {}
    checks = prof.get("focus_checks") or {}
    tolerances = prof.get("tolerances") or []
    veto_rules = prof.get("veto_rules") or []

    dim_lines: list[str] = []
    for dim in QA_DIMENSIONS:
        label = QA_DIMENSION_LABELS[dim]
        pct = int(round(float(weights.get(dim, 0.0)) * 100))
        desc = checks.get(dim, "")
        dim_lines.append(f"- {label}（{dim}，权重 {pct}%）：{desc}")

    lines: list[str] = [
        f"【题材质检模板：{prof.get('label')}（{prof.get('id')}）| 阶段：{stage}】",
        f"题材说明：{prof.get('description', '')}",
        "七大专项核查要求（含特效质感专项）：",
        *dim_lines,
    ]
    if tolerances:
        lines.append("允许项（切勿误杀）：" + "；".join(str(t) for t in tolerances))
    if veto_rules:
        lines.append("一票否决红线：" + "；".join(str(v) for v in veto_rules))

    delivery = "silent"
    if shot is not None:
        from .lipsync import classify_shot_speech_delivery

        delivery = classify_shot_speech_delivery(shot, character_bible)
        shot_ctx = getattr(shot, "scene_context", None) if isinstance(getattr(shot, "scene_context", None), dict) else {}
        framing_txt = str(shot_ctx.get("framing") or "")
        is_wide_or_action_dialogue = bool(
            shot_ctx.get("action_beat")
            or any(tok in framing_txt for tok in ("远景", "大远景", "全景", "俯瞰", "群像", "背影"))
        )
        if delivery == "on_screen_dialogue":
            if is_wide_or_action_dialogue:
                lines.append(
                    "【发声类型锁定：远景/动作对白 on_screen_dialogue】口型专项：本镜为角色正片对白（绝非旁白），允许人物开口说话；但因景别/动作原因不苛求微距逐字口型开合，请重点审核动作连贯、特效质感与音画时长匹配。"
                )
            else:
                lines.append(
                    "【发声类型锁定：正脸对白 on_screen_dialogue】口型专项：本镜已确认为角色露脸对白（绝非旁白/画外音，人物自然张嘴说话完全正确，严禁误判为旁白张嘴）；请重点结合局部放大图核查嘴唇开合是否自然，禁止全程闭嘴发声或嚼口香糖式乱动。"
                )
        elif delivery == "voiceover":
            lines.append(
                "【发声类型锁定：旁白/画外音/独白 voiceover】口型专项：当前为旁白/画外音/内心独白镜头，画面内人物（及动物）必须自然闭口，严禁无故张嘴说话。"
            )
        else:
            lines.append("【发声类型锁定：静默/无对白 silent】口型专项：本镜无对白，画面内人物保持自然闭口即可。")

    if isinstance(diagnostics, dict):
        has_dlg = delivery == "on_screen_dialogue"
        obj_qa = diagnostics.get("objective_qa")
        if not isinstance(obj_qa, dict) and stage == "video":
            obj_qa = evaluate_objective_qa_flags(
                diagnostics,
                prof,
                stage=stage,
                has_dialogue=has_dlg,
            )
        obj_aud_score = compute_objective_audio_sync_score(diagnostics, prof, has_dialogue=has_dlg)
        av_sync = diagnostics.get("av_sync") if isinstance(diagnostics.get("av_sync"), dict) else {}
        aud_diag = diagnostics.get("audio") if isinstance(diagnostics.get("audio"), dict) else {}
        if obj_aud_score is not None:
            av_d = av_sync.get("duration_delta_seconds")
            clip_r = aud_diag.get("clipping_ratio")
            lines.append(
                f"【客观音画同步核验】已由本地音轨实测：av_delta={av_d}s，clipping={clip_r}，客观音频同步基准分={obj_aud_score:.2f}。"
                "若视觉口型无违规，audio_sync 请以此客观基准分为准，切勿因无法直接听声音而盲填 0.50。"
            )
        if isinstance(obj_qa, dict):
            flagged = list(obj_qa.get("hard_fails") or []) + list(obj_qa.get("warnings") or [])
            if flagged:
                flag_text = "；".join(f"[{item['code']}] {item['message']}" for item in flagged[:5])
                lines.append(f"客观抽帧/音画预警（请重点复核）：{flag_text}")

    try:
        from .vfx_spatial_standard import rubric_appendix_for_shot

        vfx_appendix = rubric_appendix_for_shot(shot)
        if vfx_appendix:
            lines.append(vfx_appendix)
    except ImportError:
        pass

    return "\n".join(lines)
