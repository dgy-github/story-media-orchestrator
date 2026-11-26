"""Pre-production script detail audit gate & video-failure -> upstream script-beat patch loop.

Codifies the 6 core script-level failure modes distilled from serialized drama production:
1. Abstract interior emotion/monologue without visible physical action or prop handoff.
2. Multi-action overload crammed into a single shot instead of one beat per shot.
3. Vague combat/VFX descriptions ("full-screen glow / huge explosion") lacking direction, contact, or displacement.
4. Peak climax pose written as the opening frame instead of pre-action anticipation.
5. Missing scene continuity anchors (framing, location, visual_action, stage continuity).
6. Speech/dialogue duration overload or raw quoted dialogue leaked into visual action.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

from .manifest import ProjectManifest, Shot
from .supervisor import verify_takeover

SCRIPT_AUDIT_SCHEMA = "script-audit-report/v1"
SCRIPT_REHEARSAL_SCHEMA = "script-rehearsal-report/v1"

GENRE_CONTENT_CHECKLISTS: dict[str, list[dict[str, Any]]] = {
    "xianxia": [
        {
            "id": "human_grounding_or_cost",
            "label": "人间烟火 / 选择与代价",
            "pattern": r"(村|寨|家|灶|灯|水|干粮|行囊|娘|兄|妹|百姓|代价|舍弃|留下|承诺|遗物|护)",
            "missing_advice": "仙侠剧大战前缺少人间线或情感代价铺垫，建议用具体生活细节（如分水、递干粮、交托信物）插 1 镜因果桥，让观众明白为何出手。",
            "recommended_tool": "story_insert_bridge_shot",
        },
        {
            "id": "combat_spatial_causality",
            "label": "斗法起势-接触-受力空间因果",
            "pattern": r"(起势|拔剑|劈向|击中|挡下|后退|碎裂|尘土|火花|涟漪|崩塌|划破|震碎)",
            "missing_advice": "缺少可见的攻防因果链（威胁来源、接触点、位移或材质破损），容易拍成站桩对砍或全屏亮光，建议拆成起势/接触/余波三拍。",
            "recommended_tool": "story_patch_script_beat",
        },
        {
            "id": "stage_identity_and_prop_lock",
            "label": "阶段发色/服饰/法器归属锁定",
            "pattern": r"(发|袍|衣|剑|戟|枪|刀|符|灯|玉|阶段|残破|血迹|白发|黑发)",
            "missing_advice": "剧本未明确标注当前阶段的角色外貌或法器形态，生图阶段极易串戏，建议在 continuity 字段写清阶段母版与持器归属。",
            "recommended_tool": "story_patch_script_beat",
        },
        {
            "id": "aftermath_and_breath",
            "label": "战后余波与情绪落点",
            "pattern": r"(余波|烟尘|风声|静|抬头|望向|转身|远去|微光|新芽|灶火|沉默|喘息)",
            "missing_advice": "高潮后缺少留白与余波反应镜，成片节奏会显得仓促断裂，建议在结尾前插入 1 镜反应或环境余波。",
            "recommended_tool": "story_insert_bridge_shot",
        },
    ],
    "urban_suspense": [
        {
            "id": "opening_hook",
            "label": "开场异常钩子",
            "pattern": r"(血|失踪|敲门|监控|电话|纸条|倒计时|脚步声|暗处|陌生|密码|钥匙|锁)",
            "missing_advice": "悬疑题材前几镜缺少具体异常钩子或物理线索物证，建议在开场加入特写物证或异常声响反应镜。",
            "recommended_tool": "story_insert_bridge_shot",
        },
        {
            "id": "clue_prop_and_reaction",
            "label": "物证交互与微表情信息差",
            "pattern": r"(递|翻开|捏紧|停顿|眼神|瞥|躲闪|屏幕|照片|文件|指纹|录音)",
            "missing_advice": "缺少人物与线索道具的物理交互或微表情停顿，建议把口头交代改成特写查验动作与眼神反应拍。",
            "recommended_tool": "story_patch_script_beat",
        },
    ],
    "romance": [
        {
            "id": "tactile_emotion_beat",
            "label": "克制肢体动作与道具传情",
            "pattern": r"(递|撑伞|披|拉住|松开|指尖|擦|避开视线|停住|回头|微红|轻声)",
            "missing_advice": "情感戏偏抽象说明，缺少具体肢体距离变化或小道具传情（如撑伞倾斜、递热饮、指尖停顿），建议把心理描写具象化。",
            "recommended_tool": "story_patch_script_beat",
        },
        {
            "id": "reaction_pause",
            "label": "反应拍与情绪留白",
            "pattern": r"(沉默|愣住|微顿|垂眸|抬头|笑|叹息|转身)",
            "missing_advice": "对白之间缺少反应拍与情绪留白，剪辑容易显得赶场，建议在关键台词后插入 1 镜近景反应镜。",
            "recommended_tool": "story_insert_bridge_shot",
        },
    ],
    "general": [
        {
            "id": "visible_goal_and_choice",
            "label": "可见目标与行动选择",
            "pattern": r"(推|拉|握|拔|递|塞|抬|转身|迈步|冲|挡|接|举起|放下|走向|打开)",
            "missing_advice": "整体剧本偏静态描述，缺少推动剧情的明确物理选择动作，建议为关键场次补足主动行为。",
            "recommended_tool": "story_patch_script_beat",
        },
        {
            "id": "reaction_and_aftermath",
            "label": "反应拍与结果余波",
            "pattern": r"(愣住|抬头|望向|后退|沉默|转身|碎裂|落地|停下|喘息)",
            "missing_advice": "缺少动作结果后的角色反应或环境余波镜，建议在转折点后插入因果桥/反应镜。",
            "recommended_tool": "story_insert_bridge_shot",
        },
    ],
}

_ABSTRACT_EMOTION_RE = re.compile(
    r"(感到|觉得|内心|心中|心想|暗自|下定决心|悲痛|绝望|震撼|复杂的情绪|百感交集|压迫感|明白了一切|意识到)"
)
_CONCRETE_ACTION_RE = re.compile(
    r"(推|拉|握|拔|劈|斩|递|塞|抬|转身|迈步|跪|倒|冲|挡|接|扔|捧|缝|倒水|点亮|挥|落|碎|裂|退|跃|扶|抱|伸手|攥紧|举起|放下)"
)
_VAGUE_VFX_RE = re.compile(
    r"(全屏亮光|光芒大盛|剧烈爆炸|轰然炸开|耀眼白光|漫天光效|能量爆发|神光冲天|瞬间吞噬)"
)
_SPATIAL_CAUSAL_RE = re.compile(
    r"(从|向|朝|劈向|击中|碎裂|掀翻|后退|尘土|火花|涟漪|倒飞|崩塌|划破|阴影|反光|震碎|穿透)"
)
_PEAK_CLIMAX_RE = re.compile(
    r"(已经倒地|已被击飞|炸成粉碎|化作飞灰|彻底崩塌|剑已刺入|死死抵住|僵持不下)"
)
_MULTI_ACTION_CONNECTORS_RE = re.compile(r"(然后|接着|随后|紧接着|同时又|最后再|并且又)")
_RAW_QUOTED_DIALOGUE_RE = re.compile(r"[“\"「『][^”\"」』]{4,}[”\"」』]")


def _estimate_zh_speech_seconds(text: str) -> float:
    clean = re.sub(r"\s+", "", text or "")
    if not clean:
        return 0.0
    # ~4.2 spoken CJK chars/sec comfortable anime pacing
    return round(len(clean) / 4.2, 2)


def audit_shot_script(shot: Shot, *, index: int = 1) -> dict[str, Any]:
    """Audit a single shot's script beat for narrative flesh, visual clarity, and shootability."""
    findings: list[dict[str, Any]] = []
    ctx = shot.scene_context if isinstance(shot.scene_context, dict) else {}
    text = str(shot.text or "").strip()
    visual_action = str(ctx.get("visual_action") or "").strip()
    combined = f"{text} {visual_action}".strip()
    framing = str(ctx.get("framing") or "").strip()
    location = str(ctx.get("location") or "").strip()

    if len(combined) < 8:
        findings.append(
            {
                "severity": "error",
                "code": "EMPTY_OR_THIN_SCRIPT",
                "message": "镜头剧本过于空洞（<8 字），缺少可见主体、动作与结果",
                "fix_hint": "补充具体起势姿态、唯一主动作、可见结果与情绪载体道具。",
            }
        )

    # 1. Abstract emotion without concrete physical action
    if _ABSTRACT_EMOTION_RE.search(combined) and not _CONCRETE_ACTION_RE.search(combined):
        findings.append(
            {
                "severity": "error",
                "code": "ABSTRACT_INTERIOR_WITHOUT_ACTION",
                "message": "出现抽象心理/情绪总结，但缺少镜头可拍的物理行为或道具交接",
                "fix_hint": "把内心独白改成可见动作（如攥紧衣角、递出干粮、抬头避开视线、按住剑柄）。",
            }
        )

    # 2. Multi-action overload in one shot
    clause_count = len([c for c in re.split(r"[，。；、,.;!！?？]+", text) if len(c.strip()) >= 4])
    connector_hits = len(_MULTI_ACTION_CONNECTORS_RE.findall(text))
    if connector_hits >= 2 or clause_count >= 5:
        findings.append(
            {
                "severity": "warning",
                "code": "MULTI_ACTION_OVERLOAD",
                "message": f"单镜塞入过多连续动作（{clause_count} 个分句 / {connector_hits} 个顺序连接词），视频模型极易丢动作或站桩",
                "fix_hint": "按「起势 / 接触 / 受力余波」拆成多镜，使用 insert_bridge_shot 保持一镜一拍。",
            }
        )

    # 3. Vague combat / VFX without spatial causality
    if _VAGUE_VFX_RE.search(combined) and not _SPATIAL_CAUSAL_RE.search(combined):
        findings.append(
            {
                "severity": "error",
                "code": "VAGUE_COMBAT_OR_VFX",
                "message": "战斗/特效依赖全屏亮光或抽象爆炸词，缺少来源、运动方向、接触点与受光/位移结果",
                "fix_hint": "写清威胁从哪来、击中何处、环境材质如何受光与形变，禁止靠全屏白光交代结果。",
            }
        )

    # 4. Peak climax written as opening frame
    if _PEAK_CLIMAX_RE.search(combined):
        findings.append(
            {
                "severity": "warning",
                "code": "PEAK_POSE_AS_START_FRAME",
                "message": "剧本描述了已完成的峰值/僵持态，直接当首帧易导致视频锁死不动",
                "fix_hint": "将首帧前移到动作发生前的起势态，把峰值留作验片对照或独立结果镜。",
            }
        )

    # 5. Missing scene continuity anchors
    missing_anchors = [k for k, v in (("framing", framing), ("location", location)) if not v]
    if missing_anchors:
        findings.append(
            {
                "severity": "warning",
                "code": "MISSING_SCENE_ANCHORS",
                "message": f"scene_context 缺少关键空间锚点：{', '.join(missing_anchors)}",
                "fix_hint": "补齐景别（framing）与场景位置（location），必要时注明左右站位与阶段连续性（continuity）。",
            }
        )

    # 6. Speech pacing / raw dialogue leaked into visual action
    spoken_parts: list[str] = []
    if shot.narration:
        spoken_parts.append(str(shot.narration))
    for dl in shot.dialogue_lines or []:
        if isinstance(dl, dict) and dl.get("text"):
            spoken_parts.append(str(dl["text"]))
        elif isinstance(dl, str) and dl.strip():
            spoken_parts.append(dl.strip())
    spoken_text = "".join(spoken_parts)
    est_sec = _estimate_zh_speech_seconds(spoken_text)
    if est_sec > 0 and shot.duration > 0 and est_sec > float(shot.duration) * 1.15:
        findings.append(
            {
                "severity": "warning",
                "code": "SPEECH_DURATION_MISMATCH",
                "message": f"台词/旁白预计语速需 {est_sec}s，超出当前镜长 {shot.duration}s，易导致抢嘴或尾停",
                "fix_hint": "精简台词、拆分成两镜，或将镜长调整至匹配语音时长。",
            }
        )
    if visual_action and _RAW_QUOTED_DIALOGUE_RE.search(visual_action):
        findings.append(
            {
                "severity": "error",
                "code": "RAW_DIALOGUE_IN_VISUAL_ACTION",
                "message": "visual_action 中包含引号台词原文，极易诱发画面伪字幕或旁白张嘴",
                "fix_hint": "将对白移至 dialogue_lines / narration，visual_action 只保留纯视觉动作与表情。",
            }
        )

    error_count = sum(1 for f in findings if f["severity"] == "error")
    warn_count = sum(1 for f in findings if f["severity"] == "warning")
    score = max(0, 100 - error_count * 25 - warn_count * 10)
    return {
        "shot_id": shot.id,
        "index": index,
        "ok": error_count == 0,
        "score": score,
        "error_count": error_count,
        "warning_count": warn_count,
        "findings": findings,
    }


def evaluate_rework_escalation_to_script(shot: Shot) -> dict[str, Any]:
    """Decide whether a failing shot should rewind upstream to script-beat revision instead of another video rerender."""
    qr = shot.quality_report if isinstance(shot.quality_report, dict) else {}
    triage = str(qr.get("triage_category") or "").strip().lower()
    failure_code = str(qr.get("failure_code") or "NONE").strip().upper()
    decision = str(qr.get("decision") or "").strip().lower()
    rework_rounds = len(shot.rework_history or [])

    reasons: list[str] = []
    if triage in {"script_beat", "motion_lock"}:
        reasons.append(f"triage_category={triage}")
    if decision == "failed" and failure_code in {"VIDEO_MOTION_FAIL", "IMAGE_PROMPT_DRIFT"} and rework_rounds >= 2:
        reasons.append(f"repeated_{failure_code.lower()}_after_{rework_rounds}_reworks")

    return {
        "shot_id": shot.id,
        "should_rewind_to_script": bool(reasons),
        "reasons": reasons,
        "recommended_tool": (
            "story_insert_bridge_shot"
            if triage == "script_beat"
            else "story_patch_script_beat"
        ),
        "rework_rounds": rework_rounds,
        "last_failure_code": failure_code,
    }


def audit_project_script(project_dir: str | Path) -> dict[str, Any]:
    """Run deterministic pre-production script detail audit across all shots + build LLM review prompt."""
    root = Path(project_dir).resolve()
    manifest_path = root / "project.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"project.json not found in {root}")
    manifest = ProjectManifest.load(manifest_path)

    per_shot: list[dict[str, Any]] = []
    blocking_shots: list[str] = []
    warning_shots: list[str] = []
    rewind_candidates: list[dict[str, Any]] = []

    for idx, shot in enumerate(manifest.shots, 1):
        res = audit_shot_script(shot, index=idx)
        per_shot.append(res)
        if not res["ok"]:
            blocking_shots.append(shot.id)
        elif res["warning_count"] > 0:
            warning_shots.append(shot.id)
        esc = evaluate_rework_escalation_to_script(shot)
        if esc["should_rewind_to_script"]:
            rewind_candidates.append(esc)

    avg_score = round(sum(r["score"] for r in per_shot) / len(per_shot), 1) if per_shot else 0.0
    review_prompt = (
        f"请作为资深短剧编剧与导演监督，审核工程 {manifest.project_id}（共 {len(manifest.shots)} 镜）的剧本细节与因果血肉。\n"
        "请逐场核查以下 6 项核心指标，若发现漏洞，直接调用 story_patch_script_beat 修订单镜剧本，或调用 story_insert_bridge_shot 插入因果桥/反应镜：\n"
        "1. 人物选择与代价：每场是否答清「谁想要什么、谁阻止、谁承担代价」？\n"
        "2. 具象血肉：是否把抽象内心独白改成了可见的物理行为、眼神细节或道具交接？\n"
        "3. 一镜一拍：是否避免单镜塞入过多动作？动作首帧是否停在起势态而非峰值僵持态？\n"
        "4. 战斗/特效因果：是否写清威胁来源、运动方向、接触点、材质受光与空间位移（禁用全屏白光敷衍）？\n"
        "5. 空间与阶段连续性：发色、服饰、兵器归属、左右站位和出入画方向是否前后一致？\n"
        "6. 对白与画外音边界：台词长度是否匹配镜长，画面描述是否已剥离引号台词原文？"
    )
    return {
        "schema": SCRIPT_AUDIT_SCHEMA,
        "project_dir": str(root),
        "project_id": manifest.project_id,
        "ok": len(blocking_shots) == 0,
        "average_score": avg_score,
        "total_shots": len(manifest.shots),
        "blocking_shots": blocking_shots,
        "warning_shots": warning_shots,
        "video_to_script_rewind_candidates": rewind_candidates,
        "shots": per_shot,
        "llm_review_prompt": review_prompt,
    }


def patch_shot_script_beat(
    project_dir: str | Path,
    shot_id: str,
    *,
    new_text: str | None = None,
    visual_action: str | None = None,
    kinetic_action: str | None = None,
    framing: str | None = None,
    location: str | None = None,
    continuity: str | None = None,
    narration: str | None = None,
    duration: float | None = None,
    invalidate_audio: bool = False,
    reason: str = "",
    supervisor_id: str = "script-supervisor",
    lease_token: str | None = None,
) -> dict[str, Any]:
    """Revise a shot's upstream script beat and surgically invalidate only stale downstream visual assets."""
    root = Path(project_dir).resolve()
    manifest_path = root / "project.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"project.json not found in {root}")
    verify_takeover(root, lease_token, allowed_roles=("critic",))

    manifest = ProjectManifest.load(manifest_path)
    shot = manifest.shot(shot_id)

    before = {
        "text": shot.text,
        "visual_action": (shot.scene_context or {}).get("visual_action"),
        "framing": (shot.scene_context or {}).get("framing"),
        "location": (shot.scene_context or {}).get("location"),
    }

    if new_text is not None and new_text.strip():
        shot.text = new_text.strip()
    if not isinstance(shot.scene_context, dict):
        shot.scene_context = {}
    if visual_action is not None:
        shot.scene_context["visual_action"] = visual_action.strip()
    if kinetic_action is not None:
        shot.scene_context["kinetic_action"] = kinetic_action.strip()
    if framing is not None:
        shot.scene_context["framing"] = framing.strip()
    if location is not None:
        shot.scene_context["location"] = location.strip()
    if continuity is not None:
        shot.scene_context["continuity"] = continuity.strip()
    if narration is not None:
        shot.narration = narration.strip()
        invalidate_audio = True
    if duration is not None and float(duration) > 0:
        shot.duration = float(duration)

    # Surgically invalidate stale downstream visual assets & cache key; keep unaffected shots intact.
    invalidated_assets: list[str] = []
    for role in ("image", "last_frame", "video"):
        if role in shot.assets:
            shot.assets.pop(role, None)
            invalidated_assets.append(role)
    if invalidate_audio and "audio" in shot.assets:
        shot.assets.pop("audio", None)
        invalidated_assets.append("audio")

    shot.cache_key = None
    shot.image_task.clear()
    shot.status = "pending"
    shot.quality_report = {}
    manifest.output = None
    manifest.approvals.pop("assets", None)

    revision_entry = {
        "strategy": "patch_script_beat",
        "supervisor_id": supervisor_id,
        "reason": reason or "upstream script beat revision",
        "before": before,
        "invalidated_assets": invalidated_assets,
        "patched_at": datetime.now(timezone.utc).isoformat(),
    }
    shot.rework_history.append(revision_entry)
    manifest.save(manifest_path)

    post_audit = audit_shot_script(shot)
    return {
        "schema": "script-beat-patch/v1",
        "project_dir": str(root),
        "shot_id": shot.id,
        "patched": True,
        "invalidated_assets": invalidated_assets,
        "post_audit": post_audit,
        "next_steps": [
            "重新生成或通过 story_inject_external_asset 注入该镜 16:9 起手首帧",
            "按需渲染该镜视频并用 story_inspect_shot_multimodal 复核",
        ],
    }


def insert_script_bridge_shot(
    project_dir: str | Path,
    insert_after_shot_id: str,
    *,
    text: str,
    visual_action: str = "",
    framing: str = "中景",
    location: str = "",
    continuity: str = "",
    story_role: str = "causal_bridge",
    duration: float = 3.5,
    new_shot_id: str | None = None,
    supervisor_id: str = "script-supervisor",
    lease_token: str | None = None,
) -> dict[str, Any]:
    """Insert an anchored causal-bridge / reaction / handoff shot without disturbing existing shot IDs or caches."""
    if not (text or "").strip():
        raise ValueError("bridge shot text must not be empty")
    root = Path(project_dir).resolve()
    manifest_path = root / "project.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"project.json not found in {root}")
    verify_takeover(root, lease_token, allowed_roles=("critic",))

    manifest = ProjectManifest.load(manifest_path)
    anchor_idx = next((i for i, s in enumerate(manifest.shots) if s.id == insert_after_shot_id), None)
    if anchor_idx is None:
        raise KeyError(f"anchor shot not found: {insert_after_shot_id}")
    anchor_shot = manifest.shots[anchor_idx]

    existing_ids = {s.id for s in manifest.shots}
    if new_shot_id and new_shot_id.strip():
        cand_id = new_shot_id.strip()
        if cand_id in existing_ids:
            raise ValueError(f"shot_id already exists: {cand_id}")
    else:
        suffix_idx = 1
        while True:
            cand_id = f"{insert_after_shot_id}-b{suffix_idx}"
            if cand_id not in existing_ids:
                break
            suffix_idx += 1

    anchor_ctx = anchor_shot.scene_context if isinstance(anchor_shot.scene_context, dict) else {}
    new_ctx: dict[str, Any] = {
        "visual_action": (visual_action or text).strip(),
        "framing": framing.strip() or str(anchor_ctx.get("framing") or "中景"),
        "location": location.strip() or str(anchor_ctx.get("location") or ""),
        "continuity": continuity.strip() or f"紧接 {insert_after_shot_id}",
        "story_role": story_role.strip() or "causal_bridge",
        "insert_after": insert_after_shot_id,
    }
    bridge_shot = Shot(
        id=cand_id,
        text=text.strip(),
        duration=max(1.5, float(duration)),
        scene_id=anchor_shot.scene_id,
        scene_context=new_ctx,
        character_locks=list(anchor_shot.character_locks or []),
    )
    bridge_shot.rework_history.append(
        {
            "strategy": "insert_bridge_shot",
            "insert_after": insert_after_shot_id,
            "story_role": story_role,
            "supervisor_id": supervisor_id,
            "inserted_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    manifest.shots.insert(anchor_idx + 1, bridge_shot)
    manifest.output = None
    manifest.approvals.pop("storyboard", None)
    manifest.approvals.pop("assets", None)
    manifest.save(manifest_path)

    post_audit = audit_shot_script(bridge_shot, index=anchor_idx + 2)
    return {
        "schema": "script-bridge-insert/v1",
        "project_dir": str(root),
        "inserted_shot_id": cand_id,
        "insert_after": insert_after_shot_id,
        "total_shots": len(manifest.shots),
        "post_audit": post_audit,
    }


def _normalize_genre_key(raw_genre: str | None) -> str:
    g = (raw_genre or "").strip().lower()
    if any(k in g for k in ("xianxia", "xuanhuan", "wuxia", "仙侠", "玄幻", "修真", "武侠", "古风")):
        return "xianxia"
    if any(k in g for k in ("suspense", "mystery", "thriller", "悬疑", "推理", "惊悚", "刑侦")):
        return "urban_suspense"
    if any(k in g for k in ("romance", "love", "言情", "恋爱", "都市情感", "甜宠")):
        return "romance"
    return "general"


def rehearse_project_script(
    project_dir: str | Path,
    *,
    genre: str | None = None,
    server_flywheel_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Pre-production quality rehearsal combining script detail audit, genre content gaps, and historical demo flywheel lessons."""
    from .flywheel import load_active_playbook

    root = Path(project_dir).resolve()
    manifest_path = root / "project.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"project.json not found in {root}")
    manifest = ProjectManifest.load(manifest_path)

    # 1) Deterministic per-shot shootability audit
    shot_audit = audit_project_script(root)

    # 2) Resolve genre & analyze episode-level missing content elements
    effective_genre_raw = genre or getattr(manifest, "genre", None) or "general"
    genre_key = _normalize_genre_key(effective_genre_raw)
    checklist = GENRE_CONTENT_CHECKLISTS.get(genre_key, GENRE_CONTENT_CHECKLISTS["general"])

    full_script_blob = "\n".join(
        f"{s.text} {(s.scene_context or {}).get('visual_action', '')} {(s.scene_context or {}).get('continuity', '')}"
        for s in manifest.shots
    )
    genre_gaps: list[dict[str, Any]] = []
    genre_hits: list[str] = []
    for item in checklist:
        if re.search(item["pattern"], full_script_blob):
            genre_hits.append(item["id"])
        else:
            genre_gaps.append(
                {
                    "gap_id": item["id"],
                    "label": item["label"],
                    "severity": "warning",
                    "advice": item["missing_advice"],
                    "recommended_tool": item["recommended_tool"],
                }
            )

    # 3) Distill lessons from past rendered demos in Quality Flywheel (.flywheel/playbook.json)
    playbook = load_active_playbook(manifest=manifest, project_root=root, explicit_dir=server_flywheel_dir) or {}
    demo_lessons: list[dict[str, Any]] = []
    for rule in (playbook.get("rules") or [])[:6]:
        if not isinstance(rule, dict):
            continue
        demo_lessons.append(
            {
                "source": "flywheel_rule",
                "failure_code": rule.get("failure_code", "OTHER"),
                "pattern": rule.get("title") or rule.get("rule") or rule.get("advice") or "",
                "action": rule.get("action") or rule.get("negative_addition") or "",
            }
        )
    for ex in (playbook.get("exemplars") or [])[:4]:
        if not isinstance(ex, dict):
            continue
        demo_lessons.append(
            {
                "source": "verified_demo_exemplar",
                "failure_code": ex.get("failure_code", "OTHER"),
                "pattern": ex.get("defect") or ex.get("reason") or "",
                "action": ex.get("fix") or ex.get("strategy") or "",
            }
        )

    # 4) Build prioritized repair suggestions (executable via MCP / CLI)
    repair_suggestions: list[dict[str, Any]] = []
    for s_res in shot_audit["shots"]:
        for finding in s_res["findings"]:
            tool = (
                "story_insert_bridge_shot"
                if finding["code"] == "MULTI_ACTION_OVERLOAD"
                else "story_patch_script_beat"
            )
            repair_suggestions.append(
                {
                    "priority": "P0" if finding["severity"] == "error" else "P1",
                    "scope": "shot",
                    "shot_id": s_res["shot_id"],
                    "code": finding["code"],
                    "issue": finding["message"],
                    "suggested_fix": finding["fix_hint"],
                    "recommended_tool": tool,
                }
            )
    for gap in genre_gaps:
        anchor_shot = manifest.shots[0].id if manifest.shots else "shot-01"
        if gap["gap_id"] in {"aftermath_and_breath", "reaction_and_aftermath"} and manifest.shots:
            anchor_shot = manifest.shots[-1].id
        repair_suggestions.append(
            {
                "priority": "P1",
                "scope": "episode_genre",
                "shot_id": anchor_shot,
                "code": f"GENRE_GAP_{gap['gap_id'].upper()}",
                "issue": f"题材({genre_key})缺少「{gap['label']}」",
                "suggested_fix": gap["advice"],
                "recommended_tool": gap["recommended_tool"],
            }
        )

    p0_count = sum(1 for r in repair_suggestions if r["priority"] == "P0")
    genre_coverage = round(len(genre_hits) / len(checklist), 2) if checklist else 1.0
    readiness_score = max(0, min(100, int(round(shot_audit["average_score"] * 0.7 + genre_coverage * 100 * 0.3))))

    rehearsal_brief_prompt = (
        f"请对工程 {manifest.project_id}（题材：{genre_key}，共 {len(manifest.shots)} 镜）执行「成片质量剧本预演」。\n"
        f"当前预演就绪分：{readiness_score}/100（P0 阻断项 {p0_count} 个，题材要素覆盖率 {int(genre_coverage * 100)}%）。\n"
        f"历史成片 Demo 教训（来自质量飞轮）：{demo_lessons[:5]}\n"
        f"题材缺失项：{genre_gaps}\n"
        f"单镜硬伤与预警：{repair_suggestions[:8]}\n"
        "请按 P0 -> P1 顺序给出具体剧本修改方案，并直接调用 story_patch_script_beat 或 story_insert_bridge_shot 补齐漏洞后再进入生图与渲染。"
    )

    return {
        "schema": SCRIPT_REHEARSAL_SCHEMA,
        "project_dir": str(root),
        "project_id": manifest.project_id,
        "genre": genre_key,
        "ready_for_production": p0_count == 0 and len(genre_gaps) == 0,
        "readiness_score": readiness_score,
        "genre_coverage": genre_coverage,
        "genre_hits": genre_hits,
        "genre_gaps": genre_gaps,
        "demo_lessons_count": len(demo_lessons),
        "demo_lessons": demo_lessons,
        "p0_blocking_count": p0_count,
        "repair_suggestions": repair_suggestions,
        "shot_audit": shot_audit,
        "rehearsal_brief_prompt": rehearsal_brief_prompt,
    }
