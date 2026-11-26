"""Persistent Quality Flywheel: records supervisor interventions, verifies outcomes,
distills reusable rules/exemplars into a playbook, and auto-applies learned priors."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from .manifest import ProjectManifest, Shot

FAILURE_CODES = (
    "NONE",
    "PLOT_DRIFT",
    "IMAGE_PROMPT_DRIFT",
    "CHARACTER_INCONSISTENT",
    "HAND_FACE_DEFECT",
    "LOCAL_DEFECT",
    "SPLIT_PANEL_TEXT",
    "VIDEO_STATIC_OR_WARP",
    "VIDEO_MOTION_FAIL",
    "AUDIO_PACING_MISMATCH",
    "COMPOSITION_LIGHTING",
    "OTHER",
)

_ACTION_BOILERPLATE = "。仅绘制当前动作的一张完整画面、一个时间点。不要分屏、拼贴、上下多格、漫画分镜、字幕或文字。"
_DEFAULT_NEGATIVE = "无文字水印、无畸形手指、无多余人物、无闪烁"
_TRANSFERABLE_GUARDRAIL_RE = re.compile(
    r"(?:构图|特写|中景|近景|全景|远景|景别|单帧|机位|光影|光线|逆光|侧光|柔光|轮廓|骨相|手部|手指|五官|面部|比例|严禁|禁止|避免|无分屏|无文字|稳定|清晰|口型|双唇|景深|视角|镜头)"
)

_KNOWN_TRIGGER_TOKENS = (
    "推门", "推开", "开门", "关门", "转身", "回头", "奔跑", "狂奔", "跑", "追逐", "跳跃",
    "拔剑", "挥剑", "打斗", "摔倒", "跌倒", "站起", "俯身", "伸手", "递出", "接过",
    "拿起", "放下", "走出", "走进", "跑向", "对视", "争执", "拥抱", "特写", "近景",
    "全景", "远景", "雨夜", "暴雨", "黑夜", "霓虹", "钟楼", "咖啡馆", "车站", "天台",
    "走廊", "窗边", "街道", "爆炸", "飙车", "微笑", "叹气", "低声", "大喊", "手部",
)

_CJK_STOPWORDS = {
    "于是", "然后", "接着", "随后", "只见", "忽然", "突然", "一个", "一位", "一名",
    "这个", "那个", "正在", "自己", "他们", "她们", "我们", "没有", "已经", "可以",
    "画面", "镜头", "动作", "场景",
}

_ASCII_STOPWORDS = {
    "the", "and", "with", "for", "from", "into", "onto", "over", "under", "shot",
    "scene", "frame", "character", "close", "medium", "wide",
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def resolve_flywheel_dir(
    manifest: ProjectManifest | None = None,
    project_root: str | Path | None = None,
    explicit_dir: str | Path | None = None,
    *,
    server_default_dir: str | Path | None = None,
) -> Path | None:
    """Resolve the active flywheel directory with explicit > project > env > server > local fallback."""
    if explicit_dir is not None and str(explicit_dir).strip():
        return Path(str(explicit_dir).strip())
    if manifest is not None and getattr(manifest, "flywheel_path", None):
        raw = str(manifest.flywheel_path).strip()
        if raw:
            return Path(raw)
    env_dir = os.environ.get("STORY_MEDIA_FLYWHEEL_DIR", "").strip()
    if env_dir:
        return Path(env_dir)
    if server_default_dir is not None and str(server_default_dir).strip():
        return Path(str(server_default_dir).strip())
    if project_root is not None:
        return Path(project_root) / "flywheel"
    return None


def extract_trigger_keywords(text: str, *, limit: int = 8) -> list[str]:
    """Extract normalized action/scene trigger keywords for rule matching."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    found: list[str] = []
    for token in _KNOWN_TRIGGER_TOKENS:
        if token in cleaned and token not in found:
            found.append(token)

    for match in re.finditer(r"[a-zA-Z]{3,18}", cleaned.lower()):
        word = match.group(0)
        if word not in _ASCII_STOPWORDS and word not in found:
            found.append(word)

    if not found:
        for seq in re.findall(r"[\u4e00-\u9fff]{2,4}", cleaned):
            if seq not in _CJK_STOPWORDS and seq not in found:
                found.append(seq)
            if len(found) >= limit:
                break

    return found[:limit]


def _read_cases(flywheel_dir: Path) -> list[dict[str, Any]]:
    cases_file = flywheel_dir / "cases.jsonl"
    if not cases_file.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in cases_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("case_id"):
            records.append(item)
    return records


def _write_cases(flywheel_dir: Path, cases: list[dict[str, Any]]) -> None:
    lines = [json.dumps(item, ensure_ascii=False) for item in cases]
    payload = ("\n".join(lines) + "\n") if lines else ""
    _atomic_write_text(flywheel_dir / "cases.jsonl", payload)


def _read_manual_config(flywheel_dir: Path) -> dict[str, Any]:
    path = flywheel_dir / "manual_rules.json"
    if not path.is_file():
        return {"rules": [], "disabled_rule_ids": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"rules": [], "disabled_rule_ids": []}
    if not isinstance(data, dict):
        return {"rules": [], "disabled_rule_ids": []}
    data.setdefault("rules", [])
    data.setdefault("disabled_rule_ids", [])
    return data


def _write_manual_config(flywheel_dir: Path, config: dict[str, Any]) -> None:
    _atomic_write_text(
        flywheel_dir / "manual_rules.json",
        json.dumps(config, ensure_ascii=False, indent=2),
    )


def _extract_reusable_guidance(
    before_prompt: str,
    prompt_override: str,
    manifest: ProjectManifest | None,
) -> str:
    """Extract reusable visual/camera/framing modifiers introduced in prompt_override."""
    char_names: set[str] = set()
    if manifest is not None and isinstance(manifest.character_bible, dict):
        bible = manifest.character_bible
        raw_profiles = bible.get("characters") if isinstance(bible.get("characters"), dict) else bible
        for prof in (raw_profiles or {}).values():
            if isinstance(prof, dict) and prof.get("name"):
                char_names.add(str(prof["name"]).strip())

    clean_before = (before_prompt or "").replace(_ACTION_BOILERPLATE, "")
    clean_after = (prompt_override or "").replace(_ACTION_BOILERPLATE, "")
    raw_clauses = [c.strip() for c in re.split(r"[，,；;。]+", clean_after) if c.strip()]

    reusable: list[str] = []
    for clause in raw_clauses:
        if clause in clean_before:
            continue
        if any(name and name in clause for name in char_names):
            continue
        if not _TRANSFERABLE_GUARDRAIL_RE.search(clause):
            continue
        if clause not in reusable:
            reusable.append(clause)
    return "；".join(reusable)[:120]


def record_intervention_case(
    flywheel_dir: str | Path,
    *,
    manifest: ProjectManifest,
    shot: Shot,
    failure_code: str,
    failure_reason: str,
    strategy: str,
    intervention: dict[str, Any],
    supervisor_id: str = "gemini-4-carbon",
    before_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record a supervisor intervention as pending_verification in cases.jsonl."""
    root = Path(flywheel_dir)
    report = shot.quality_report if isinstance(shot.quality_report, dict) else {}
    snap = before_snapshot if isinstance(before_snapshot, dict) else {}
    raw_score = snap.get("before_score", report.get("overall_score"))
    if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool):
        before_score = float(raw_score)
    else:
        before_score = 0.30 if report.get("decision") == "failed" else 0.60

    ctx = shot.scene_context if isinstance(shot.scene_context, dict) else {}
    before_prompt = str(
        snap.get("before_prompt")
        or shot.prompt_override
        or ctx.get("visual_action")
        or shot.text
        or ""
    ).strip()
    before_negative = str(
        snap.get("before_negative")
        if snap.get("before_negative") is not None
        else (shot.negative_override or ctx.get("negative") or "")
    ).strip()
    before_route = str(
        snap.get("before_route")
        or ctx.get("_video_route")
        or shot.video_route
        or manifest.video_routing_policy
        or "auto"
    )

    clean_code = (failure_code or report.get("failure_code") or "OTHER").strip().upper()
    if clean_code not in FAILURE_CODES:
        clean_code = "OTHER"

    explicit_triggers = intervention.get("trigger_keywords")
    if isinstance(explicit_triggers, list) and explicit_triggers:
        triggers = [str(k).strip() for k in explicit_triggers if str(k).strip()]
    else:
        triggers = extract_trigger_keywords(f"{shot.text} {before_prompt}")

    enriched_intervention = dict(intervention)
    if (
        enriched_intervention.get("prompt_override")
        and not enriched_intervention.get("action_guidance")
    ):
        derived = _extract_reusable_guidance(
            before_prompt,
            str(enriched_intervention["prompt_override"]),
            manifest,
        )
        if derived:
            enriched_intervention["action_guidance"] = derived

    seed_override = enriched_intervention.get("seed_override")
    before_seed = snap.get("before_seed")
    seed_changed = bool(seed_override is not None and seed_override != before_seed)

    causal_axes: list[str] = []
    new_prompt = str(enriched_intervention.get("prompt_override") or "").strip()
    if (new_prompt and new_prompt != before_prompt) or str(enriched_intervention.get("action_guidance") or "").strip():
        causal_axes.append("prompt")
    new_neg = str(enriched_intervention.get("negative_override") or "").strip()
    if new_neg and new_neg != before_negative:
        causal_axes.append("negative")
    if isinstance(enriched_intervention.get("character_dna_patch"), dict) and str(enriched_intervention.get("character_name") or "").strip():
        causal_axes.append("character_dna")
    new_route = str(enriched_intervention.get("video_route") or "").strip()
    new_motion = str(enriched_intervention.get("motion") or "").strip()
    if (new_route and new_route != before_route) or new_motion:
        causal_axes.append("route")
    if seed_changed:
        causal_axes.append("seed")
    is_seed_luck = bool(causal_axes == ["seed"])

    case = {
        "schema": "flywheel-case/v1",
        "case_id": f"fw_{uuid4().hex[:12]}",
        "project_id": manifest.project_id,
        "shot_id": shot.id,
        "shot_text": shot.text,
        "style_preset": manifest.style_preset,
        "stage": str(enriched_intervention.get("stage") or report.get("stage") or "image"),
        "failure_code": clean_code,
        "failure_reason": str(failure_reason or report.get("reason") or "").strip(),
        "strategy": str(strategy),
        "before_score": round(before_score, 4),
        "before_prompt": before_prompt,
        "before_negative": before_negative,
        "before_route": before_route,
        "trigger_keywords": triggers,
        "seed_changed": seed_changed,
        "causal_axes": causal_axes,
        "is_seed_luck": is_seed_luck,
        "intervention": enriched_intervention,
        "supervisor_id": str(supervisor_id or "gemini-4-carbon"),
        "status": "pending_verification",
        "after_decision": None,
        "after_score": None,
        "delta_score": None,
        "created_at": _utcnow_iso(),
        "resolved_at": None,
    }

    cases = _read_cases(root)
    cases.append(case)
    _write_cases(root, cases)
    if isinstance(shot.scene_context, dict):
        shot.scene_context["_pending_flywheel_case_id"] = case["case_id"]
    return case


def resolve_pending_case_on_verdict(
    flywheel_dir: str | Path,
    shot: Shot,
    verdict_report: dict[str, Any],
    *,
    project_id: str | None = None,
) -> dict[str, Any] | None:
    """Verify or reject a pending flywheel intervention when a new verdict arrives."""
    root = Path(flywheel_dir)
    ctx = shot.scene_context if isinstance(shot.scene_context, dict) else {}
    pending_id = ctx.get("_pending_flywheel_case_id")
    cases = _read_cases(root)
    if not cases:
        ctx.pop("_pending_flywheel_case_id", None)
        return None

    target_idx: int | None = None
    if pending_id:
        target_idx = next((idx for idx, item in enumerate(cases) if item.get("case_id") == pending_id), None)
    if target_idx is None and project_id:
        verdict_stage = str(verdict_report.get("stage") or "").strip().lower()
        for idx in range(len(cases) - 1, -1, -1):
            item = cases[idx]
            if (
                item.get("status") == "pending_verification"
                and item.get("project_id") == project_id
                and item.get("shot_id") == shot.id
                and (not verdict_stage or str(item.get("stage") or "").lower() == verdict_stage)
            ):
                target_idx = idx
                break
    if target_idx is None:
        ctx.pop("_pending_flywheel_case_id", None)
        return None

    case = dict(cases[target_idx])
    if case.get("status") != "pending_verification":
        ctx.pop("_pending_flywheel_case_id", None)
        return None
    after_decision = str(verdict_report.get("decision") or "").lower()
    after_score = float(verdict_report.get("overall_score") or 0.0)
    before_score = float(case.get("before_score") or 0.0)
    delta_score = round(after_score - before_score, 4)

    if after_decision == "passed" or (after_decision == "warning" and delta_score >= 0.15):
        next_status = "verified_effective"
    elif after_decision == "failed":
        next_status = "rejected_ineffective"
    else:
        return None

    case["status"] = next_status
    case["after_decision"] = after_decision
    case["after_score"] = round(after_score, 4)
    case["delta_score"] = delta_score
    case["resolved_at"] = _utcnow_iso()
    cases[target_idx] = case
    _write_cases(root, cases)
    ctx.pop("_pending_flywheel_case_id", None)

    if next_status == "verified_effective":
        distill_playbook(root)
    return case


def _short_rule_id(prefix: str, seed_text: str) -> str:
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def distill_playbook(flywheel_dir: str | Path) -> dict[str, Any]:
    """Aggregate verified trajectories + manual overrides into playbook.json."""
    root = Path(flywheel_dir)
    cases = _read_cases(root)
    manual_cfg = _read_manual_config(root)
    disabled_ids = set(str(x) for x in (manual_cfg.get("disabled_rule_ids") or []))

    verified = [c for c in cases if c.get("status") == "verified_effective"]
    rejected = [c for c in cases if c.get("status") == "rejected_ineffective"]
    pending = [c for c in cases if c.get("status") == "pending_verification"]
    seed_luck_count = 0

    neg_agg: dict[tuple[str, str], dict[str, Any]] = {}
    guard_agg: dict[str, dict[str, Any]] = {}
    dna_agg: dict[str, dict[str, Any]] = {}
    route_agg: dict[str, dict[str, Any]] = {}
    failure_dist: dict[str, int] = {}

    for case in cases:
        code = str(case.get("failure_code") or "OTHER")
        if code != "NONE":
            failure_dist[code] = failure_dist.get(code, 0) + 1

    for case in verified:
        if case.get("is_seed_luck") or case.get("causal_axes") == ["seed"]:
            seed_luck_count += 1
            continue
        intervention = case.get("intervention") if isinstance(case.get("intervention"), dict) else {}
        style_key = str(case.get("style_preset") or "")
        code = str(case.get("failure_code") or "OTHER")
        delta = float(case.get("delta_score") or 0.0)
        triggers = [str(t).strip() for t in (case.get("trigger_keywords") or []) if str(t).strip()]
        clean_hit_inc = 0 if case.get("seed_changed") else 1

        neg_raw = str(intervention.get("negative_override") or "").strip()
        universal_defect = code in {
            "SPLIT_PANEL_TEXT",
            "HAND_FACE_DEFECT",
            "LOCAL_DEFECT",
            "CHARACTER_INCONSISTENT",
            "PLOT_DRIFT",
            "IMAGE_PROMPT_DRIFT",
            "VIDEO_STATIC_OR_WARP",
            "VIDEO_MOTION_FAIL",
        }
        rule_style = "" if universal_defect else style_key
        if neg_raw:
            for term in re.split(r"[，,、;；]+", neg_raw):
                clean_term = term.strip()
                if not clean_term:
                    continue
                bucket_key = (clean_term, rule_style)
                rule_id = _short_rule_id("neg", f"{clean_term}:{rule_style}")
                if rule_id in disabled_ids:
                    continue
                entry = neg_agg.setdefault(
                    bucket_key,
                    {
                        "rule_id": rule_id,
                        "term": clean_term,
                        "style_preset": rule_style or None,
                        "failure_code": code,
                        "hits": 0,
                        "clean_hits": 0,
                        "delta_sum": 0.0,
                        "source": "distilled",
                    },
                )
                entry["hits"] += 1
                entry["clean_hits"] += clean_hit_inc
                entry["delta_sum"] += delta

        guidance = str(intervention.get("action_guidance") or "").strip()
        if guidance:
            rule_id = _short_rule_id("act", f"{guidance}:{rule_style}")
            if rule_id not in disabled_ids:
                entry = guard_agg.setdefault(
                    rule_id,
                    {
                        "rule_id": rule_id,
                        "trigger_keywords": list(triggers),
                        "append_guidance": guidance,
                        "style_preset": rule_style or None,
                        "failure_code": code,
                        "hits": 0,
                        "clean_hits": 0,
                        "delta_sum": 0.0,
                        "source": "distilled",
                    },
                )
                entry["hits"] += 1
                entry["clean_hits"] += clean_hit_inc
                entry["delta_sum"] += delta
                for kw in triggers:
                    if kw not in entry["trigger_keywords"]:
                        entry["trigger_keywords"].append(kw)

        dna_patch = intervention.get("character_dna_patch")
        char_name = str(intervention.get("character_name") or "").strip()
        if isinstance(dna_patch, dict) and char_name:
            rule_id = _short_rule_id("dna", char_name)
            if rule_id not in disabled_ids:
                entry = dna_agg.setdefault(
                    rule_id,
                    {
                        "rule_id": rule_id,
                        "character_name": char_name,
                        "appearance": str(dna_patch.get("appearance") or "").strip(),
                        "clothing": str(dna_patch.get("clothing") or "").strip(),
                        "visual_dna": str(dna_patch.get("visual_dna") or "").strip(),
                        "failure_code": code,
                        "hits": 0,
                        "clean_hits": 0,
                        "delta_sum": 0.0,
                        "source": "distilled",
                    },
                )
                entry["hits"] += 1
                entry["clean_hits"] += clean_hit_inc
                entry["delta_sum"] += delta
                for field in ("appearance", "clothing", "visual_dna"):
                    if dna_patch.get(field):
                        entry[field] = str(dna_patch[field]).strip()

        pref_route = str(intervention.get("video_route") or "").strip()
        pref_motion = str(intervention.get("motion") or "").strip() or None
        if pref_route or pref_motion:
            avoid_route = str(case.get("before_route") or "").strip() or None
            trig_key = ",".join(sorted(triggers))
            rule_id = _short_rule_id("rte", f"{pref_route}:{pref_motion}:{avoid_route}:{trig_key}")
            if rule_id not in disabled_ids:
                entry = route_agg.setdefault(
                    rule_id,
                    {
                        "rule_id": rule_id,
                        "trigger_keywords": list(triggers),
                        "avoid_route": avoid_route,
                        "preferred_route": pref_route or None,
                        "preferred_motion": pref_motion,
                        "failure_code": code,
                        "hits": 0,
                        "clean_hits": 0,
                        "delta_sum": 0.0,
                        "source": "distilled",
                    },
                )
                entry["hits"] += 1
                entry["clean_hits"] += clean_hit_inc
                entry["delta_sum"] += delta

    def finalize_confidence(item: dict[str, Any]) -> dict[str, Any]:
        hits = max(1, int(item.pop("hits", 1)))
        clean_hits = int(item.pop("clean_hits", hits))
        avg_delta = float(item.pop("delta_sum", 0.0)) / hits
        item["hits"] = hits
        item["clean_hits"] = clean_hits
        if clean_hits == 0 and hits == 1 and avg_delta < 0.25:
            item["confidence"] = 0.55
        else:
            item["confidence"] = round(min(0.99, 0.65 + 0.12 * hits + max(0.0, avg_delta) * 0.25), 3)
        return item

    negative_boosts = [finalize_confidence(v) for v in neg_agg.values()]
    action_guardrails = [finalize_confidence(v) for v in guard_agg.values()]
    character_dna_patches = [finalize_confidence(v) for v in dna_agg.values()]
    routing_priors = [finalize_confidence(v) for v in route_agg.values()]

    # Merge explicit manual rules
    for m_rule in manual_cfg.get("rules") or []:
        if not isinstance(m_rule, dict):
            continue
        rid = str(m_rule.get("rule_id") or "")
        if not rid or rid in disabled_ids:
            continue
        kind = str(m_rule.get("kind") or "")
        payload = {**m_rule, "source": "manual"}
        payload.setdefault("hits", 1)
        payload.setdefault("confidence", 0.95)
        if kind == "negative_boost":
            negative_boosts = [r for r in negative_boosts if r["rule_id"] != rid] + [payload]
        elif kind == "action_guardrail":
            action_guardrails = [r for r in action_guardrails if r["rule_id"] != rid] + [payload]
        elif kind == "character_dna_patch":
            character_dna_patches = [r for r in character_dna_patches if r["rule_id"] != rid] + [payload]
        elif kind == "routing_prior":
            routing_priors = [r for r in routing_priors if r["rule_id"] != rid] + [payload]

    negative_boosts.sort(key=lambda r: (r.get("confidence", 0.0), r.get("hits", 0)), reverse=True)
    action_guardrails.sort(key=lambda r: (r.get("confidence", 0.0), r.get("hits", 0)), reverse=True)
    routing_priors.sort(key=lambda r: (r.get("confidence", 0.0), r.get("hits", 0)), reverse=True)

    sorted_verified = sorted(
        verified,
        key=lambda c: (float(c.get("delta_score") or 0.0), str(c.get("resolved_at") or "")),
        reverse=True,
    )
    exemplars = [
        {
            "case_id": c["case_id"],
            "failure_code": c.get("failure_code"),
            "strategy": c.get("strategy"),
            "shot_text": c.get("shot_text"),
            "before_prompt": c.get("before_prompt"),
            "prompt_override": (c.get("intervention") or {}).get("prompt_override"),
            "negative_override": (c.get("intervention") or {}).get("negative_override"),
            "video_route": (c.get("intervention") or {}).get("video_route"),
            "delta_score": c.get("delta_score"),
        }
        for c in sorted_verified[:12]
    ]

    terminal_count = len(verified) + len(rejected)
    verification_rate = round(len(verified) / terminal_count, 4) if terminal_count > 0 else 0.0

    playbook = {
        "schema": "flywheel-playbook/v1",
        "updated_at": _utcnow_iso(),
        "stats": {
            "total_cases": len(cases),
            "verified_cases": len(verified),
            "rejected_cases": len(rejected),
            "pending_cases": len(pending),
            "seed_luck_cases": seed_luck_count,
            "verification_rate": verification_rate,
            "failure_distribution": failure_dist,
        },
        "negative_boosts": negative_boosts,
        "action_guardrails": action_guardrails,
        "character_dna_patches": character_dna_patches,
        "routing_priors": routing_priors,
        "exemplars": exemplars,
    }
    _atomic_write_text(root / "playbook.json", json.dumps(playbook, ensure_ascii=False, indent=2))
    return playbook


def load_active_playbook(
    manifest: ProjectManifest | None = None,
    project_root: str | Path | None = None,
    *,
    explicit_dir: str | Path | None = None,
    server_default_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    """Return active playbook.json unless flywheel auto-application is disabled or absent."""
    if manifest is not None and isinstance(manifest.supervisor_policy, dict):
        if manifest.supervisor_policy.get("auto_apply_flywheel") is False:
            return None
    fly_dir = resolve_flywheel_dir(
        manifest,
        project_root,
        explicit_dir,
        server_default_dir=server_default_dir,
    )
    if fly_dir is None:
        return None
    playbook_file = fly_dir / "playbook.json"
    if not playbook_file.is_file():
        return None
    try:
        data = json.loads(playbook_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("schema") != "flywheel-playbook/v1":
        return None
    return data


def _triggers_match(triggers: list[str] | None, text_blob: str) -> bool:
    if not triggers:
        return True
    lowered = text_blob.lower()
    return any(str(tok).strip().lower() in lowered for tok in triggers if str(tok).strip())


def _matching_negative_boosts(
    playbook: dict[str, Any],
    manifest: ProjectManifest,
) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    style_preset = getattr(manifest, "style_preset", None)
    for rule in playbook.get("negative_boosts") or []:
        if not isinstance(rule, dict):
            continue
        if float(rule.get("confidence", 1.0)) < 0.60:
            continue
        r_style = rule.get("style_preset")
        if r_style and style_preset and r_style != style_preset:
            continue
        if r_style and not style_preset:
            continue
        if rule.get("term"):
            matched.append(rule)
    return matched


def _matching_action_guardrails(
    playbook: dict[str, Any],
    manifest: ProjectManifest,
    text_blob: str,
) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    style_preset = getattr(manifest, "style_preset", None)
    for rule in playbook.get("action_guardrails") or []:
        if not isinstance(rule, dict):
            continue
        if float(rule.get("confidence", 1.0)) < 0.60:
            continue
        r_style = rule.get("style_preset")
        if r_style and style_preset != r_style:
            continue
        if rule.get("append_guidance") and _triggers_match(rule.get("trigger_keywords"), text_blob):
            matched.append(rule)
    return matched


def flywheel_shot_signature(
    manifest: ProjectManifest,
    shot: Shot,
    *,
    project_root: str | Path | None = None,
) -> str:
    """Return a compact cache-busting signature of flywheel rules active for this shot."""
    playbook = load_active_playbook(manifest, project_root)
    if not playbook:
        return ""
    ctx = shot.scene_context if isinstance(shot.scene_context, dict) else {}
    text_blob = f"{shot.text} {ctx.get('visual_action') or ''}"
    tokens: list[str] = []
    for item in _matching_negative_boosts(playbook, manifest)[:6]:
        tokens.append(f"neg:{item.get('term')}:{item.get('rule_id')}")
    if not getattr(shot, "prompt_override", None):
        for item in _matching_action_guardrails(playbook, manifest, text_blob)[:2]:
            tokens.append(f"act:{item.get('rule_id')}:{item.get('append_guidance')}")
    if not tokens:
        return ""
    return hashlib.sha256("|".join(tokens).encode("utf-8")).hexdigest()[:12]


def apply_flywheel_to_image_scene(
    manifest: ProjectManifest,
    shot: Shot,
    scene: dict[str, Any],
    *,
    project_root: str | Path | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Apply learned negative boosts & action guardrails to an image scene dict."""
    playbook = load_active_playbook(manifest, project_root)
    if not playbook:
        return scene, []

    updated = dict(scene)
    applied_ids: list[str] = []

    neg_rules = _matching_negative_boosts(playbook, manifest)[:6]
    if neg_rules:
        base_neg = str(updated.get("negative") or _DEFAULT_NEGATIVE).strip()
        existing_terms = {t.strip() for t in re.split(r"[，,、;；]+", base_neg) if t.strip()}
        added_terms: list[str] = []
        for rule in neg_rules:
            term = str(rule["term"]).strip()
            if term and term not in existing_terms:
                existing_terms.add(term)
                added_terms.append(term)
                applied_ids.append(str(rule.get("rule_id") or term))
        if added_terms:
            updated["negative"] = "，".join([t for t in [base_neg, *added_terms] if t])

    if not getattr(shot, "prompt_override", None):
        text_blob = f"{shot.text} {updated.get('action') or ''}"
        guard_rules = _matching_action_guardrails(playbook, manifest, text_blob)
        extra_guidance: list[str] = []
        for rule in guard_rules[:2]:
            guidance = str(rule["append_guidance"]).strip()
            if guidance and guidance not in str(updated.get("action") or ""):
                extra_guidance.append(guidance)
                applied_ids.append(str(rule.get("rule_id") or guidance))
        if extra_guidance:
            addition = "；".join(extra_guidance)
            current_action = str(updated.get("action") or "")
            if _ACTION_BOILERPLATE in current_action:
                head, tail = current_action.split(_ACTION_BOILERPLATE, 1)
                updated["action"] = f"{head}；{addition}{_ACTION_BOILERPLATE}{tail}"
            elif current_action:
                updated["action"] = f"{current_action}；{addition}"
            else:
                updated["action"] = addition

    return updated, applied_ids


def apply_flywheel_to_video_route(
    manifest: ProjectManifest,
    shot: Shot,
    text_blob: str,
    decision: dict[str, Any],
    *,
    has_first_frame: bool = False,
    has_last_frame: bool = False,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Apply matching video routing priors when shot.video_route is not manually pinned."""
    if getattr(shot, "video_route", None):
        return decision
    playbook = load_active_playbook(manifest, project_root)
    if not playbook:
        return decision

    priors = playbook.get("routing_priors") or []
    if not priors:
        return decision

    current_route = str(decision.get("route") or "")
    current_provider = str(decision.get("provider") or "")
    route_model_map = {
        "i2v_fast": ("dashscope", "wan2.2-i2v-flash", "i2v"),
        "i2v_quality": ("dashscope", "wan2.2-i2v-plus", "i2v"),
        "kf2v": ("dashscope", "wan2.2-kf2v-flash", "kf2v"),
        "r2v": ("dashscope", "wan2.1-vace-plus", "r2v"),
        "t2v": ("dashscope", "wanx2.1-t2v-turbo", "t2v"),
        "comfy_i2v": ("comfyui", None, "i2v"),
        "comfy_fl2v": ("comfyui", None, "i2v"),
        "comfy_t2v": ("comfyui", None, "t2v"),
        "preview": ("preview", None, "preview"),
    }

    _I2V_FAMILY = {"i2v_fast", "i2v_quality", "t2v", "comfy_i2v", "comfy_t2v"}
    for prior in priors:
        if not isinstance(prior, dict):
            continue
        if not _triggers_match(prior.get("trigger_keywords"), text_blob):
            continue
        avoid = prior.get("avoid_route")
        pref_route = prior.get("preferred_route")
        if current_provider == "comfyui" and pref_route in {"kf2v", "i2v_quality"}:
            pref_route = "comfy_fl2v"
        elif current_provider == "comfyui" and pref_route in {"i2v_fast", "t2v", "r2v"}:
            pref_route = "comfy_i2v" if has_first_frame else "comfy_t2v"
        if avoid and current_route and avoid != current_route:
            # Allow kf2v / comfy_fl2v priors learned on any single-frame route to upgrade other single-frame routes.
            if not (pref_route in {"kf2v", "comfy_fl2v"} and avoid in _I2V_FAMILY and current_route in _I2V_FAMILY):
                continue
        pref_motion = prior.get("preferred_motion")
        updated = dict(decision)
        changed = False

        if pref_route and pref_route in route_model_map and pref_route != current_route:
            prov, model_name, wf_mode = route_model_map[pref_route]
            feasible = True
            if pref_route == "kf2v" and not (has_first_frame and has_last_frame):
                feasible = False
            elif pref_route in {"i2v_fast", "i2v_quality", "comfy_i2v", "comfy_fl2v"} and not has_first_frame:
                feasible = False
            elif prov != current_provider and current_provider == "preview" and pref_route != "preview":
                # Keep preview when no video backend is configured at all.
                feasible = False
            elif prov != current_provider and manifest.video_provider == "comfyui" and pref_route != "preview":
                feasible = False
            if feasible:
                # Respect remaining project budget before upgrading to a more expensive route.
                try:
                    from .cost_ledger import check_rework_budget, estimate_operation_cost

                    est = estimate_operation_cost("video", pref_route, manifest=manifest)
                    check_rework_budget(manifest, est)
                except Exception:
                    feasible = False
            if feasible:
                updated["route"] = pref_route
                updated["provider"] = prov
                updated["model"] = model_name
                updated["workflow_mode"] = wf_mode
                if pref_route == "comfy_fl2v":
                    updated["needs_end_frame"] = True
                updated["reason"] = f"flywheel_prior:{prior.get('rule_id')}:{pref_route}"
                changed = True

        if pref_motion and (shot.motion == "none" or not shot.motion):
            updated["suggested_motion"] = pref_motion
            if not changed:
                updated["reason"] = f"flywheel_prior:{prior.get('rule_id')}:motion:{pref_motion}"
            changed = True

        if changed:
            return updated

    return decision


def upsert_manual_rule(flywheel_dir: str | Path, rule_payload: dict[str, Any]) -> dict[str, Any]:
    """Insert or update an explicit rule in the flywheel and rebuild playbook.json."""
    if not isinstance(rule_payload, dict):
        raise ValueError("rule_payload must be a dict")
    kind = str(rule_payload.get("kind") or "").strip()
    if kind not in {"negative_boost", "action_guardrail", "character_dna_patch", "routing_prior"}:
        raise ValueError(f"unsupported flywheel rule kind: {kind!r}")

    root = Path(flywheel_dir)
    cfg = _read_manual_config(root)
    rule_id = str(rule_payload.get("rule_id") or "").strip() or _short_rule_id("man", json.dumps(rule_payload, sort_keys=True, ensure_ascii=False))
    entry = {**rule_payload, "kind": kind, "rule_id": rule_id}
    rules = [r for r in cfg.get("rules", []) if isinstance(r, dict) and r.get("rule_id") != rule_id]
    rules.append(entry)
    cfg["rules"] = rules
    cfg["disabled_rule_ids"] = [x for x in cfg.get("disabled_rule_ids", []) if x != rule_id]
    _write_manual_config(root, cfg)
    playbook = distill_playbook(root)
    return {"rule": entry, "playbook": playbook}


def delete_rule(flywheel_dir: str | Path, rule_id: str) -> dict[str, Any]:
    """Disable/delete a rule by rule_id and rebuild playbook.json."""
    clean_id = str(rule_id or "").strip()
    if not clean_id:
        raise ValueError("rule_id must not be empty")
    root = Path(flywheel_dir)
    cfg = _read_manual_config(root)
    cfg["rules"] = [r for r in cfg.get("rules", []) if isinstance(r, dict) and r.get("rule_id") != clean_id]
    disabled = set(str(x) for x in cfg.get("disabled_rule_ids", []))
    disabled.add(clean_id)
    cfg["disabled_rule_ids"] = sorted(disabled)
    _write_manual_config(root, cfg)
    playbook = distill_playbook(root)
    return {"deleted_rule_id": clean_id, "playbook": playbook}


def get_flywheel_stats(flywheel_dir: str | Path) -> dict[str, Any]:
    """Return current flywheel playbook & case summary (distilling if needed)."""
    root = Path(flywheel_dir)
    playbook_file = root / "playbook.json"
    if playbook_file.is_file():
        playbook = json.loads(playbook_file.read_text(encoding="utf-8"))
    else:
        playbook = distill_playbook(root)
    return {
        "flywheel_dir": str(root),
        "playbook": playbook,
        "cases_count": len(_read_cases(root)),
    }


def export_flywheel_dataset(
    flywheel_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Export verified interventions as DPO/SFT JSONL preference pairs."""
    root = Path(flywheel_dir)
    cases = _read_cases(root)
    verified = [c for c in cases if c.get("status") == "verified_effective"]
    target = Path(output_path) if output_path else (root / "dpo_sft_dataset.jsonl")
    records: list[dict[str, Any]] = []
    for case in verified:
        intervention = case.get("intervention") if isinstance(case.get("intervention"), dict) else {}
        records.append(
            {
                "case_id": case.get("case_id"),
                "stage": case.get("stage"),
                "failure_code": case.get("failure_code"),
                "failure_reason": case.get("failure_reason"),
                "input_story": case.get("shot_text"),
                "rejected_prompt": case.get("before_prompt"),
                "chosen_prompt": intervention.get("prompt_override") or case.get("before_prompt"),
                "rejected_negative": case.get("before_negative") or "",
                "chosen_negative": intervention.get("negative_override") or case.get("before_negative") or "",
                "rejected_route": case.get("before_route"),
                "chosen_route": intervention.get("video_route") or case.get("before_route"),
                "delta_score": case.get("delta_score"),
            }
        )
    lines = [json.dumps(item, ensure_ascii=False) for item in records]
    _atomic_write_text(target, ("\n".join(lines) + "\n") if lines else "")
    return {"path": str(target), "count": len(records)}
