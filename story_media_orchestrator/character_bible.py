"""Persistent character bible, visual DNA prompts, and image consistency conditioning."""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path
from typing import Any

from .manifest import ProjectManifest, Shot

CONSISTENCY_MODES = ("off", "auto", "prompt_lock", "anchor_ref")

STYLE_PRESETS: dict[str, dict[str, str]] = {
    "realistic_drama": {
        "label": "写实短剧",
        "prompt": "写实影视短剧质感，电影级自然肤质与真实光影",
        "negative": "卡通，二次元，畸变五官，多余手指",
    },
    "cinematic": {
        "label": "电影胶片",
        "prompt": "电影感浅景深光影，细腻胶片色调，戏剧明暗层次",
        "negative": "廉价塑料感，过曝，畸形肢体",
    },
    "anime_2d": {
        "label": "日漫赛璐璐",
        "prompt": "精致二次元赛璐璐动画风格，清晰线条，通透色彩",
        "negative": "真人照片，写实皮肤纹理，崩坏线稿",
    },
    "guofeng_3d": {
        "label": "国风CG",
        "prompt": "国风玄幻电影级3D渲染，精细东方服饰纹理与体积光",
        "negative": "低模，模糊贴图，现代西装混搭",
    },
    "ink_wash": {
        "label": "新中式水墨",
        "prompt": "新中式水墨工笔插画风，写意留白与雅致设色",
        "negative": "霓虹赛博，3D塑料感，杂乱高饱和",
    },
    "cyberpunk": {
        "label": "赛博霓虹",
        "prompt": "赛博朋克都市霓虹冷暖对比光影，湿润反光夜景质感",
        "negative": "古风田园，过曝纯白，扁平简笔画",
    },
    "stylized_3d_anime": {
        "label": "半3D立体骨相漫",
        "prompt": (
            "次世代半写实3D动漫渲染质感，立体3D面部骨相轮廓，高挺鼻梁高光与鼻翼转折，"
            "利落下颌线折角，深邃眼窝立体阴影，45度柔光主光配侧逆轮廓光，次表面散射细腻瓷感肤质"
        ),
        "negative": "扁平赛璐璐纸片脸，粗黑描边，正面死白平光，真人毛孔照片感，五官塌陷畸变",
    },
}

_SKIP_SPEAKERS = {"旁白", "narrator", "角色", "unknown", ""}
_FEMALE_PRONOUNS = ("她",)
_MALE_PRONOUNS = ("他",)
_FEMALE_HINTS = ("女", "female", "woman", "girl", "姐姐", "妹妹", "少女", "母亲", "阿姨", "小姐", "妇人", "女孩", "侍女", "巫女")
_MALE_HINTS = ("男", "male", "man", "boy", "哥哥", "弟弟", "少年", "父亲", "叔叔", "先生", "老人", "老者", "老伯", "男孩", "将军", "书生", "修钟匠")
_ACTION_BOILERPLATE = "。仅绘制当前动作的一张完整画面、一个时间点。不要分屏、拼贴、上下多格、漫画分镜、字幕或文字。"
_MAX_PROMPT_BUDGET = 480
_COMPOSE_OVERHEAD = 42
_PURE_T2I_NO_REF_RE = re.compile(r"^wanx?2\.[0-9]+-t2i\b", re.IGNORECASE)

_ROLE_NOUNS = (
    "修钟匠", "信使", "少女", "少年", "青年", "老人", "老者", "老伯", "妇人", "女子", "男子",
    "女孩", "男孩", "工匠", "剑客", "侠客", "侦探", "警察", "医生", "护士", "学生", "老师",
    "掌柜", "老板", "店员", "侍女", "将军", "书生", "画师", "乐师", "总裁", "秘书", "助理",
    "保镖", "司机", "船长", "旅人", "猎人", "巫女", "法师", "骑士",
)
_ROLE_NOUN_ALT = "|".join(_ROLE_NOUNS)
_NON_NAME_PREFIXES = (
    "来到", "穿过", "停下", "放下", "站在", "望着", "看着", "走进", "走出", "走过", "转身",
    "抬头", "低头", "冲上", "拿起", "取出", "发现", "看见", "听见", "握着", "攥着", "拉动",
    "正", "刚", "将", "在", "从", "向", "朝", "对", "和", "与", "把", "被", "的", "了", "着", "是", "有",
)
_ATTIRE_ROLE_RE = re.compile(
    rf"(?:身穿|穿着|穿|披着|一袭)(?P<clothing>[^，。；！？“”\n]{{2,22}}?)的"
    rf"(?P<desc_role>[\u4e00-\u9fff]{{0,6}}?(?P<role>{_ROLE_NOUN_ALT}))(?P<tail>[\u4e00-\u9fff]{{0,6}})"
)
_ARTICLE_ROLE_RE = re.compile(
    rf"(?:一位|一名|一个)(?P<appearance>[\u4e00-\u9fff、]{{1,10}}?)(?P<role>{_ROLE_NOUN_ALT})(?P<tail>[\u4e00-\u9fff]{{0,4}})"
)
_ROLE_ALIAS_POOL = ("老人", "老者", "老伯", "少女", "少年", "青年", "女子", "男子")


def model_supports_ref_img(image_model: str | None) -> bool:
    """Return False for pure Wan 2.x T2I endpoints that ignore or reject input.ref_img."""
    model = (image_model or "").strip()
    if not model:
        return True
    return not bool(_PURE_T2I_NO_REF_RE.match(model))


def _hint_matches(blob: str, hints: tuple[str, ...]) -> bool:
    for hint in hints:
        if hint.isascii():
            if re.search(rf"(?<![a-z]){re.escape(hint)}(?![a-z])", blob):
                return True
        elif hint in blob:
            return True
    return False


def _infer_text_gender(text_blob: str) -> str:
    lowered = text_blob.lower()
    is_female = _hint_matches(lowered, _FEMALE_HINTS)
    is_male = _hint_matches(lowered, _MALE_HINTS)
    if is_female and not is_male:
        return "female"
    if is_male and not is_female:
        return "male"
    return ""


def _pick_recurring_name(tail: str, story: str) -> str | None:
    best: str | None = None
    best_score = (0, 0)
    for length in (2, 3):
        if len(tail) < length:
            continue
        cand = tail[:length]
        if cand.startswith(_NON_NAME_PREFIXES) or any(tok in cand for tok in ("，", "。", "的", "了", "在", "被", "把")):
            continue
        if cand.endswith(("来", "去", "站", "走", "看", "说", "问", "喊", "笑", "拿", "取", "放", "被", "将")):
            continue
        count = story.count(cand)
        if count >= 2:
            score = (count, length)
            if score > best_score:
                best = cand
                best_score = score
    return best


def validate_consistency_settings(
    consistency_mode: str = "auto",
    style_preset: str | None = None,
) -> tuple[str, str | None]:
    mode = (consistency_mode or "auto").strip()
    if mode not in CONSISTENCY_MODES:
        raise ValueError(f"unsupported consistency mode: {consistency_mode}")
    preset = style_preset.strip() if isinstance(style_preset, str) and style_preset.strip() else None
    if preset is not None and preset not in STYLE_PRESETS:
        raise ValueError(f"unsupported style preset: {style_preset}")
    return mode, preset


def deterministic_character_seed(project_id: str, character_key: str) -> int:
    digest = hashlib.sha256(f"{project_id}:{character_key}".encode("utf-8")).hexdigest()[:8]
    return max(1, int(digest, 16) & 0x7FFFFFFF)


def _slug_character_key(raw: str) -> str:
    cleaned = re.sub(r"\s+", "_", raw.strip())
    return cleaned or "character"


def _join_unique(parts: list[str]) -> str:
    seen: list[str] = []
    for part in parts:
        text = part.strip(" ，,。;；")
        if text and text not in seen:
            seen.append(text)
    return "，".join(seen)


def _coerce_seed(value: Any, fallback: int) -> int:
    if value is None or value == "":
        return fallback
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(1, parsed & 0x7FFFFFFF)


def _match_bible_key(bible: dict[str, dict[str, Any]], token: str) -> str | None:
    target = token.strip()
    if not target:
        return None
    if target in bible:
        return target
    lowered = target.lower()
    for key, profile in bible.items():
        if key.lower() == lowered:
            return key
        node_id = str(profile.get("node_id") or "").strip()
        name = str(profile.get("name") or "").strip()
        if node_id and node_id.lower() == lowered:
            return key
        if name and name.lower() == lowered:
            return key
    return None


def _profile_from_dict(project_id: str, key: str, raw: dict[str, Any]) -> dict[str, Any]:
    name = str(raw.get("name") or raw.get("node_id") or key).strip()
    appearance = _join_unique([
        str(raw.get(field) or "")
        for field in ("appearance", "features", "face", "hair", "age", "gender")
        if isinstance(raw.get(field), str) and raw.get(field)
    ])
    description = str(raw.get("description") or raw.get("visual") or "").strip()
    if not appearance and description:
        appearance = description
    clothing = _join_unique([
        str(raw.get(field) or "")
        for field in ("clothing", "outfit", "costume", "attire")
        if isinstance(raw.get(field), str) and raw.get(field)
    ])
    visual_dna = str(raw.get("visual_dna") or "").strip()
    if not visual_dna:
        visual_dna = _join_unique([appearance, clothing, description if appearance != description else ""])
    seed = _coerce_seed(raw.get("seed"), deterministic_character_seed(project_id, key))
    profile: dict[str, Any] = {
        "key": key,
        "node_id": str(raw.get("node_id") or "").strip() or None,
        "name": name,
        "role": str(raw.get("role") or "").strip(),
        "gender": str(raw.get("gender") or "").strip(),
        "appearance": appearance,
        "clothing": clothing,
        "visual_dna": visual_dna,
        "seed": seed,
        "ref_image": str(raw.get("ref_image")).strip() if isinstance(raw.get("ref_image"), str) and raw.get("ref_image").strip() else None,
        "anchor_shot_id": str(raw.get("anchor_shot_id")).strip() if isinstance(raw.get("anchor_shot_id"), str) and raw.get("anchor_shot_id").strip() else None,
    }
    return profile


def extract_characters_from_manifest(manifest: ProjectManifest) -> dict[str, dict[str, Any]]:
    """Collect character definitions from story_package, shot contexts, locks, and dialogue."""
    extracted: dict[str, dict[str, Any]] = {}

    def register(raw_item: dict[str, Any] | str) -> None:
        if isinstance(raw_item, str):
            name = raw_item.strip()
            if not name or name.lower() in _SKIP_SPEAKERS:
                return
            existing_key = _match_bible_key(extracted, name)
            if existing_key:
                return
            key = _slug_character_key(name)
            extracted[key] = _profile_from_dict(manifest.project_id, key, {"name": name})
            return
        if not isinstance(raw_item, dict):
            return
        node_id = str(raw_item.get("node_id") or "").strip()
        name = str(raw_item.get("name") or node_id).strip()
        if not name or name.lower() in _SKIP_SPEAKERS:
            return
        existing_key = _match_bible_key(extracted, node_id or name) or _match_bible_key(extracted, name)
        key = existing_key or _slug_character_key(node_id or name)
        incoming = _profile_from_dict(manifest.project_id, key, raw_item)
        if key not in extracted:
            extracted[key] = incoming
            return
        current = extracted[key]
        for field in ("node_id", "name", "role", "gender", "appearance", "clothing", "visual_dna", "ref_image", "anchor_shot_id"):
            if not current.get(field) and incoming.get(field):
                current[field] = incoming[field]

    if isinstance(manifest.story_package, dict):
        for char in manifest.story_package.get("characters", []) or []:
            if isinstance(char, (dict, str)):
                register(char)

    for shot in manifest.shots:
        if isinstance(shot.scene_context, dict):
            for char in shot.scene_context.get("characters", []) or []:
                if isinstance(char, (dict, str)):
                    register(char)
        for lock in shot.character_locks:
            if isinstance(lock, str):
                register(lock)
        for line in shot.dialogue_lines:
            if isinstance(line, dict):
                speaker = str(line.get("speaker") or "").strip()
                if speaker and speaker.lower() not in _SKIP_SPEAKERS:
                    register(speaker)

    if not manifest.story_package and isinstance(manifest.story, str) and re.search(r"[\u4e00-\u9fff]", manifest.story):
        _extract_plain_text_chinese_characters(manifest, extracted, register)

    return extracted


def _extract_plain_text_chinese_characters(
    manifest: ProjectManifest,
    extracted: dict[str, dict[str, Any]],
    register,
) -> None:
    """Heuristically extract named roles, clothing, appearance, and gender from Chinese plain text."""
    story = manifest.story

    for match in _ATTIRE_ROLE_RE.finditer(story):
        clothing = match.group("clothing").strip(" ，,、")
        desc_role = match.group("desc_role").strip()
        role_noun = match.group("role").strip()
        tail = match.group("tail").strip()
        cand_name = _pick_recurring_name(tail, story)
        target_name = cand_name or (role_noun if story.count(role_noun) >= 2 else "")
        if not target_name:
            continue
        register({
            "name": target_name,
            "role": role_noun,
            "appearance": desc_role,
            "clothing": clothing,
            "gender": _infer_text_gender(desc_role + " " + target_name),
        })

    for match in _ARTICLE_ROLE_RE.finditer(story):
        appearance = match.group("appearance").strip(" ，,、")
        role_noun = match.group("role").strip()
        tail = match.group("tail").strip()
        cand_name = _pick_recurring_name(tail, story)
        target_name = cand_name or (role_noun if story.count(role_noun) >= 2 else "")
        if not target_name:
            sent_start = max((story.rfind(sep, 0, match.start()) for sep in "。！？\n"), default=-1) + 1
            sent_end_candidates = [pos for sep in "。！？\n" if (pos := story.find(sep, match.end())) != -1]
            sent_end = min(sent_end_candidates) if sent_end_candidates else len(story)
            sentence_after = story[match.end():sent_end]
            sentence_full = story[sent_start:sent_end]
            for alias in _ROLE_ALIAS_POOL:
                if alias in sentence_after and story.count(alias) >= 2:
                    target_name = alias
                    break
            if not target_name:
                for alias in _ROLE_ALIAS_POOL:
                    if alias in sentence_full and story.count(alias) >= 2:
                        target_name = alias
                        break
        if not target_name:
            continue
        app_desc = f"{appearance}{role_noun}"
        register({
            "name": target_name,
            "role": role_noun,
            "appearance": app_desc,
            "gender": _infer_text_gender(app_desc + " " + target_name),
        })

    # Enrich already-discovered speakers/roles with attire clauses and pronoun gender.
    known_names = [str(prof.get("name") or "").strip() for prof in extracted.values() if prof.get("name")]
    for profile in extracted.values():
        name = str(profile.get("name") or "").strip()
        if not name:
            continue
        if not profile.get("clothing"):
            escaped = re.escape(name)
            attire_before = re.search(
                rf"(?:身穿|穿着|穿|披着|一袭)(?P<clothing>[^，。；！？“”\n]{{2,20}}?)的[\u4e00-\u9fff]{{0,6}}?{escaped}",
                story,
            )
            attire_after = re.search(
                rf"{escaped}(?:身穿|穿着|穿|披着|一袭)(?P<clothing>[^，。；！？“”\n]{{2,20}})",
                story,
            )
            found_attire = attire_before or attire_after
            if found_attire:
                clothing = found_attire.group("clothing").strip(" ，,、")
                if clothing:
                    profile["clothing"] = clothing
                    profile["visual_dna"] = _join_unique([
                        str(profile.get("appearance") or ""),
                        clothing,
                    ])
        if not profile.get("gender"):
            inferred = _infer_text_gender(" ".join(str(profile.get(k) or "") for k in ("role", "appearance", "visual_dna", "name")))
            if not inferred:
                for shot in manifest.shots:
                    present = [other for other in known_names if _name_in_text(other, shot.text)]
                    if present == [name]:
                        has_f = any(p in shot.text for p in _FEMALE_PRONOUNS)
                        has_m = any(p in shot.text for p in _MALE_PRONOUNS)
                        if has_f and not has_m:
                            inferred = "female"
                            break
                        if has_m and not has_f:
                            inferred = "male"
                            break
            if inferred:
                profile["gender"] = inferred


def sync_character_bible(
    manifest: ProjectManifest,
    *,
    project_root: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Sync extracted story characters into manifest.character_bible without overwriting edits."""
    extracted = extract_characters_from_manifest(manifest)
    bible = manifest.character_bible
    for key, profile in extracted.items():
        matched = _match_bible_key(bible, key) or (
            _match_bible_key(bible, profile["name"]) if profile.get("name") else None
        )
        if matched is None:
            bible[key] = profile
            continue
        existing = bible[matched]
        if not existing.get("key"):
            existing["key"] = matched
        for field in ("node_id", "name", "role", "gender", "appearance", "clothing", "visual_dna", "ref_image", "anchor_shot_id"):
            if not existing.get(field) and profile.get(field):
                existing[field] = profile[field]
        existing["seed"] = _coerce_seed(
            existing.get("seed"),
            deterministic_character_seed(manifest.project_id, matched),
        )
    # Apply verified character DNA patches from active flywheel playbook without clobbering edits.
    from .flywheel import load_active_playbook

    playbook = load_active_playbook(manifest, project_root)
    if playbook:
        for patch in playbook.get("character_dna_patches") or []:
            if not isinstance(patch, dict):
                continue
            char_name = str(patch.get("character_name") or "").strip()
            if not char_name:
                continue
            matched = _match_bible_key(bible, char_name)
            if not matched:
                continue
            existing = bible[matched]
            for field in ("appearance", "clothing", "visual_dna"):
                if not existing.get(field) and patch.get(field):
                    existing[field] = str(patch[field]).strip()
    # Normalize any manually entered bible entries
    for key, entry in list(bible.items()):
        if not isinstance(entry, dict):
            bible[key] = _profile_from_dict(manifest.project_id, key, {"name": key})
            continue
        entry.setdefault("key", key)
        entry.setdefault("name", key)
        entry["seed"] = _coerce_seed(
            entry.get("seed"),
            deterministic_character_seed(manifest.project_id, key),
        )
        if not entry.get("visual_dna"):
            entry["visual_dna"] = _join_unique(
                [str(entry.get("appearance") or ""), str(entry.get("clothing") or "")]
            )
    return bible


def upsert_character_profile(
    manifest: ProjectManifest,
    name_or_id: str,
    *,
    name: str | None = None,
    appearance: str | None = None,
    clothing: str | None = None,
    visual_dna: str | None = None,
    seed: int | None = None,
    ref_image: str | None = None,
    role: str | None = None,
    gender: str | None = None,
    clear_ref: bool = False,
) -> dict[str, Any]:
    sync_character_bible(manifest)
    bible = manifest.character_bible
    matched = _match_bible_key(bible, name_or_id)
    key = matched or _slug_character_key(name_or_id)
    if key not in bible:
        bible[key] = _profile_from_dict(
            manifest.project_id,
            key,
            {"name": name or name_or_id},
        )
    profile = bible[key]
    if name is not None and name.strip():
        profile["name"] = name.strip()
    if appearance is not None:
        profile["appearance"] = appearance.strip()
    if clothing is not None:
        profile["clothing"] = clothing.strip()
    if role is not None:
        profile["role"] = role.strip()
    if gender is not None:
        profile["gender"] = gender.strip()
    if visual_dna is not None:
        profile["visual_dna"] = visual_dna.strip()
    elif appearance is not None or clothing is not None:
        profile["visual_dna"] = _join_unique(
            [str(profile.get("appearance") or ""), str(profile.get("clothing") or "")]
        )
    if seed is not None:
        profile["seed"] = _coerce_seed(seed, deterministic_character_seed(manifest.project_id, key))
    if clear_ref:
        profile["ref_image"] = None
        profile["anchor_shot_id"] = None
    elif ref_image is not None:
        profile["ref_image"] = ref_image.strip() or None
    return profile


def _name_in_text(name: str, text: str) -> bool:
    cleaned = name.strip()
    if not cleaned or not text:
        return False
    pattern = re.escape(cleaned)
    if cleaned.isascii():
        pattern = r"(?<!\w)" + pattern + r"(?!\w)"
    return re.search(pattern, text) is not None


def _profile_matches_pronoun(profile: dict[str, Any], pronoun: str) -> bool:
    blob = " ".join(
        str(profile.get(f) or "")
        for f in ("gender", "appearance", "visual_dna", "role", "name")
    ).lower()
    if pronoun in _FEMALE_PRONOUNS:
        return _hint_matches(blob, _FEMALE_HINTS)
    if pronoun in _MALE_PRONOUNS:
        return _hint_matches(blob, _MALE_HINTS)
    return False


def _direct_shot_characters(
    manifest: ProjectManifest,
    shot: Shot,
    bible: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    def add_profile(raw_entry: dict[str, Any] | str) -> None:
        if isinstance(raw_entry, str):
            token = raw_entry.strip()
            if not token or token.lower() in _SKIP_SPEAKERS:
                return
            key = _match_bible_key(bible, token) or _slug_character_key(token)
            profile = dict(bible.get(key) or _profile_from_dict(manifest.project_id, key, {"name": token}))
        elif isinstance(raw_entry, dict):
            node_id = str(raw_entry.get("node_id") or "").strip()
            name = str(raw_entry.get("name") or node_id).strip()
            if not name and not node_id:
                return
            key = _match_bible_key(bible, node_id or name) or _match_bible_key(bible, name) or _slug_character_key(node_id or name)
            base = dict(bible.get(key) or _profile_from_dict(manifest.project_id, key, raw_entry))
            # Prefer user-defined bible DNA if present, fallback to raw_entry attributes.
            for field in ("appearance", "clothing", "visual_dna", "name", "node_id", "role", "gender"):
                if not base.get(field) and raw_entry.get(field):
                    base[field] = raw_entry[field]
            profile = base
        else:
            return
        dedupe_key = str(profile.get("key") or profile.get("node_id") or profile.get("name") or "").lower()
        if dedupe_key and dedupe_key not in seen_keys:
            seen_keys.add(dedupe_key)
            resolved.append(profile)

    for lock in shot.character_locks:
        add_profile(lock)
    if isinstance(shot.scene_context, dict) and shot.scene_context.get("_strict_character_locks"):
        return resolved

    raw_context_chars = shot.scene_context.get("characters") if isinstance(shot.scene_context, dict) else None
    if isinstance(raw_context_chars, list):
        for item in raw_context_chars:
            add_profile(item)
    elif isinstance(raw_context_chars, str) and raw_context_chars.strip():
        for part in re.split(r"[，,、;；]+", raw_context_chars):
            if part.strip():
                add_profile(part.strip())

    for line in shot.dialogue_lines:
        if isinstance(line, dict):
            speaker = str(line.get("speaker") or "").strip()
            if speaker and speaker.lower() not in _SKIP_SPEAKERS:
                add_profile(speaker)

    ctx_dict = shot.scene_context if isinstance(shot.scene_context, dict) else {}
    raw_loc = str(ctx_dict.get("location") or "")
    if "以动作描述中的" in raw_loc:
        raw_loc = ""
    raw_frm = str(ctx_dict.get("framing") or "")
    text_blob = " ".join(
        part
        for part in (
            shot.text,
            str(ctx_dict.get("visual_action") or ""),
            raw_loc,
            raw_frm,
        )
        if part
    )
    if text_blob:
        for key, profile in bible.items():
            name = str(profile.get("name") or key).strip()
            aliases = [str(a).strip() for a in (profile.get("aliases") or []) if str(a).strip()]
            if (name and _name_in_text(name, text_blob)) or any(_name_in_text(a, text_blob) for a in aliases):
                add_profile(profile)

    return resolved


def resolve_shot_characters(manifest: ProjectManifest, shot: Shot) -> list[dict[str, Any]]:
    """Resolve characters for `shot`, carrying over pronouns within the same scene_id."""
    bible = manifest.character_bible
    direct = _direct_shot_characters(manifest, shot, bible)
    if direct:
        return direct

    text_blob = " ".join(
        part
        for part in (
            shot.text,
            str(shot.scene_context.get("visual_action") or "") if isinstance(shot.scene_context, dict) else "",
        )
        if part
    )
    has_female = any(p in text_blob for p in _FEMALE_PRONOUNS)
    has_male = any(p in text_blob for p in _MALE_PRONOUNS)
    if not (has_female or has_male):
        return []

    prior_candidates: list[dict[str, Any]] = []
    for candidate in manifest.shots:
        if candidate.id == shot.id:
            break
        if candidate.scene_id == shot.scene_id:
            found = _direct_shot_characters(manifest, candidate, bible)
            if found:
                prior_candidates = found
    if not prior_candidates:
        return []

    if has_female and not has_male:
        matched = [p for p in prior_candidates if _profile_matches_pronoun(p, "她")]
        if matched:
            return matched
    if has_male and not has_female:
        matched = [p for p in prior_candidates if _profile_matches_pronoun(p, "他")]
        if matched:
            return matched
    return prior_candidates


def _shot_references_character(manifest: ProjectManifest, shot: Shot, target_key: str, target_profile: dict[str, Any]) -> bool:
    candidates = {
        target_key.lower(),
        str(target_profile.get("node_id") or "").strip().lower(),
        str(target_profile.get("name") or "").strip().lower(),
    } - {""}
    for profile in resolve_shot_characters(manifest, shot):
        keys = {
            str(profile.get("key") or "").strip().lower(),
            str(profile.get("node_id") or "").strip().lower(),
            str(profile.get("name") or "").strip().lower(),
        } - {""}
        if candidates & keys:
            return True
    for lock in shot.character_locks:
        if lock.strip().lower() in candidates:
            return True
    return False


def invalidate_character_shots(
    manifest: ProjectManifest,
    character_key: str,
    *,
    invalidate_audio: bool = False,
) -> list[str]:
    """Invalidate cached image/video/optional-audio artifacts for shots bound to `character_key`."""
    sync_character_bible(manifest)
    matched_key = _match_bible_key(manifest.character_bible, character_key)
    key = matched_key or character_key.strip()
    profile = manifest.character_bible.get(key) or {"name": key}
    invalidated: list[str] = []
    for shot in manifest.shots:
        if _shot_references_character(manifest, shot, key, profile):
            shot.assets.pop("image", None)
            shot.assets.pop("video", None)
            if invalidate_audio:
                shot.assets.pop("audio", None)
            shot.image_task.clear()
            shot.cache_key = None
            shot.mode = "preview"
            if shot.status in ("generated", "done", "assembled", "imaged", "completed", "approved", "failed"):
                shot.status = "planned"
            invalidated.append(shot.id)
    if invalidated:
        manifest.output = None
        if manifest.status in ("done", "completed", "assembled"):
            manifest.status = "planned"
    return invalidated


def encode_image_data_uri(image_path: str | Path) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"reference image not found: {image_path}")
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def format_character_visuals(profiles: list[dict[str, Any]], *, max_chars: int = 150) -> str:
    if not profiles:
        return "未指定"
    chunks: list[str] = []
    for profile in profiles:
        name = str(profile.get("name") or profile.get("key") or "角色").strip()
        dna = str(profile.get("visual_dna") or "").strip()
        if not dna:
            dna = _join_unique(
                [str(profile.get("appearance") or ""), str(profile.get("clothing") or "")]
            )
        if dna:
            chunks.append(f"{name}({dna})")
        else:
            chunks.append(name)
    joined = "；".join(chunks)
    if len(joined) <= max_chars:
        return joined
    if max_chars <= 8:
        return joined[:max_chars]
    # Trim DNA segments proportionally while preserving every character's name.
    per_char_budget = max(12, (max_chars - len(profiles) + 1) // len(profiles))
    shortened: list[str] = []
    for profile in profiles:
        name = str(profile.get("name") or profile.get("key") or "角色").strip()
        dna = str(profile.get("visual_dna") or "").strip() or _join_unique(
            [str(profile.get("appearance") or ""), str(profile.get("clothing") or "")]
        )
        if dna and per_char_budget > len(name) + 4:
            allowed_dna = per_char_budget - len(name) - 2
            shortened.append(f"{name}({dna[:allowed_dna]})")
        else:
            shortened.append(name)
    return "；".join(shortened)[:max_chars]


def _resolve_scene_continuity_context(manifest: ProjectManifest, shot: Shot) -> tuple[str | None, str | None]:
    """Carry forward scene location/lighting from earlier shots in the same scene_id."""
    location = shot.scene_context.get("location") if isinstance(shot.scene_context, dict) else None
    lighting = None
    if isinstance(shot.scene_context, dict):
        lighting = shot.scene_context.get("lighting") or shot.scene_context.get("time")
    if location and lighting:
        return None, None
    inherited_location: str | None = None
    inherited_lighting: str | None = None
    for prior in manifest.shots:
        if prior.id == shot.id:
            break
        if prior.scene_id != shot.scene_id or not isinstance(prior.scene_context, dict):
            continue
        if not location and not inherited_location and isinstance(prior.scene_context.get("location"), str):
            val = prior.scene_context["location"].strip()
            if val and "以动作描述中的" not in val:
                inherited_location = val
        if not lighting and not inherited_lighting:
            val = str(prior.scene_context.get("lighting") or prior.scene_context.get("time") or "").strip()
            if val and "以动作描述中的" not in val:
                inherited_lighting = val
    return inherited_location, inherited_lighting


def _has_meaningful_visual_dna(profiles: list[dict[str, Any]]) -> bool:
    return any(
        bool(str(p.get("visual_dna") or p.get("appearance") or p.get("clothing") or "").strip())
        for p in profiles
    )


def _trim_action_for_budget(action: str, overflow: int) -> str:
    if overflow <= 0:
        return action
    if _ACTION_BOILERPLATE in action:
        core, tail = action.split(_ACTION_BOILERPLATE, 1)
        new_core_len = max(16, len(core) - overflow)
        return core[:new_core_len].rstrip(" ，,。;；") + _ACTION_BOILERPLATE + tail
    new_len = max(16, len(action) - overflow)
    return action[:new_len].rstrip(" ，,。;；")


def _apply_shot_prompt_overrides(
    shot: Shot,
    scene: dict[str, Any],
    seed: int | None,
    *,
    allow_seed_override: bool = True,
) -> tuple[dict[str, Any], int | None]:
    prompt_override = getattr(shot, "prompt_override", None)
    if isinstance(prompt_override, str) and prompt_override.strip():
        override_text = prompt_override.strip()
        scene["description"] = override_text
        scene["action"] = (
            override_text
            if _ACTION_BOILERPLATE in override_text
            else override_text + _ACTION_BOILERPLATE
        )

    negative_override = getattr(shot, "negative_override", None)
    if isinstance(negative_override, str) and negative_override.strip():
        base_neg = str(scene.get("negative") or "无文字水印、无畸形手指、无多余人物、无闪烁").strip()
        parts: list[str] = []
        seen: set[str] = set()
        for raw in (base_neg, negative_override.strip()):
            for tok in re.split(r"[，,、;；]+", raw):
                cleaned = tok.strip()
                if cleaned and cleaned not in seen:
                    seen.add(cleaned)
                    parts.append(cleaned)
        if parts:
            scene["negative"] = "，".join(parts)

    seed_override = getattr(shot, "seed_override", None)
    if allow_seed_override and seed_override is not None:
        seed = _coerce_seed(seed_override, seed or 1)
    return scene, seed


def build_consistent_image_scene(
    manifest: ProjectManifest,
    shot: Shot,
    base_scene: dict[str, Any],
    *,
    for_candidates: bool = False,
    candidate_count: int = 1,
    project_root: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Enrich an image prompt scene dict and compute transport conditioning."""
    from .flywheel import apply_flywheel_to_image_scene

    mode, style_preset = validate_consistency_settings(
        manifest.consistency_mode,
        manifest.style_preset,
    )
    scene = dict(base_scene)
    allow_seed_override = not (for_candidates and candidate_count > 1)
    if mode == "off":
        scene, seed_off = _apply_shot_prompt_overrides(
            shot, scene, None, allow_seed_override=allow_seed_override
        )
        scene, applied_fw = apply_flywheel_to_image_scene(
            manifest, shot, scene, project_root=project_root
        )
        return scene, {
            "seed": seed_off,
            "ref_img": None,
            "mode": mode,
            "resolved_characters": [],
            **({"applied_flywheel_rules": applied_fw} if applied_fw else {}),
        }

    resolved = resolve_shot_characters(manifest, shot)
    inherited_location, inherited_lighting = _resolve_scene_continuity_context(manifest, shot)
    if inherited_location and (not scene.get("location") or "以动作描述中的地点为准" in str(scene.get("location"))):
        scene["location"] = inherited_location
    if inherited_lighting and (not scene.get("lighting") or "以动作描述中的时间与光线为准" in str(scene.get("lighting"))):
        scene["lighting"] = inherited_lighting

    # Determine seed & optional reference image from primary resolved character (or style lock).
    seed: int | None = None
    ref_img: str | None = None
    if resolved:
        primary = resolved[0]
        key = str(primary.get("key") or primary.get("node_id") or primary.get("name") or shot.id)
        seed = _coerce_seed(primary.get("seed"), deterministic_character_seed(manifest.project_id, key))
        use_ref_img = mode == "anchor_ref" or (mode == "auto" and model_supports_ref_img(manifest.image_model))
        if use_ref_img and str(primary.get("anchor_shot_id") or "") != shot.id:
            ref_path = primary.get("ref_image")
            if isinstance(ref_path, str) and ref_path.strip() and Path(ref_path.strip()).is_file():
                ref_img = encode_image_data_uri(ref_path.strip())
    elif style_preset is not None:
        seed = deterministic_character_seed(manifest.project_id, f"style:{style_preset}")

    used_tail_relay = False
    if ref_img is None and isinstance(shot.scene_context, dict):
        relay_ref = shot.scene_context.get("_relay_ref_image")
        if isinstance(relay_ref, str) and relay_ref.strip() and Path(relay_ref.strip()).is_file():
            ref_img = encode_image_data_uri(relay_ref.strip())
            used_tail_relay = True

    if for_candidates and candidate_count > 1:
        seed = None

    # Only overwrite base_scene['characters'] when characters have bible visual DNA
    # or pronoun carry-over resolved a character missing from shot.scene_context.
    if resolved and (
        _has_meaningful_visual_dna(resolved)
        or not shot.scene_context.get("characters")
    ):
        scene["characters"] = format_character_visuals(resolved, max_chars=150)

    style_prompt = STYLE_PRESETS[style_preset]["prompt"] if style_preset else ""
    synthetic_style_field: str | None = None
    if style_prompt:
        if "mood" not in shot.scene_context and not scene.get("mood"):
            scene["mood"] = style_prompt
            synthetic_style_field = "mood"
        elif "continuity" not in shot.scene_context and not scene.get("continuity"):
            scene["continuity"] = style_prompt
            synthetic_style_field = "continuity"
        else:
            action = str(scene.get("action") or "")
            if _ACTION_BOILERPLATE in action:
                head, tail = action.split(_ACTION_BOILERPLATE, 1)
                scene["action"] = f"{head}；{style_prompt}{_ACTION_BOILERPLATE}{tail}"
            elif action:
                scene["action"] = f"{action}；{style_prompt}"
            else:
                scene["action"] = style_prompt

    if style_preset == "stylized_3d_anime":
        preset_neg = STYLE_PRESETS["stylized_3d_anime"].get("negative", "")
        if preset_neg:
            base_neg = str(scene.get("negative") or "").strip()
            parts: list[str] = []
            seen_neg: set[str] = set()
            for raw_neg in (base_neg, preset_neg):
                for tok in re.split(r"[，,、;；]+", raw_neg):
                    cleaned = tok.strip()
                    if cleaned and cleaned not in seen_neg:
                        seen_neg.add(cleaned)
                        parts.append(cleaned)
            if parts:
                scene["negative"] = "，".join(parts)

    scene, seed = _apply_shot_prompt_overrides(
        shot, scene, seed, allow_seed_override=allow_seed_override
    )
    scene, applied_fw = apply_flywheel_to_image_scene(
        manifest, shot, scene, project_root=project_root
    )

    # Enforce strict <= 480-char total composed prompt budget (< 500 Wan limit).
    def estimated_length() -> int:
        total = _COMPOSE_OVERHEAD
        for field in ("location", "characters", "action", "framing", "lighting", "mood", "continuity"):
            val = scene.get(field)
            if isinstance(val, str):
                total += len(val)
        return total

    overflow = estimated_length() - _MAX_PROMPT_BUDGET
    if overflow > 0 and synthetic_style_field:
        current_style = str(scene.get(synthetic_style_field) or "")
        trimmed_style = current_style[: max(8, len(current_style) - overflow)]
        if len(trimmed_style) < len(current_style):
            scene[synthetic_style_field] = trimmed_style
        overflow = estimated_length() - _MAX_PROMPT_BUDGET

    if overflow > 0 and resolved and isinstance(scene.get("characters"), str):
        new_char_cap = max(24, len(scene["characters"]) - overflow)
        scene["characters"] = format_character_visuals(resolved, max_chars=new_char_cap)
        overflow = estimated_length() - _MAX_PROMPT_BUDGET

    if overflow > 0 and isinstance(scene.get("action"), str):
        scene["action"] = _trim_action_for_budget(scene["action"], overflow)

    conditioning: dict[str, Any] = {
        "seed": seed,
        "ref_img": ref_img,
        "mode": mode,
        "resolved_characters": resolved,
    }
    if used_tail_relay:
        conditioning["relay_from_prev_tail"] = True
    if applied_fw:
        conditioning["applied_flywheel_rules"] = applied_fw
    return scene, conditioning


def shot_consistency_fingerprint(
    manifest: ProjectManifest,
    shot: Shot,
    *,
    project_root: str | Path | None = None,
) -> str:
    from .flywheel import flywheel_shot_signature

    mode, style_preset = validate_consistency_settings(
        manifest.consistency_mode,
        manifest.style_preset,
    )
    fw_sig = flywheel_shot_signature(manifest, shot, project_root=project_root)
    override_parts = [
        str(getattr(shot, "prompt_override", None) or ""),
        str(getattr(shot, "negative_override", None) or ""),
        int(getattr(shot, "seed_override", None) or 0),
        fw_sig,
    ]
    has_overrides = any(override_parts)
    if mode == "off" and not has_overrides:
        return ""
    resolved = resolve_shot_characters(manifest, shot) if mode != "off" else []
    inherited_location, inherited_lighting = (
        _resolve_scene_continuity_context(manifest, shot) if mode != "off" else (None, None)
    )
    if not resolved and style_preset is None and not inherited_location and not inherited_lighting and not has_overrides:
        return ""
    use_ref_img = mode == "anchor_ref" or (mode == "auto" and model_supports_ref_img(manifest.image_model))
    parts: list[Any] = [mode, style_preset or "", inherited_location or "", inherited_lighting or ""]
    for p in resolved:
        ref_path = str(p.get("ref_image") or "").strip()
        ref_sig = ""
        if use_ref_img and ref_path and str(p.get("anchor_shot_id") or "") != shot.id:
            path_obj = Path(ref_path)
            if path_obj.is_file():
                stat = path_obj.stat()
                ref_sig = f"{ref_path}:{stat.st_size}:{int(stat.st_mtime)}"
        parts.append(
            [
                str(p.get("key") or p.get("name") or ""),
                str(p.get("appearance") or ""),
                str(p.get("clothing") or ""),
                str(p.get("visual_dna") or ""),
                int(p.get("seed") or 0),
                ref_sig,
            ]
        )
    if has_overrides:
        parts.append(override_parts)
    digest = hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()[:16]
    return digest


def record_auto_anchor_frame(
    manifest: ProjectManifest,
    root: str | Path,
    shot: Shot,
    image_path: str | Path,
) -> list[str]:
    """In anchor_ref mode, capture the first generated shot image as an anchor ref."""
    mode, _ = validate_consistency_settings(manifest.consistency_mode, manifest.style_preset)
    if mode != "anchor_ref":
        return []
    path = Path(image_path)
    if not path.is_file():
        return []
    sync_character_bible(manifest)
    resolved = resolve_shot_characters(manifest, shot)
    updated: list[str] = []
    for char in resolved:
        key = _match_bible_key(manifest.character_bible, str(char.get("key") or char.get("name") or ""))
        if not key:
            continue
        profile = manifest.character_bible[key]
        existing_ref = profile.get("ref_image")
        if isinstance(existing_ref, str) and existing_ref.strip() and Path(existing_ref.strip()).is_file():
            continue
        profile["ref_image"] = str(path)
        profile["anchor_shot_id"] = shot.id
        updated.append(key)
    return updated


def safe_character_filename(key: str) -> str:
    """Return a collision-free, ASCII-safe filename stem for a character key."""
    cleaned = str(key or "").strip() or "character"
    ascii_part = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned).strip("_")
    if ascii_part and ascii_part == cleaned:
        return ascii_part
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:8]
    return f"{ascii_part}_{digest}" if ascii_part else f"char_{digest}"


def build_photo_to_3d_anime_prompt(
    profile: dict[str, Any],
    *,
    deidentify_traits: str | None = None,
    style_preset: str | None = "stylized_3d_anime",
) -> dict[str, str]:
    """Build de-identified 3D bone-structure stylization prompt from a real reference portrait."""
    visual = format_character_visuals([profile], max_chars=160)
    deid_head = (
        f"原创脱敏改造（{deidentify_traits.strip()}）；"
        if isinstance(deidentify_traits, str) and deidentify_traits.strip()
        else ""
    )
    style_suffix = ""
    if style_preset and style_preset in STYLE_PRESETS and style_preset != "stylized_3d_anime":
        style_suffix = f"；画风基调：{STYLE_PRESETS[style_preset]['prompt']}"
    prompt = (
        f"{deid_head}严格提取所附真人参考照片的三维面部骨相比例（立体鼻梁与山根转折、颧弓饱满度、利落下颌线折角、眼窝立体深浅与唇形体积），"
        "完全剥离真人照片皮肤毛孔质感，重塑为次世代半写实3D动漫角色标准设定母版（正面微侧15度立体半身像，纯色简洁影棚背景）；"
        "原创化重构动漫发色层次、瞳孔高光几何与原创服饰领口细节，"
        "严禁保留可识别真人肖像身份，严禁撞脸任何已知商业番剧或游戏角色"
        + (f"；角色设定：{visual}" if visual else "")
        + style_suffix
    )
    lighting = (
        "45度立体伦勃朗柔光主光配侧逆轮廓光（Rim Light），次表面散射（SSS）细腻立体肤质，"
        "凸显3D鼻梁与下颌线骨相体积"
    )
    framing = "正面微侧15度半身3D骨相标准设定母版"
    negative = (
        "真人实拍照片皮肤毛孔，直接复制真人身份，热门番剧角色同款发型服饰，"
        "扁平赛璐璐2D纸片脸，正面死白平光，多格分屏，文字水印"
    )
    return {
        "prompt": prompt,
        "negative": negative,
        "lighting": lighting,
        "framing": framing,
    }


class PortraitDeidentificationError(ValueError):
    """Raised when a stylized 3D character master stays too close biometrically to a real source photo."""


def _detect_face_box_or_center(rgb_arr) -> tuple[tuple[int, int, int, int], int]:
    import cv2

    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    casc_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    faces = ()
    if casc_path.is_file():
        cascade = cv2.CascadeClassifier(str(casc_path))
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(24, 24))
    if len(faces) > 0:
        x, y, bw, bh = max(faces, key=lambda b: int(b[2]) * int(b[3]))
        pad_x = max(2, int(round(bw * 0.10)))
        pad_y = max(2, int(round(bh * 0.10)))
        x0 = max(0, int(x) - pad_x)
        y0 = max(0, int(y) - pad_y)
        x1 = min(w, int(x + bw) + pad_x)
        y1 = min(h, int(y + bh) + pad_y)
        return (x0, y0, x1, y1), int(len(faces))
    return (int(w * 0.20), int(h * 0.12), int(w * 0.80), int(h * 0.78)), 0


def _cosine_clamped(a, b) -> float:
    import numpy as np

    va = a.astype(np.float32).ravel()
    vb = b.astype(np.float32).ravel()
    va = va - va.mean()
    vb = vb - vb.mean()
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-6:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(va, vb) / denom)))


def verify_portrait_deidentification(
    source_photo_path: str | Path,
    stylized_portrait_path: str | Path,
    *,
    max_similarity: float | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Audit that a stylized 3D character master is biometrically de-identified from the source photo."""
    import cv2
    import numpy as np
    from PIL import Image

    src_p = Path(source_photo_path)
    sty_p = Path(stylized_portrait_path)
    if not src_p.is_file():
        raise FileNotFoundError(f"source portrait photo not found: {src_p}")
    if not sty_p.is_file():
        raise FileNotFoundError(f"stylized portrait not found: {sty_p}")

    with Image.open(src_p) as s_img:
        src_rgb = np.asarray(s_img.convert("RGB"))
    with Image.open(sty_p) as d_img:
        sty_rgb = np.asarray(d_img.convert("RGB"))

    # Full-frame pixel/re-encoded clone check
    src_thumb = cv2.resize(src_rgb, (160, 90), interpolation=cv2.INTER_AREA)
    sty_thumb = cv2.resize(sty_rgb, (160, 90), interpolation=cv2.INTER_AREA)
    pixel_clone_similarity = _cosine_clamped(src_thumb, sty_thumb)

    # Crop primary face ROI for both portraits
    (sx0, sy0, sx1, sy1), src_face_count = _detect_face_box_or_center(src_rgb)
    (dx0, dy0, dx1, dy1), sty_face_count = _detect_face_box_or_center(sty_rgb)
    raw_src_crop = src_rgb[sy0:sy1, sx0:sx1]
    src_crop = cv2.resize(raw_src_crop, (96, 96), interpolation=cv2.INTER_AREA)
    sty_crop = cv2.resize(sty_rgb[dy0:dy1, dx0:dx1], (96, 96), interpolation=cv2.INTER_AREA)

    src_gray = cv2.cvtColor(src_crop, cv2.COLOR_RGB2GRAY)
    sty_gray = cv2.cvtColor(sty_crop, cv2.COLOR_RGB2GRAY)
    face_struct_cos = _cosine_clamped(src_gray, sty_gray)

    src_grad = np.hypot(
        cv2.Sobel(src_gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(src_gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    sty_grad = np.hypot(
        cv2.Sobel(sty_gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(sty_gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    face_edge_cos = _cosine_clamped(src_grad, sty_grad)

    src_hsv = cv2.cvtColor(src_crop, cv2.COLOR_RGB2HSV)
    sty_hsv = cv2.cvtColor(sty_crop, cv2.COLOR_RGB2HSV)
    h1 = cv2.calcHist([src_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    h2 = cv2.calcHist([sty_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    face_hsv_sim = max(0.0, min(1.0, float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))))

    face_identity_similarity = round(
        0.50 * face_struct_cos + 0.35 * face_edge_cos + 0.15 * face_hsv_sim,
        4,
    )

    # Measure palette richness on the raw unresized crop so resize interpolation does not inflate flat synthetic stubs.
    sampled_raw = raw_src_crop[:: max(1, raw_src_crop.shape[0] // 96), :: max(1, raw_src_crop.shape[1] // 96)]
    unique_5bit_colors = int(len(np.unique((sampled_raw >> 3).reshape(-1, 3), axis=0)))
    unique_exact_colors = int(len(np.unique(sampled_raw.reshape(-1, 3), axis=0)))
    is_photographic_source = unique_5bit_colors >= 32 and unique_exact_colors >= 128

    if pixel_clone_similarity > 0.992:
        raise PortraitDeidentificationError(
            f"角色定妆图与原始真人照片像素高度雷同 (clone_sim={pixel_clone_similarity:.4f} > 0.992)，未完成去身份化风格转绘"
        )

    enforce_biometric = bool(max_similarity is not None or strict or is_photographic_source)
    effective_max = float(max_similarity) if max_similarity is not None else 0.72
    if enforce_biometric and face_identity_similarity > effective_max:
        raise PortraitDeidentificationError(
            f"角色去身份化未达标：脸部特征相似度 {face_identity_similarity:.4f} 超过安全上限 {effective_max:.4f}，"
            "请强化 3D 动漫骨相重塑与发色/瞳色区分"
        )

    return {
        "schema": "portrait-deid-audit/v1",
        "passed": True,
        "face_identity_similarity": face_identity_similarity,
        "pixel_clone_similarity": round(pixel_clone_similarity, 4),
        "is_photographic_source": is_photographic_source,
        "unique_5bit_colors": unique_5bit_colors,
        "faces_detected": {"source": src_face_count, "stylized": sty_face_count},
        "effective_max_similarity": effective_max if enforce_biometric else 0.992,
    }


def generate_character_portrait(
    manifest: ProjectManifest,
    root: str | Path,
    character_key: str,
    *,
    image_provider: Any = None,
    from_photo: str | Path | None = None,
    deidentify_traits: str | None = None,
    three_d_contour: bool | None = None,
    max_identity_similarity: float | None = None,
) -> dict[str, Any]:
    """Generate and persist a dedicated character reference portrait into artifacts/characters/."""
    sync_character_bible(manifest)
    matched = _match_bible_key(manifest.character_bible, character_key)
    key = matched or _slug_character_key(character_key)
    profile = manifest.character_bible.get(key)
    if profile is None:
        profile = upsert_character_profile(manifest, character_key)

    from .providers import build_project_image_provider

    provider = image_provider or build_project_image_provider(manifest, Path(root))
    portrait_dir = Path(root) / "artifacts" / "characters"
    portrait_dir.mkdir(parents=True, exist_ok=True)
    safe_file = safe_character_filename(key)
    output_path = portrait_dir / f"{safe_file}.png"

    saved_mode = manifest.consistency_mode
    saved_ref = profile.get("ref_image")

    if from_photo is not None:
        photo_path = Path(from_photo).expanduser().resolve()
        if not photo_path.is_file():
            raise FileNotFoundError(f"source portrait photo not found: {from_photo}")
        spec = build_photo_to_3d_anime_prompt(
            profile,
            deidentify_traits=deidentify_traits,
            style_preset=manifest.style_preset or "stylized_3d_anime",
        )
        portrait_shot = Shot(
            id=f"char-{safe_file}",
            text=spec["prompt"],
            scene_id="character-bible",
            character_locks=[key],
            prompt_override=spec["prompt"],
            negative_override=spec["negative"],
            scene_context={
                "visual_action": spec["prompt"],
                "location": "纯色简洁深灰立体影棚背景",
                "lighting": spec["lighting"],
                "framing": spec["framing"],
                "characters": [profile],
                "_relay_ref_image": str(photo_path),
            },
        )
        try:
            profile["ref_image"] = None
            manifest.consistency_mode = "auto"
            saved_path = None
            audit_result: dict[str, Any] | None = None
            for attempt in (0, 1):
                saved_path = provider.generate(portrait_shot, output_path)
                try:
                    audit_result = verify_portrait_deidentification(
                        photo_path,
                        saved_path,
                        max_similarity=max_identity_similarity,
                    )
                    break
                except PortraitDeidentificationError:
                    if attempt == 0:
                        boost = "；强制重构五官比例、眉眼间距、发色瞳色与3D动漫光影着色，严禁贴近原照片真人脸部像素纹理"
                        portrait_shot.prompt_override = f"{portrait_shot.prompt_override or spec['prompt']}{boost}"
                        portrait_shot.text = portrait_shot.prompt_override
                        portrait_shot.scene_context["visual_action"] = portrait_shot.prompt_override
                        portrait_shot.negative_override = (
                            f"{portrait_shot.negative_override or spec['negative']}，真人照片直出，换皮滤镜，原照片五官复刻"
                        )
                        portrait_shot.assets.clear()
                        portrait_shot.cache_key = None
                        continue
                    try:
                        Path(saved_path).unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise
        except Exception:
            profile["ref_image"] = saved_ref
            raise
        finally:
            manifest.consistency_mode = saved_mode
        profile["ref_image"] = str(saved_path)
        profile["anchor_shot_id"] = portrait_shot.id
        profile["stylized_from_photo"] = True
        profile["source_photo_sha256"] = hashlib.sha256(photo_path.read_bytes()).hexdigest()[:16]
        if audit_result is not None:
            profile["biometric_audit"] = audit_result
        if isinstance(deidentify_traits, str) and deidentify_traits.strip():
            profile["deidentify_traits"] = deidentify_traits.strip()
        if saved_ref and str(saved_ref) != str(saved_path):
            invalidate_character_shots(manifest, key)
        return profile

    visual = format_character_visuals([profile], max_chars=180)
    use_3d = (
        bool(three_d_contour)
        if three_d_contour is not None
        else (getattr(manifest, "style_preset", None) == "stylized_3d_anime")
    )
    if use_3d:
        action_text = (
            f"单人角色标准设定立绘，正面微侧15度半身构图，立体3D面部骨相轮廓，高挺鼻梁与清晰下颌线转折，"
            f"纯色简洁深灰背景，人物：{visual}"
        )
        lighting_text = "45度立体柔光主光配柔和侧逆轮廓光，凸显3D鼻梁与下颌线骨相体积"
        framing_text = "正面微侧15度立体半身标准立绘"
    else:
        action_text = f"单人角色标准设定立绘，正面半身构图，纯色简洁浅灰背景，人物：{visual}"
        lighting_text = "均匀柔和正面棚拍光"
        framing_text = "正面半身标准立绘"

    portrait_shot = Shot(
        id=f"char-{safe_file}",
        text=action_text,
        scene_id="character-bible",
        character_locks=[key],
        scene_context={
            "visual_action": action_text,
            "location": "纯色简洁影棚背景",
            "lighting": lighting_text,
            "framing": framing_text,
            "characters": [profile],
        },
    )
    try:
        manifest.consistency_mode = "prompt_lock"
        saved_path = provider.generate(portrait_shot, output_path)
    finally:
        manifest.consistency_mode = saved_mode
    profile["ref_image"] = str(saved_path)
    profile["anchor_shot_id"] = portrait_shot.id
    if saved_ref and str(saved_ref) != str(saved_path):
        invalidate_character_shots(manifest, key)
    return profile
