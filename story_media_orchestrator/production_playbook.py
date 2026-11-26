"""Production playbook contracts derived from Xinhuo Ji V1-V5 delivery loops.

Codifies repeatable engineering gates for serialized story-video production:
1. Minimal shot handoff contract (`ShotHandoffSpec`) and validator (`validate_shot_handoff`).
2. Four-gate promotion state machine (`design_approved` -> `keyframe_approved` ->
   `video_approved` -> `master_included`) with explicit evidence & input-fingerprint
   invalidation (preventing stale cache reuse when keyframes/prompts/audio change).
3. Anchor-based timeline insertion (`resolve_anchored_timeline`) supporting V4.1/V5 style
   patch-in shots (`insert_after` / `insert_before`) without disturbing baseline shot IDs.
4. Five-bucket rough-cut triage (`triage_rough_cut_gap`) mapping observed viewer gaps to
   the earliest responsible stage rather than blind full-act regeneration.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

PLAYBOOK_SCHEMA = "production-playbook/v1"

PROMOTION_STAGES: tuple[str, ...] = (
    "design_approved",
    "keyframe_approved",
    "video_approved",
    "master_included",
)

ROUGH_CUT_GAP_TYPES: dict[str, dict[str, Any]] = {
    "missing_motivation_or_cost": {
        "earliest_stage": "story_beat",
        "rework_scope": "insert_causal_bridge_shots",
        "allow_full_act_regen": False,
        "recommended_actions": [
            "revise_scene_beat_with_choice_and_cost",
            "insert_anchored_bridge_or_reaction_shot",
            "render_local_window_rough_cut_with_neighbors",
        ],
    },
    "missing_anticipation_or_reaction": {
        "earliest_stage": "action_breakdown",
        "rework_scope": "resplit_single_beat_and_keyframes",
        "allow_full_act_regen": False,
        "recommended_actions": [
            "split_into_anticipation_contact_followthrough",
            "render_fresh_start_keyframe_before_motion_begins",
            "prefer_clean_tail_relay_on_same_axis_else_fresh_keyframe",
        ],
    },
    "keyframe_asset_defect": {
        "earliest_stage": "keyframe_png",
        "rework_scope": "rework_keyframe_and_refs_only",
        "allow_full_act_regen": False,
        "recommended_actions": [
            "pass_stage_locked_character_and_prop_refs_with_request",
            "keep_peak_or_contact_sheet_images_separate_from_start_keyframe",
            "reverify_png_decode_aspect_and_visual_content_before_video",
        ],
    },
    "video_motion_or_speech_defect": {
        "earliest_stage": "video_render",
        "rework_scope": "single_shot_clean_or_resplit",
        "allow_full_act_regen": False,
        "recommended_actions": [
            "reduce_to_one_visible_action_and_one_result",
            "trim_tail_stall_before_paying_for_rerender_when_opening_is_good",
            "check_voiceover_flag_and_strip_raw_dialogue_from_visual_prompt",
        ],
    },
    "edit_or_sound_discontinuity": {
        "earliest_stage": "edit_and_audio",
        "rework_scope": "cut_points_and_audio_alignment",
        "allow_full_act_regen": False,
        "recommended_actions": [
            "adjust_cut_point_add_reaction_beat_or_short_breath_hold",
            "carry_ambient_sound_across_cut_and_align_sfx_to_impact_frame",
            "preserve_separate_dialogue_sfx_ambience_music_stems",
        ],
    },
}

VALID_VIDEO_MODES = frozenset({"i2va", "fl2va", "i2v", "fl2v", "preview"})


@dataclass(frozen=True)
class PlaybookFinding:
    """Single validation error or warning from playbook contract checks."""

    severity: str
    code: str
    shot_id: str | None
    message: str
    detail: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "shot_id": self.shot_id,
            "message": self.message,
        }
        if self.detail is not None:
            out["detail"] = self.detail
        return out


@dataclass(frozen=True)
class ShotHandoffSpec:
    """Minimal shot handoff contract for reproducible patch-in and review."""

    shot_id: str
    version: str
    story_role: str
    starting_pose: str
    primary_action: str
    visible_result: str
    handoff_state: str
    target_frames: int
    insert_after: str | None = None
    next_shot_id: str | None = None
    gap_evidence: str | None = None
    character_refs: tuple[str, ...] = ()
    prop_refs: tuple[str, ...] = ()
    first_frame: str | None = None
    last_frame: str | None = None
    peak_ref: str | None = None
    video_mode: str = "i2va"
    dialogue_or_voiceover: str | None = None
    soundscape: str | None = None
    acceptance_checks: tuple[str, ...] = ()
    artifact_paths: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ShotHandoffSpec":
        def _seq(val: Any) -> tuple[str, ...]:
            if val is None:
                return ()
            if isinstance(val, str):
                s = val.strip()
                return (s,) if s else ()
            return tuple(str(x).strip() for x in val if str(x).strip())

        raw_frames = data.get("target_frames", 0)
        try:
            frames = int(raw_frames) if not isinstance(raw_frames, bool) else 0
        except (TypeError, ValueError):
            frames = 0

        raw_artifacts = data.get("artifact_paths")
        artifacts = (
            {str(k): str(v) for k, v in raw_artifacts.items() if str(v).strip()}
            if isinstance(raw_artifacts, Mapping)
            else {}
        )
        return cls(
            shot_id=str(data.get("shot_id") or "").strip(),
            version=str(data.get("version") or "").strip(),
            story_role=str(data.get("story_role") or "").strip(),
            starting_pose=str(data.get("starting_pose") or "").strip(),
            primary_action=str(data.get("primary_action") or "").strip(),
            visible_result=str(data.get("visible_result") or "").strip(),
            handoff_state=str(data.get("handoff_state") or "").strip(),
            target_frames=frames,
            insert_after=str(data.get("insert_after") or "").strip() or None,
            next_shot_id=str(data.get("next_shot_id") or "").strip() or None,
            gap_evidence=str(data.get("gap_evidence") or "").strip() or None,
            character_refs=_seq(data.get("character_refs")),
            prop_refs=_seq(data.get("prop_refs")),
            first_frame=str(data.get("first_frame") or "").strip() or None,
            last_frame=str(data.get("last_frame") or "").strip() or None,
            peak_ref=str(data.get("peak_ref") or "").strip() or None,
            video_mode=str(data.get("video_mode") or "i2va").strip().lower(),
            dialogue_or_voiceover=str(data.get("dialogue_or_voiceover") or "").strip() or None,
            soundscape=str(data.get("soundscape") or "").strip() or None,
            acceptance_checks=_seq(data.get("acceptance_checks")),
            artifact_paths=artifacts,
        )


def validate_shot_handoff(
    spec: ShotHandoffSpec | Mapping[str, Any],
    *,
    require_first_frame_exists: bool = False,
    base_dir: Path | None = None,
) -> list[PlaybookFinding]:
    """Validate a minimal shot handoff record against V1-V5 production discipline."""
    item = spec if isinstance(spec, ShotHandoffSpec) else ShotHandoffSpec.from_mapping(spec)
    sid = item.shot_id or None
    findings: list[PlaybookFinding] = []

    required_text = (
        ("shot_id", item.shot_id, "MISSING_SHOT_ID"),
        ("version", item.version, "MISSING_VERSION"),
        ("story_role", item.story_role, "MISSING_STORY_ROLE"),
        ("starting_pose", item.starting_pose, "MISSING_STARTING_POSE"),
        ("primary_action", item.primary_action, "MISSING_PRIMARY_ACTION"),
        ("visible_result", item.visible_result, "MISSING_VISIBLE_RESULT"),
        ("handoff_state", item.handoff_state, "MISSING_HANDOFF_STATE"),
    )
    for field_name, val, code in required_text:
        if not val:
            findings.append(PlaybookFinding("error", code, sid, f"Handoff record missing required field '{field_name}'"))

    if item.target_frames <= 0:
        findings.append(PlaybookFinding("error", "INVALID_TARGET_FRAMES", sid, "target_frames must be a positive integer"))

    if item.video_mode not in VALID_VIDEO_MODES:
        findings.append(
            PlaybookFinding(
                "error",
                "INVALID_VIDEO_MODE",
                sid,
                f"Unsupported video_mode '{item.video_mode}' (expected one of {sorted(VALID_VIDEO_MODES)})",
            )
        )
    elif item.video_mode in {"fl2va", "fl2v"} and not item.last_frame:
        findings.append(
            PlaybookFinding(
                "error",
                "FL2V_MISSING_LAST_FRAME",
                sid,
                f"video_mode='{item.video_mode}' requires an explicit last_frame path",
            )
        )

    if item.first_frame and item.peak_ref and item.first_frame == item.peak_ref:
        findings.append(
            PlaybookFinding(
                "error",
                "PEAK_REF_USED_AS_FIRST_FRAME",
                sid,
                "peak_ref cannot be identical to first_frame; start keyframe must precede the peak action",
            )
        )

    if not item.acceptance_checks:
        findings.append(
            PlaybookFinding(
                "warning",
                "MISSING_ACCEPTANCE_CHECKS",
                sid,
                "Shot handoff lacks explicit acceptance_checks for visual/audio playback verification",
            )
        )

    if require_first_frame_exists:
        if not item.first_frame:
            findings.append(PlaybookFinding("error", "MISSING_FIRST_FRAME", sid, "first_frame path is required"))
        else:
            p = Path(item.first_frame)
            if not p.is_absolute() and base_dir is not None:
                p = (Path(base_dir) / p).resolve()
            if not p.is_file():
                findings.append(
                    PlaybookFinding("error", "FIRST_FRAME_NOT_FOUND", sid, f"first_frame file not found: {p}")
                )

    return findings


def compute_asset_lineage_fingerprint(
    *,
    first_frame: Path | None = None,
    last_frame: Path | None = None,
    prompt_text: str = "",
    audio_path: Path | None = None,
    target_frames: int | None = None,
    video_mode: str = "i2va",
) -> str:
    """Compute a deterministic input fingerprint to invalidate stale video caches on edits."""
    h = hashlib.sha256()
    h.update(f"mode={video_mode}|frames={target_frames}|prompt={prompt_text.strip()}|".encode("utf-8"))
    for label, path in (("first", first_frame), ("last", last_frame), ("audio", audio_path)):
        if path is None:
            h.update(f"|{label}=none".encode("utf-8"))
            continue
        p = Path(path)
        if p.is_file():
            h.update(f"|{label}=".encode("utf-8"))
            h.update(hashlib.sha256(p.read_bytes()).hexdigest().encode("ascii"))
        else:
            h.update(f"|{label}=missing:{p}".encode("utf-8"))
    return h.hexdigest()[:20]


@dataclass
class ShotPromotionRecord:
    """Four-gate promotion status for a shot: design -> keyframe -> video -> master."""

    shot_id: str
    gates: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            stage: {"passed": False, "evidence": None, "input_fingerprint": None}
            for stage in PROMOTION_STAGES
        }
    )

    def set_gate(
        self,
        stage: str,
        *,
        passed: bool,
        evidence: str | None = None,
        input_fingerprint: str | None = None,
    ) -> None:
        if stage not in PROMOTION_STAGES:
            raise ValueError(f"Unknown promotion stage: {stage}")
        idx = PROMOTION_STAGES.index(stage)
        if passed and idx > 0:
            prev_stage = PROMOTION_STAGES[idx - 1]
            if not self.gates[prev_stage].get("passed"):
                raise ValueError(
                    f"Cannot promote shot '{self.shot_id}' to '{stage}' before '{prev_stage}' passes"
                )
        self.gates[stage] = {
            "passed": bool(passed),
            "evidence": evidence,
            "input_fingerprint": input_fingerprint,
        }
        if not passed:
            for downstream in PROMOTION_STAGES[idx + 1 :]:
                self.gates[downstream] = {
                    "passed": False,
                    "evidence": f"invalidated_by:{stage}",
                    "input_fingerprint": None,
                }

    def is_video_cache_fresh(self, current_input_fingerprint: str) -> bool:
        """Return True only when video_approved passed with the exact current input fingerprint."""
        gate = self.gates["video_approved"]
        return bool(
            gate.get("passed")
            and gate.get("input_fingerprint")
            and gate.get("input_fingerprint") == current_input_fingerprint
        )

    def invalidate_for_input_change(self, new_input_fingerprint: str) -> list[str]:
        """Downgrade video_approved and master_included if inputs changed since approval."""
        cleared: list[str] = []
        vgate = self.gates["video_approved"]
        if vgate.get("passed") and vgate.get("input_fingerprint") != new_input_fingerprint:
            for stage in ("video_approved", "master_included"):
                if self.gates[stage].get("passed"):
                    cleared.append(stage)
                self.gates[stage] = {
                    "passed": False,
                    "evidence": "invalidated_by_input_change",
                    "input_fingerprint": None,
                }
        return cleared

    def highest_stage(self) -> str:
        highest = "unverified"
        for stage in PROMOTION_STAGES:
            if self.gates[stage].get("passed"):
                highest = stage
            else:
                break
        return highest

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "highest_stage": self.highest_stage(),
            "gates": self.gates,
        }


def resolve_anchored_timeline(
    baseline_ids: Sequence[str],
    insertions: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Insert patch shots relative to stable baseline anchors without renumbering baseline IDs.

    Each insertion mapping must have `shot_id` and either `insert_after` or `insert_before`
    (optional `order` int breaks ties deterministically while preserving declaration order).
    Chained insertions anchored to an earlier inserted shot are resolved iteratively.
    """
    timeline = [str(s).strip() for s in baseline_ids if str(s).strip()]
    if len(set(timeline)) != len(timeline):
        raise ValueError("baseline_ids contains duplicate shot IDs")

    pending: list[tuple[int, int, dict[str, Any]]] = []
    seen = set(timeline)
    for idx, raw in enumerate(insertions):
        sid = str(raw.get("shot_id") or "").strip()
        if not sid:
            raise ValueError(f"Insertion #{idx} is missing shot_id")
        if sid in seen:
            raise ValueError(f"Duplicate shot_id in timeline insertions: {sid}")
        seen.add(sid)
        after = str(raw.get("insert_after") or "").strip() or None
        before = str(raw.get("insert_before") or "").strip() or None
        if bool(after) == bool(before):
            raise ValueError(f"Insertion '{sid}' must specify exactly one of insert_after or insert_before")
        order = int(raw.get("order", idx))
        pending.append((order, idx, {"shot_id": sid, "insert_after": after, "insert_before": before}))

    pending.sort(key=lambda item: (item[0], item[1]))
    after_cursor: dict[str, str] = {}

    while pending:
        progressed = False
        next_pending: list[tuple[int, int, dict[str, Any]]] = []
        for entry in pending:
            spec = entry[2]
            sid = spec["shot_id"]
            after = spec["insert_after"]
            before = spec["insert_before"]
            if after is not None:
                anchor = after_cursor.get(after, after)
                if anchor in timeline:
                    pos = timeline.index(anchor) + 1
                    timeline.insert(pos, sid)
                    after_cursor[after] = sid
                    progressed = True
                else:
                    next_pending.append(entry)
            else:
                assert before is not None
                if before in timeline:
                    pos = timeline.index(before)
                    timeline.insert(pos, sid)
                    progressed = True
                else:
                    next_pending.append(entry)
        if not progressed:
            unresolved = [item[2]["shot_id"] for item in next_pending]
            raise ValueError(f"Unresolved timeline anchors for shots: {unresolved}")
        pending = next_pending

    return timeline


def triage_rough_cut_gap(
    gap_type: str,
    *,
    version: str,
    shot_id: str,
    missing_for_viewer: str,
    timecode: str | None = None,
    neighbor_shots: Sequence[str] = (),
) -> dict[str, Any]:
    """Map a rough-cut observation to the earliest stage to rework (no blind full-act rerun)."""
    key = str(gap_type or "").strip()
    if key not in ROUGH_CUT_GAP_TYPES:
        raise ValueError(f"Unknown gap_type '{gap_type}'; expected one of {sorted(ROUGH_CUT_GAP_TYPES)}")
    if not str(missing_for_viewer or "").strip():
        raise ValueError("missing_for_viewer must describe what narrative or visual information is missing")
    rule = ROUGH_CUT_GAP_TYPES[key]
    return {
        "schema": PLAYBOOK_SCHEMA,
        "version": str(version).strip(),
        "shot_id": str(shot_id).strip(),
        "timecode": str(timecode).strip() if timecode else None,
        "neighbor_shots": [str(s).strip() for s in neighbor_shots if str(s).strip()],
        "gap_type": key,
        "missing_for_viewer": str(missing_for_viewer).strip(),
        "earliest_stage": rule["earliest_stage"],
        "rework_scope": rule["rework_scope"],
        "allow_full_act_regen": rule["allow_full_act_regen"],
        "recommended_actions": list(rule["recommended_actions"]),
    }


def lint_handoff_manifest(
    manifest_path: Path,
    *,
    require_first_frame_exists: bool = False,
) -> dict[str, Any]:
    """Validate a JSON handoff/patch manifest containing `shots` and optional `baseline_ids`."""
    p = Path(manifest_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    shots = data.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError("Handoff manifest must contain a non-empty 'shots' list")

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for raw in shots:
        for finding in validate_shot_handoff(
            raw,
            require_first_frame_exists=require_first_frame_exists,
            base_dir=p.parent,
        ):
            (errors if finding.severity == "error" else warnings).append(finding.to_dict())

    resolved_timeline: list[str] | None = None
    baseline_ids = data.get("baseline_ids")
    if isinstance(baseline_ids, list) and baseline_ids:
        insertions = [s for s in shots if s.get("insert_after") or s.get("insert_before")]
        try:
            resolved_timeline = resolve_anchored_timeline(baseline_ids, insertions)
        except ValueError as exc:
            errors.append(
                PlaybookFinding("error", "INVALID_ANCHORED_TIMELINE", None, str(exc)).to_dict()
            )

    return {
        "schema": PLAYBOOK_SCHEMA,
        "manifest_path": str(p),
        "ok": len(errors) == 0,
        "shot_count": len(shots),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "resolved_timeline": resolved_timeline,
    }
