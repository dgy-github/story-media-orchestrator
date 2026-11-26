"""Resolve MiniMaxH3ImageToVideo first/last-frame input ports from object_info.

The 2026-11-20 H3 node contract uses optional first_frame/last_frame.
Older graphs used start_image/end_image. Callers must pick one documented
pair from the live schema (or the recorded current default) and must not
emit both alias families on the same node.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

NODE_CLASS = "MiniMaxH3ImageToVideo"
NEW_FRAME_PORTS = ("first_frame", "last_frame")
LEGACY_FRAME_PORTS = ("start_image", "end_image")
ALL_FRAME_PORTS = NEW_FRAME_PORTS + LEGACY_FRAME_PORTS


class FramePortContractError(ValueError):
    """Raised when the H3 image-to-video node has no usable frame-input contract."""


@dataclass(frozen=True)
class FramePortSpec:
    first: str
    last: str
    source: str
    schema_names: frozenset[str] | None = None

    @property
    def family(self) -> str:
        if (self.first, self.last) == NEW_FRAME_PORTS:
            return "new"
        if (self.first, self.last) == LEGACY_FRAME_PORTS:
            return "legacy"
        return "custom"

    def accepts_last(self) -> bool:
        if self.schema_names is None:
            return True
        return self.last in self.schema_names


def node_input_names(
    object_info: Mapping[str, Any],
    *,
    class_type: str = NODE_CLASS,
) -> set[str]:
    """Return required+optional input names for a ComfyUI node class."""
    if not isinstance(object_info, Mapping):
        raise FramePortContractError("object_info must be a mapping")
    node = object_info.get(class_type)
    if not isinstance(node, Mapping):
        raise FramePortContractError(
            f"object_info is missing node class {class_type}"
        )
    inputs = node.get("input")
    if not isinstance(inputs, Mapping):
        raise FramePortContractError(
            f"{class_type} object_info is missing an input schema"
        )
    names: set[str] = set()
    for section in ("required", "optional"):
        block = inputs.get(section) or {}
        if isinstance(block, Mapping):
            names.update(str(key) for key in block)
    return names


def resolve_frame_ports(
    object_info: Mapping[str, Any] | None = None,
    *,
    class_type: str = NODE_CLASS,
) -> FramePortSpec:
    """Choose a single first/last port pair.

    Policy:
    - No schema: default to the current verified contract first_frame/last_frame.
    - If the schema contains any new port, use only the new pair (never also
      emit legacy aliases, even if they are listed).
    - If the schema contains only the legacy pair, use start_image/end_image.
    - If neither pair is present, raise. Do not invent or mix unknown ports.
    """
    if object_info is None:
        return FramePortSpec(*NEW_FRAME_PORTS, source="default_current_contract")

    names = node_input_names(object_info, class_type=class_type)
    has_new = any(port in names for port in NEW_FRAME_PORTS)
    has_legacy = any(port in names for port in LEGACY_FRAME_PORTS)

    if has_new:
        if NEW_FRAME_PORTS[0] not in names:
            raise FramePortContractError(
                f"{class_type} lists a new-family frame port but is missing "
                f"{NEW_FRAME_PORTS[0]}; found {sorted(names)}"
            )
        return FramePortSpec(
            *NEW_FRAME_PORTS,
            source="object_info_new",
            schema_names=frozenset(names),
        )
    if has_legacy:
        if LEGACY_FRAME_PORTS[0] not in names:
            raise FramePortContractError(
                f"{class_type} lists a legacy-family frame port but is missing "
                f"{LEGACY_FRAME_PORTS[0]}; found {sorted(names)}"
            )
        return FramePortSpec(
            *LEGACY_FRAME_PORTS,
            source="object_info_legacy",
            schema_names=frozenset(names),
        )
    raise FramePortContractError(
        f"{class_type} has no first_frame/last_frame or start_image/end_image "
        f"inputs; found {sorted(names)}"
    )


def apply_frame_links(
    inputs: dict[str, Any],
    spec: FramePortSpec,
    *,
    first_link: Any | None = None,
    last_link: Any | None = None,
) -> dict[str, Any]:
    """Write only provided frame links onto the chosen ports.

    Empty/missing links are omitted (T2V must not inject blank frame ports).
    The unused alias family is never written. Asking for a last frame when
    the chosen last port is absent from the schema is an error.
    """
    if first_link is not None:
        inputs[spec.first] = first_link
    if last_link is not None:
        if not spec.accepts_last():
            raise FramePortContractError(
                f"{spec.source} contract does not accept last-frame port {spec.last!r}"
            )
        inputs[spec.last] = last_link
    for alias in ALL_FRAME_PORTS:
        if alias in (spec.first, spec.last):
            continue
        inputs.pop(alias, None)
    return inputs


def frame_ports_present(inputs: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Return which known frame ports are actually set on a node inputs dict."""
    if not isinstance(inputs, Mapping):
        return ()
    return tuple(port for port in ALL_FRAME_PORTS if port in inputs)
