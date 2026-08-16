"""VR format detection — projection, stereo layout, and field of view.

Before a VR frame can be de-warped (see :mod:`peaks_vr.reprojection`) we must
know how it is encoded. VR libraries are a zoo: 180 vs 360, side-by-side (SBS)
vs top-bottom (TB / over-under), and a spread of fisheye variants (MKX200,
RF52, VRCA220, fisheye190, …), at 4K–8K with varied FOV. Each combination needs
its own reprojection parameters, so classification is the input to everything
downstream — and a *misclassification* silently corrupts the embeddings.

Signals, in rough order of reliability (README, "The fix", step 1):

  1. **Filename conventions** — the strongest signal in practice. Studios and
     the tools that tag VR encode the format right in the name:
     ``_180_sbs``, ``_MKX200``, ``_FISHEYE190``, ``_TB``, ``_oculus``, …
  2. **Stash tags** — the library manager often carries explicit format tags.
  3. **Aspect ratio** — a fallback: SBS 180 tends toward ~1:1 per eye (2:1
     full frame), TB stacks the eyes vertically, etc.

`stash-vr` already implements this classification well — study/borrow its logic
as coverage expands. This module starts with the **most common case, 180° SBS**
(equirect and fisheye), and is meant to grow one projection at a time.

Status: scaffold. `detect_from_filename` covers the common tokens; tag- and
aspect-based detection and the long tail of fisheye variants are TODO.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Projection(str, Enum):
    """How the sphere is flattened into the frame."""

    EQUIRECT = "equirect"      # standard equirectangular (180 or 360)
    FISHEYE = "fisheye"        # lens fisheye (MKX200, RF52, fisheye190, …)
    UNKNOWN = "unknown"


class StereoLayout(str, Enum):
    """How the two eyes are packed into one frame."""

    SBS = "sbs"                # side-by-side: left eye = left half
    TB = "tb"                  # top-bottom / over-under: left eye = top half
    MONO = "mono"              # single image, no stereo pair
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class VRFormat:
    """A scene's decoded VR geometry — the input to reprojection.

    ``fov_deg`` is the horizontal field of view of the *source* projection
    (e.g. 180 for 180° content, 200 for an MKX200 fisheye). ``confidence`` in
    [0, 1] lets callers gate on classifier certainty and fall back to a manual
    override when it is low (an open risk called out in the README).
    """

    projection: Projection
    layout: StereoLayout
    fov_deg: float | None
    confidence: float = 0.0
    source: str = ""           # which signal decided this ("filename", "tag", …)

    @property
    def is_stereo(self) -> bool:
        return self.layout in (StereoLayout.SBS, StereoLayout.TB)

    @property
    def is_known(self) -> bool:
        return (
            self.projection is not Projection.UNKNOWN
            and self.layout is not StereoLayout.UNKNOWN
        )


# --- filename tokens --------------------------------------------------------

# Named fisheye encodings → their nominal horizontal FOV. Extend as coverage
# grows; the values drive v360 in the reprojection stage.
_FISHEYE_FOV: dict[str, float] = {
    "mkx200": 200.0,
    "mkx220": 220.0,
    "vrca220": 220.0,
    "rf52": 190.0,
    "fisheye190": 190.0,
    "fisheye": 180.0,
}

_LAYOUT_TOKENS: dict[str, StereoLayout] = {
    "sbs": StereoLayout.SBS,
    "lr": StereoLayout.SBS,
    "tb": StereoLayout.TB,
    "ou": StereoLayout.TB,
    "mono": StereoLayout.MONO,
}


def _tokens(name: str) -> set[str]:
    """Lower-cased alphanumeric tokens from a filename (splits on _, -, ., space)."""
    return {t for t in re.split(r"[^a-z0-9]+", name.lower()) if t}


def detect_from_filename(name: str) -> VRFormat:
    """Best-effort classification from a filename alone — the strongest single
    signal. Returns a :class:`VRFormat`; ``confidence`` reflects how many
    independent tokens agreed. Unknown fields stay ``UNKNOWN``/``None`` so a
    caller can escalate to tags, aspect ratio, or a manual override.
    """
    toks = _tokens(name)

    # Projection + FOV from an explicit fisheye token, else default to equirect
    # when we see a VR marker at all.
    projection = Projection.UNKNOWN
    fov: float | None = None
    hits = 0
    for tok, deg in _FISHEYE_FOV.items():
        if tok in toks:
            projection, fov, hits = Projection.FISHEYE, deg, hits + 1
            break

    if "180" in toks:
        fov = fov or 180.0
        hits += 1
        if projection is Projection.UNKNOWN:
            projection = Projection.EQUIRECT
    elif "360" in toks:
        fov = fov or 360.0
        hits += 1
        if projection is Projection.UNKNOWN:
            projection = Projection.EQUIRECT

    # Stereo layout
    layout = StereoLayout.UNKNOWN
    for tok, lay in _LAYOUT_TOKENS.items():
        if tok in toks:
            layout, hits = lay, hits + 1
            break

    confidence = min(1.0, hits / 3.0)
    return VRFormat(
        projection=projection,
        layout=layout,
        fov_deg=fov,
        confidence=confidence,
        source="filename",
    )


def detect_from_tags(tags: list[str]) -> VRFormat:
    """Classify from Stash tag names. TODO: map common VR tag vocabularies
    (studio tags, stash-vr's tags) onto :class:`VRFormat`. Reuse the token
    logic above once the tag vocabulary is pinned down."""
    raise NotImplementedError("tag-based VR format detection not yet implemented")


def _layout_from_aspect(aspect_ratio: float) -> StereoLayout:
    """Full-frame aspect → stereo layout: SBS packs two eyes side by side (wide,
    ~2:1), TB stacks them (tall, ~1:2). The band in between is left UNKNOWN so a
    weaker signal doesn't override an explicit assumption."""
    if aspect_ratio >= 1.5:
        return StereoLayout.SBS
    if aspect_ratio <= 1.1:
        return StereoLayout.TB
    return StereoLayout.UNKNOWN


def detect(name: str, *, tags: list[str] | None = None,
           aspect_ratio: float | None = None,
           assume: str | None = None) -> VRFormat:
    """Best guess at a scene's VR format, fusing available signals.

    Precedence: the **filename** wins whenever it fully identifies the format
    (so working, hinted files are unchanged). When it doesn't, the **aspect
    ratio** fills in the stereo layout (SBS vs TB — the axis most likely to vary
    within a library), and finally the **assume** spec fills any remaining gaps
    (projection / FOV / layout) so an un-annotated file is still de-warped rather
    than skipped. ``assume`` is a format token string parsed with the same logic
    as filenames — e.g. ``"180_sbs"``, ``"mkx200_tb"``, ``"fisheye190_sbs"``.
    """
    fmt = detect_from_filename(name)
    if fmt.is_known:
        return fmt

    projection, layout, fov = fmt.projection, fmt.layout, fmt.fov_deg
    source_bits = ["filename"] if fmt.confidence else []

    if layout is StereoLayout.UNKNOWN and aspect_ratio:
        layout = _layout_from_aspect(aspect_ratio)
        if layout is not StereoLayout.UNKNOWN:
            source_bits.append("aspect")

    if assume:
        a = detect_from_filename(assume)
        if projection is Projection.UNKNOWN:
            projection = a.projection
        if fov is None:
            fov = a.fov_deg
        if layout is StereoLayout.UNKNOWN:
            layout = a.layout
        source_bits.append("assumed")

    return VRFormat(
        projection=projection, layout=layout, fov_deg=fov,
        confidence=0.3 if "assumed" in source_bits else fmt.confidence,
        source="+".join(source_bits) or "none",
    )
