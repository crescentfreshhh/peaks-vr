"""VR reprojection — de-warp one eye of a stereo VR frame to a flat viewport.

**This is the load-bearing preprocessing step** (README, "The core problem: VR
frames break vision models"). A raw VR frame is a side-by-side or over-under
stereo pair of fisheye/equirectangular-warped images. Center-cropping such a
frame lands on the seam between the two eyes, and every pixel is geometrically
distorted — embeddings of that are meaningless, so the taste model would be
blind to VR content.

The fix, per scene:

  1. **Take one eye** — left half of SBS, top half of TB.
  2. **Reproject to a flat perspective viewport** with ffmpeg's ``v360`` filter,
     at a central, forward-facing action FOV (~90–110°). This yields a "normal"
     rectilinear image the vision model understands — and the same viewport is
     reused for thumbnails (raw fisheye previews look terrible).

This module emits the **ffmpeg filtergraph fragment** for that transform, so it
composes with the existing sampler pipeline (:class:`peaks_vr.sampling.FrameSampler`
prepends it before its own resize/crop). Building the filter string is pure and
unit-testable offline; actually running it needs ffmpeg (and, for 4K–8K decode,
NVDEC hardware decode — the real bottleneck).

Status: scaffold. The **180° SBS** case (equirect and fisheye) is implemented as
the first, most-common projection. Other projections raise until their ``v360``
parameters are dialed in — build coverage one projection at a time.
"""

from __future__ import annotations

from dataclasses import dataclass

from .vr_format import Projection, StereoLayout, VRFormat


@dataclass(frozen=True)
class Reprojector:
    """Turns a detected :class:`VRFormat` into an ffmpeg ``v360`` de-warp.

    ``viewport_fov_deg`` is the horizontal FOV of the *output* flat viewport
    (the "action FOV"); ~90–110° keeps the forward action in frame without the
    edge stretch a wider viewport introduces. ``yaw``/``pitch`` aim the viewport
    — VR action often sits below the horizon, so a downward ``pitch`` is a
    plausible second viewport to embed (an open question in the README).
    ``out_size`` is the square edge, in pixels, of the emitted viewport; the
    sampler's own resize/crop takes it to the model's input geometry afterward.
    """

    fmt: VRFormat
    viewport_fov_deg: float = 100.0
    yaw: float = 0.0
    pitch: float = 0.0
    out_size: int = 512

    # --- one-eye crop -------------------------------------------------------

    def _eye_crop(self) -> str:
        """ffmpeg crop selecting the left eye (SBS) or top eye (TB)."""
        if self.fmt.layout is StereoLayout.SBS:
            return "crop=iw/2:ih:0:0"
        if self.fmt.layout is StereoLayout.TB:
            return "crop=iw:ih/2:0:0"
        if self.fmt.layout is StereoLayout.MONO:
            return ""  # nothing to split
        raise ValueError(f"cannot pick an eye for layout {self.fmt.layout!r}")

    # --- v360 de-warp -------------------------------------------------------

    def _v360(self) -> str:
        """The ``v360`` reprojection to a flat forward viewport.

        Currently supports 180° equirect (``input=he``, half-equirectangular)
        and fisheye (``input=fisheye`` with the source lens FOV). Adding a new
        projection means adding its ``input=`` mode and FOV wiring here.
        """
        fov = self.fmt.fov_deg or 180.0

        if self.fmt.projection is Projection.EQUIRECT and fov <= 180.0 + 1e-6:
            in_spec = "input=he"  # half-equirectangular (180°)
        elif self.fmt.projection is Projection.FISHEYE:
            in_spec = f"input=fisheye:ih_fov={fov:g}:iv_fov={fov:g}"
        else:
            raise NotImplementedError(
                "reprojection currently supports 180° SBS/TB equirect and "
                f"fisheye only; got projection={self.fmt.projection.value!r} "
                f"fov={fov!r}. Add its v360 parameters to Reprojector._v360()."
            )

        return (
            f"v360={in_spec}:output=flat"
            f":h_fov={self.viewport_fov_deg:g}:v_fov={self.viewport_fov_deg:g}"
            f":yaw={self.yaw:g}:pitch={self.pitch:g}"
            f":w={self.out_size}:h={self.out_size}"
        )

    # --- public -------------------------------------------------------------

    def ffmpeg_filter(self) -> str:
        """Return the comma-joined filtergraph fragment: one-eye crop → v360.

        Compose this *before* the sampler's resize/crop. Raises for projections
        that aren't supported yet, so an unsupported scene fails loudly rather
        than silently embedding garbage.
        """
        parts = [p for p in (self._eye_crop(), self._v360()) if p]
        return ",".join(parts)

    @classmethod
    def for_format(cls, fmt: VRFormat, **kwargs) -> "Reprojector":
        """Build a reprojector for a detected format. Convenience constructor so
        callers read as ``Reprojector.for_format(detect(name))``."""
        return cls(fmt=fmt, **kwargs)
