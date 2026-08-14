"""peaks-vr — taste-driven moment discovery for VR video.

A standalone, VR-only sibling to `peaks` (the 2D taste engine for Stash). It
learns which *moments* inside a VR library you actually like, finds more like
them, and plays them back in the headset (HereSphere) without leaving VR.

This package reuses peaks' proven core — frame sampling, the embedding cache,
the taste model, and similarity search — and adds the VR-specific pieces:

  * ``vr_format``    — detect projection / stereo layout / FOV per scene.
  * ``reprojection`` — de-warp one eye of a stereo VR frame to a flat viewport
                       (ffmpeg ``v360``) so vision models can embed it. This is
                       the load-bearing preprocessing step; see the README.
  * ``heresphere``   — read live playback state from and send commands to the
                       HereSphere headset player (DeoVR-remote compatible).

The ported core installs light (just ``requests`` + ``numpy``); the heavy ML
and web dependencies live behind the ``[ml]`` and ``[web]`` extras, exactly as
in peaks. The offline test suite runs with neither, using ``FakeEmbedder``.
"""

__version__ = "0.1.0"
