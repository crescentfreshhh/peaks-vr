"""Frame embedders.

Two real channels (see docs/ARCHITECTURE.md):
  - DINOv2  → visual *structure* (positions, angles, body type)
  - CLIP    → *nameable* attributes (outfits, heels, ...) and open-vocab text

Plus a deterministic FakeEmbedder so the whole pipeline (sampler → cache →
scorer) can be tested offline without torch or model downloads.

Real models are imported lazily inside the classes, so importing this module is
cheap and torch is only required when you actually instantiate a real embedder
(installed via the `[ml]` extra).

All embedders return float32 arrays of shape (n, dim), L2-normalized.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image


class Embedder(ABC):
    name: str
    dim: int
    # raw-pipeline geometry: ffmpeg reproduces the model's preprocessing as
    # "bicubic resize short side -> raw_resize, center crop -> raw_crop"
    raw_resize: int = 256
    raw_crop: int = 224

    @abstractmethod
    def embed_images(self, images: Sequence["Image"]) -> np.ndarray:
        """Embed a batch of PIL images → (len(images), dim) float32, normalized."""

    def embed_array(self, frames: np.ndarray) -> np.ndarray:
        """Embed (n, raw_crop, raw_crop, 3) uint8 frames from the raw
        pipeline. Subclasses normalize on their own device (GPU when
        available) — no PIL, no CPU resize."""
        raise NotImplementedError(f"{self.name} has no raw-array path")


# --- deterministic fake (tests / pipeline smoke) ----------------------------


class FakeEmbedder(Embedder):
    """Hashes each image's bytes to a stable pseudo-random unit vector.

    Deterministic: the same image always maps to the same vector, so cache
    round-trips and scoring can be tested without any real model.
    """

    def __init__(self, dim: int = 32, name: str = "fake"):
        self.dim = dim
        self.name = name

    def _vec_from_bytes(self, data: bytes) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(data).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)

    def embed_images(self, images: Sequence["Image"]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.empty((len(images), self.dim), dtype=np.float32)
        for i, img in enumerate(images):
            data = img.tobytes() if hasattr(img, "tobytes") else bytes(img)
            out[i] = self._vec_from_bytes(data)
        return out

    def embed_array(self, frames: np.ndarray) -> np.ndarray:
        if frames.shape[0] == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.empty((frames.shape[0], self.dim), dtype=np.float32)
        for i in range(frames.shape[0]):
            out[i] = self._vec_from_bytes(frames[i].tobytes())
        return out


# --- real channels (lazy torch) ---------------------------------------------


# DINOv2 backbones, small → giant. Bigger = more accurate structure at more
# cost. Cache is namespaced per backbone (dino_cache_name), so switching one in
# means re-embedding DINO into its own namespace.
DINO_BACKBONES = (
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitl14",
    "dinov2_vitg14",
)


def dino_cache_name(model_name: str) -> str:
    """Cache subdir for a DINOv2 backbone. The small backbone keeps the legacy
    "dinov2" name (backward compatible); bigger backbones get their own dir so
    their different-dim vectors never collide (e.g. "dinov2-vitb14")."""
    if model_name in ("dinov2_vits14", "dinov2"):
        return "dinov2"
    return "dinov2-" + model_name.replace("dinov2_", "").replace("_", "-")


class DinoV2Embedder(Embedder):
    """DINOv2 ViT features (CLS token). Backbone size is configurable."""

    raw_resize = 256  # matches transforms.Resize(256)
    raw_crop = 224  # matches transforms.CenterCrop(224)
    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(self, model_name: str = "dinov2_vits14", device: str | None = None):
        import torch  # lazy

        self._torch = torch
        self.name = dino_cache_name(model_name)  # cache namespaced by backbone
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(self.device)
        self.dim = int(self.model.embed_dim)
        self._build_transform()
        # normalization constants resident on the device for the raw path
        self._mean_t = torch.tensor(self._MEAN, device=self.device).view(1, 3, 1, 1)
        self._std_t = torch.tensor(self._STD, device=self.device).view(1, 3, 1, 1)

    def _build_transform(self) -> None:
        from torchvision import transforms  # lazy

        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def embed_images(self, images: Sequence["Image"]) -> np.ndarray:
        torch = self._torch
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        batch = torch.stack([self.transform(im.convert("RGB")) for im in images])
        with torch.no_grad():
            feats = self.model(batch.to(self.device))
        feats = torch.nn.functional.normalize(feats, dim=1)
        return feats.cpu().numpy().astype(np.float32)

    def embed_array(self, frames: np.ndarray) -> np.ndarray:
        """Raw path: uint8 (n, 224, 224, 3) → normalize on device → model."""
        torch = self._torch
        if frames.shape[0] == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        t = torch.from_numpy(np.ascontiguousarray(frames))
        with torch.no_grad():
            t = t.to(self.device, non_blocking=True)
            t = t.permute(0, 3, 1, 2).float().div_(255.0)
            t = (t - self._mean_t) / self._std_t
            feats = self.model(t)
            feats = torch.nn.functional.normalize(feats, dim=1)
        return feats.cpu().numpy().astype(np.float32)


def clip_cache_name(model_name: str) -> str:
    """Cache subdir for a CLIP variant. ViT-B-32 keeps the legacy "clip" name
    (backward compatible); every other variant gets its own dir so their
    different-dim vectors never collide (e.g. "clip-vit-l-14")."""
    if model_name == "ViT-B-32":
        return "clip"
    slug = model_name.lower().replace("/", "-").replace(" ", "")
    return f"clip-{slug}"


# The open_clip pretrained weights that pair with each variant, so the GUI can
# offer a bare variant name and we resolve the matching checkpoint for it.
_CLIP_PRETRAINED = {
    "ViT-B-32": "laion2b_s34b_b79k",
    "ViT-L-14": "laion2b_s32b_b82k",
    "ViT-H-14": "laion2b_s32b_b79k",
}


def default_pretrained(model_name: str) -> str:
    """Default open_clip checkpoint for a CLIP variant (falls back to the
    ViT-L-14 laion2b weights for anything unlisted)."""
    return _CLIP_PRETRAINED.get(model_name, "laion2b_s32b_b82k")


class ClipEmbedder(Embedder):
    """OpenCLIP image embedder. Also exposes text embedding for open-vocab
    attribute scoring (e.g. "high heels") used by attribute profiles later.
    """

    raw_resize = 224  # open_clip preprocess: Resize(224) + CenterCrop(224)
    raw_crop = 224
    _MEAN = (0.48145466, 0.4578275, 0.40821073)  # CLIP normalization
    _STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "laion2b_s32b_b82k",
        device: str | None = None,
    ):
        import open_clip  # lazy
        import torch  # lazy

        self._torch = torch
        self.name = clip_cache_name(model_name)  # cache namespaced by variant
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        with torch.no_grad():
            # tokens must follow the model onto its device (CUDA crash otherwise)
            dummy = self.model.encode_text(self.tokenizer(["x"]).to(self.device))
        self.dim = int(dummy.shape[1])
        self._mean_t = torch.tensor(self._MEAN, device=self.device).view(1, 3, 1, 1)
        self._std_t = torch.tensor(self._STD, device=self.device).view(1, 3, 1, 1)

    def embed_images(self, images: Sequence["Image"]) -> np.ndarray:
        torch = self._torch
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        batch = torch.stack([self.preprocess(im.convert("RGB")) for im in images])
        with torch.no_grad():
            feats = self.model.encode_image(batch.to(self.device))
        feats = torch.nn.functional.normalize(feats, dim=1)
        return feats.cpu().numpy().astype(np.float32)

    def embed_array(self, frames: np.ndarray) -> np.ndarray:
        """Raw path: uint8 (n, 224, 224, 3) → normalize on device → model."""
        torch = self._torch
        if frames.shape[0] == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        t = torch.from_numpy(np.ascontiguousarray(frames))
        with torch.no_grad():
            t = t.to(self.device, non_blocking=True)
            t = t.permute(0, 3, 1, 2).float().div_(255.0)
            t = (t - self._mean_t) / self._std_t
            feats = self.model.encode_image(t)
            feats = torch.nn.functional.normalize(feats, dim=1)
        return feats.cpu().numpy().astype(np.float32)

    def embed_text(self, prompts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        tokens = self.tokenizer(list(prompts))
        with torch.no_grad():
            feats = self.model.encode_text(tokens.to(self.device))
        feats = torch.nn.functional.normalize(feats, dim=1)
        return feats.cpu().numpy().astype(np.float32)


_REGISTRY = {
    "fake": FakeEmbedder,
    "dino": DinoV2Embedder,
    "dinov2": DinoV2Embedder,
    "clip": ClipEmbedder,
}


# Canonical embedder name == the cache subdirectory it writes to. Lets train/
# score resolve the cache without instantiating a (torch-heavy) embedder.
_CANONICAL = {"fake": "fake", "dino": "dinov2", "dinov2": "dinov2", "clip": "clip"}


def canonical_name(alias: str) -> str:
    key = alias.lower()
    if key not in _CANONICAL:
        raise ValueError(f"unknown embedder {alias!r}; choices: {sorted(_REGISTRY)}")
    return _CANONICAL[key]


def model_cache_name(alias: str, cfg_embedding) -> str:
    """Backbone-aware cache name for a channel: resolves dino/clip to the dir
    for the *configured* variant (so bigger models get their own namespace)."""
    key = alias.lower()
    if key in ("dino", "dinov2"):
        return dino_cache_name(getattr(cfg_embedding, "dino_model", "dinov2_vits14"))
    if key == "clip":
        return clip_cache_name(getattr(cfg_embedding, "clip_model", "ViT-B-32"))
    return canonical_name(alias)


def get_embedder(name: str, **kwargs) -> Embedder:
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown embedder {name!r}; choices: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)
