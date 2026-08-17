"""DJ taste-profile curation — build your taste in bulk, without the headset.

Feature #2 (flagging) grows a taste profile one ❤️ at a time *while a scene plays
in HereSphere*. That's great for fine-tuning, but it's slow to bootstrap and it's
blocked whenever headset connectivity is down. This module bootstraps the same
profile from the **already-embedded library**: it proposes a contact sheet of
frames sampled across every scene, you thumb-up the ones that match your taste,
and each 👍 becomes a positive label — exactly the signal
:func:`peaks_vr.recommend.recommend_from_labels` already consumes.

**Categories.** VR sessions cover distinct acts, and the taste model is already
*multi-modal*: :func:`peaks_vr.scoring.make_similarity_scorer` with ``reduce="max"``
scores a candidate by its single nearest liked frame, so cowgirl likes pull
cowgirl neighbours and blowjob likes pull blowjob neighbours — no averaged
centroid blurring them together. Categories are therefore an **organisational**
layer, not a different matcher: each like is optionally tagged, stored as the
sub-profile ``base:category`` (the same ``profile:tag`` convention the engine
already uses, e.g. ``apex:heels``). That gives per-category counts now and, for
the DJ later, round-robin balancing across acts + per-category tuning — while the
global match still spans every liked frame via :func:`liked_vectors`.

Everything here reuses existing machinery: :func:`peaks_vr.pipeline.gather_candidates`
to propose frames, :class:`peaks_vr.labels.LabelStore` to record likes,
:func:`peaks_vr.pipeline.build_training_set` to resolve likes back to vectors.
"""

from __future__ import annotations

import numpy as np

from .cache import EmbeddingCache
from .pipeline import build_training_set, gather_candidates, safe_tag
from .scoring import make_similarity_scorer

# Below this many liked frames the suggestions are pure random exploration; above
# it they shift to active learning (mostly "more like what you like", a few
# random) so the profile sharpens instead of wandering.
WARM_AT = 12


def category_profiles(store, base: str) -> list[str]:
    """Every profile that belongs to this taste base: ``base`` itself plus any
    ``base:category`` sub-profile."""
    prefix = base + ":"
    return [p for p in store.profiles() if p == base or p.startswith(prefix)]


def _liked_ids(store, base: str) -> set[tuple[str, float]]:
    """(key, rounded-time) of every frame already liked under this base (any
    category), so suggestions never re-show something you've rated."""
    ids: set[tuple[str, float]] = set()
    for prof in category_profiles(store, base):
        ids |= store.labeled_ids(prof)
    return ids


def liked_vectors(store, cache: EmbeddingCache, model_name: str,
                  base: str) -> np.ndarray:
    """All positively-liked frame vectors across the base and every category —
    the multi-modal reference set the matcher scores against."""
    rows = []
    for prof in category_profiles(store, base):
        X, y = build_training_set(store, cache, model_name, prof)
        if X.shape[0]:
            rows.append(X[y == 1])
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(rows, axis=0)


def suggest_frames(store, cache: EmbeddingCache, model_name: str, base: str, *,
                   count: int = 60, seed: int | None = None) -> list[dict]:
    """Propose ``count`` frames to thumb-up, sampled across all embedded scenes.

    Cold start (few likes) → pure random exploration. Warm → active learning:
    mostly frames near your current taste plus a few random, so you reinforce and
    expand categories. Already-liked frames are excluded. Returns dicts of
    ``{key, path, time, score}`` the UI renders via ``/api/preview?path=&time=``.
    """
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    exclude = _liked_ids(store, base)
    liked = liked_vectors(store, cache, model_name, base)

    if liked.shape[0] >= WARM_AT:
        score_frames = make_similarity_scorer(liked, reduce="max")
        top_per_scene, random_per_scene = 2, 1
    else:  # cold: everything random, no scoring signal yet
        score_frames = lambda vecs: np.zeros(len(vecs), dtype=np.float32)  # noqa: E731
        top_per_scene, random_per_scene = 0, 3

    cands = gather_candidates(
        cache, model_name, score_frames,
        top_per_scene=top_per_scene, random_per_scene=random_per_scene,
        seed=seed, limit=count, exclude=exclude,
    )
    return [{"key": c.key, "path": c.path, "time": round(c.time, 3),
             "score": round(c.score, 4)} for c in cands]


def resolve_profile(base: str, category: str | None) -> str:
    """The label profile a like lands in: ``base`` or ``base:category``."""
    cat = (category or "").strip()
    return f"{base}:{safe_tag(cat)}" if cat else base


def like_frame(store, key: str, time: float, path: str | None, base: str,
               category: str | None = None, label: int = 1) -> str:
    """Record a thumbs-up (``label=1``) or thumbs-down (0) for a frame. Returns
    the resolved profile. Caller persists via ``store.save()``."""
    profile = resolve_profile(base, category)
    store.add(key, float(time), int(label), profile, scene_id=path)
    return profile


def unlike_frame(store, key: str, time: float, base: str,
                 category: str | None = None) -> bool:
    """Remove a previously recorded like (toggle off). Returns True if one was
    dropped. Caller persists via ``store.save()``."""
    profile = resolve_profile(base, category)
    ident = store._id(key, time, profile)
    if ident in store._labels:
        del store._labels[ident]
        return True
    return False


def taste_summary(store, base: str) -> dict:
    """Per-category like counts + the total, for the Taste tab header."""
    cats = []
    total = 0
    for prof in sorted(category_profiles(store, base)):
        pos, _ = store.counts(prof)
        if pos == 0:
            continue
        name = prof[len(base) + 1:] if prof.startswith(base + ":") else "(untagged)"
        cats.append({"name": name, "profile": prof, "count": pos})
        total += pos
    cats.sort(key=lambda c: c["count"], reverse=True)
    return {"base": base, "total": total, "categories": cats}
