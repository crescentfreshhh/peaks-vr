"""Real scikit-learn classifier tests (sklearn is installed in the dev env)."""

import numpy as np
import pytest

from peaks_vr.classifier import TasteClassifier


def _separable_data(n=60, dim=16, seed=0):
    """Two linearly separable clusters: class 1 around +1, class 0 around -1."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(1.0, 0.3, size=(n, dim)).astype(np.float32)
    neg = rng.normal(-1.0, 0.3, size=(n, dim)).astype(np.float32)
    X = np.vstack([pos, neg])
    y = np.array([1] * n + [0] * n)
    return X, y


def test_train_and_predict_separates_classes():
    X, y = _separable_data()
    clf = TasteClassifier(kind="logreg", model_name="fake", profile="apex").train(X, y)
    assert clf.fitted
    # a clearly-positive point scores high, clearly-negative scores low
    hi = clf.predict_proba(np.ones((1, 16), dtype=np.float32))[0]
    lo = clf.predict_proba(-np.ones((1, 16), dtype=np.float32))[0]
    assert hi > 0.8 and lo < 0.2


def test_predict_proba_shape_and_range():
    X, y = _separable_data()
    clf = TasteClassifier().train(X, y)
    scores = clf.predict_proba(X)
    assert scores.shape == (X.shape[0],)
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_empty_predict():
    X, y = _separable_data()
    clf = TasteClassifier().train(X, y)
    assert clf.predict_proba(np.zeros((0, 16), dtype=np.float32)).shape == (0,)


def test_requires_both_classes():
    X = np.ones((10, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="both"):
        TasteClassifier().train(X, np.ones(10, dtype=int))


def test_dim_mismatch_rejected():
    X, y = _separable_data(dim=16)
    clf = TasteClassifier().train(X, y)
    with pytest.raises(ValueError, match="dim"):
        clf.predict_proba(np.ones((1, 8), dtype=np.float32))


def test_predict_before_train_raises():
    with pytest.raises(RuntimeError):
        TasteClassifier().predict_proba(np.ones((1, 4), dtype=np.float32))


def test_save_load_roundtrip(tmp_path):
    X, y = _separable_data()
    clf = TasteClassifier(kind="logreg", model_name="dinov2", profile="apex:heels")
    clf.train(X, y)
    before = clf.predict_proba(X)

    path = clf.save(tmp_path / "m.pkl")
    loaded = TasteClassifier.load(path)
    assert loaded.model_name == "dinov2" and loaded.profile == "apex:heels"
    assert loaded.dim == 16
    np.testing.assert_allclose(loaded.predict_proba(X), before, atol=1e-6)
