"""Offline tests for the optional WebGUI password gate + sliding idle session."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from peaks_vr import web  # noqa: E402
from peaks_vr.web import auth  # noqa: E402
from peaks_vr.labels import LabelStore  # noqa: E402
from peaks_vr.web.flagging import PlaybackMirror, create_app  # noqa: E402

SECRET = "test-secret-key"


def _app(tmp_path, password=None):
    return TestClient(create_app(PlaybackMirror(),
                                 LabelStore(tmp_path / "l.json"),
                                 media_root=None,
                                 cache_root=str(tmp_path / "cache"),
                                 model="fake"))


def test_open_when_no_password(tmp_path, monkeypatch):
    monkeypatch.delenv("PEAKS_VR_PASSWORD", raising=False)
    c = _app(tmp_path)
    assert c.get("/api/state").status_code == 200          # wide open, as before
    assert c.get("/").status_code == 200


def test_gate_blocks_without_session(tmp_path, monkeypatch):
    monkeypatch.setenv("PEAKS_VR_PASSWORD", "hunter2")
    monkeypatch.setenv("PEAKS_VR_SECRET", SECRET)
    c = _app(tmp_path)
    assert c.get("/api/state").status_code == 401          # api → 401
    r = c.get("/", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"
    assert c.get("/login").status_code == 200              # login page is open


def test_login_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("PEAKS_VR_PASSWORD", "hunter2")
    monkeypatch.setenv("PEAKS_VR_SECRET", SECRET)
    c = _app(tmp_path)
    assert c.post("/api/login", json={"password": "wrong"}).status_code == 401
    r = c.post("/api/login", json={"password": "hunter2"})
    assert r.status_code == 200 and auth.COOKIE in r.cookies
    assert c.get("/api/state").status_code == 200          # cookie now carried
    # logout clears it
    c.post("/api/logout")
    c.cookies.clear()
    assert c.get("/api/state").status_code == 401


def test_expired_cookie_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("PEAKS_VR_PASSWORD", "hunter2")
    monkeypatch.setenv("PEAKS_VR_SECRET", SECRET)
    c = _app(tmp_path)
    expired, _ = auth.make_cookie(SECRET.encode(), -10)     # already in the past
    c.cookies.set(auth.COOKIE, expired)
    assert c.get("/api/state").status_code == 401


def test_polls_do_not_renew_but_activity_does(tmp_path, monkeypatch):
    monkeypatch.setenv("PEAKS_VR_PASSWORD", "hunter2")
    monkeypatch.setenv("PEAKS_VR_SECRET", SECRET)
    c = _app(tmp_path)
    c.post("/api/login", json={"password": "hunter2"})
    # a background poll must not reset the idle clock (no fresh cookie)
    assert "set-cookie" not in {k.lower() for k in
                                c.get("/api/state").headers.keys()}
    # a real interaction (ping) slides the session forward (fresh cookie)
    assert "set-cookie" in {k.lower() for k in
                            c.post("/api/ping").headers.keys()}


def test_login_body_available():
    # models are defined only when pydantic is present (the pragma branch)
    assert web.flagging.LoginBody is not None
