"""Password gate for the control-panel WebGUI.

peaks-vr's web UI (Embed / QC / Flag) is otherwise open to anyone on the LAN.
This adds an opt-in shared-password gate with a sliding **idle** timeout: a signed
session cookie is issued on login and renewed on real user activity, so it lapses
after a period of no interaction (default 1 hour) and the next request bounces to
the login page.

It's a single shared secret — a personal, single-user tool, so there are no
accounts. Auth is **enabled only when ``PEAKS_VR_PASSWORD`` is set**; unset means
the UI stays open exactly as before (no lockout on upgrade).

Scope: the HTTP app only. The HereSphere TCP links (timestamp intake, DeoVR
remote) can't send a password, so they're unaffected.

The cookie is ``"{expiry_epoch}.{hex hmac_sha256(secret, expiry)}"`` — stateless,
tamper-evident (verified with :func:`hmac.compare_digest`), and self-expiring. The
signing secret comes from ``PEAKS_VR_SECRET`` or a random key persisted beside the
cache so sessions survive a restart.
"""

from __future__ import annotations

import hmac
import os
import secrets
import time
from hashlib import sha256
from pathlib import Path

COOKIE = "pv_session"


def load_secret(cache_root: str) -> bytes:
    """The HMAC signing key: ``PEAKS_VR_SECRET`` if set, else a random 32-byte key
    persisted to ``<cache_root>/../session.key`` (so sessions survive a restart).
    Falls back to an ephemeral in-memory key if the file can't be written."""
    env = os.environ.get("PEAKS_VR_SECRET")
    if env:
        return env.encode()
    path = Path(cache_root).parent / "session.key"
    try:
        if path.exists():
            data = path.read_bytes().strip()
            if data:
                return data
        key = secrets.token_hex(32).encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(key)
        tmp.replace(path)
        try:
            path.chmod(0o600)
        except OSError:
            pass
        return key
    except OSError:
        return secrets.token_hex(32).encode()


def _sign(secret: bytes, expiry: int) -> str:
    return hmac.new(secret, str(expiry).encode(), sha256).hexdigest()


def make_cookie(secret: bytes, timeout: int) -> tuple[str, int]:
    """A fresh session token valid for ``timeout`` seconds. Returns (value,
    max_age)."""
    expiry = int(time.time()) + int(timeout)
    return f"{expiry}.{_sign(secret, expiry)}", int(timeout)


def verify(secret: bytes, cookie: str | None) -> bool:
    """True iff ``cookie`` carries a valid, unexpired signature."""
    if not cookie or "." not in cookie:
        return False
    exp_s, _, sig = cookie.partition(".")
    try:
        expiry = int(exp_s)
    except ValueError:
        return False
    if not hmac.compare_digest(sig, _sign(secret, expiry)):
        return False
    return time.time() < expiry


def check_password(password: str, supplied: str) -> bool:
    """Constant-time password comparison."""
    return hmac.compare_digest(password.encode(), (supplied or "").encode())


LOGIN_PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>peaks-vr — sign in</title><link rel="icon" href="data:,">
<style>
  :root{--bg:#0b0b0d;--panel:#141417;--panel2:#1b1b20;--fg:#e8e8ea;--dim:#8a8a92;
    --line:#2a2a30;--accent:#c8a24a;--bad:#e0604d}
  *{box-sizing:border-box}html,body{height:100%}
  body{margin:0;background:var(--bg);color:var(--fg);
    font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    display:flex;align-items:center;justify-content:center}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:14px;
    padding:26px;width:100%;max-width:340px;text-align:center}
  h1{font-size:20px;margin:0 0 2px}h1 span{color:var(--accent)}
  .sub{color:var(--dim);font-size:12px;margin-bottom:18px}
  input{width:100%;background:var(--panel2);color:var(--fg);border:1px solid var(--line);
    border-radius:8px;padding:11px 12px;font:inherit;margin-bottom:10px}
  button{width:100%;font:inherit;font-weight:700;border:1px solid var(--accent);
    background:var(--accent);color:#1a1400;border-radius:10px;padding:11px;cursor:pointer}
  .err{color:var(--bad);font-size:12px;min-height:16px;margin-top:8px}
</style></head><body>
<form class="card" id="f">
  <h1><span>peaks</span>-vr</h1>
  <div class="sub">enter password to continue</div>
  <input id="pw" type="password" autofocus autocomplete="current-password" placeholder="Password">
  <button type="submit">Sign in</button>
  <div class="err" id="e"></div>
</form>
<script>
document.getElementById('f').addEventListener('submit', async (ev)=>{
  ev.preventDefault();
  const r = await fetch('/api/login',{method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify({password:document.getElementById('pw').value})});
  if(r.ok){ location.href='/'; }
  else { document.getElementById('e').textContent='Incorrect password.'; document.getElementById('pw').select(); }
});
</script></body></html>"""
