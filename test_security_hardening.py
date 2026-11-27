"""Stdlib unittest suite for DPAPI vault, constant-time API-key auth, per-IP 429 limiter, and strict CSP."""
import asyncio
import base64
import os
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

# Stub optional external HTTP/RSS deps when absent so app imports cleanly in minimal environments.
_ORIG_MODULES = {}
for _mod_name, _attrs in {
    "httpx": {
        "AsyncClient": lambda *a, **k: types.SimpleNamespace(),
        "Timeout": lambda *a, **k: None,
    },
    "feedparser": {
        "parse": lambda *_a, **_k: types.SimpleNamespace(entries=[]),
    },
}.items():
    if _mod_name not in sys.modules:
        try:
            __import__(_mod_name)
        except ImportError:
            _ORIG_MODULES[_mod_name] = None
            _stub = types.ModuleType(_mod_name)
            _stub.__dict__.update(_attrs)
            sys.modules[_mod_name] = _stub

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import app as app_mod
    import dpapi_vault
finally:
    for _mod_name, _orig in _ORIG_MODULES.items():
        if _orig is None:
            sys.modules.pop(_mod_name, None)
        else:
            sys.modules[_mod_name] = _orig


def _make_fake_dpapi_runner():
    """Simulate Windows DPAPI Protect/Unprotect via XOR envelope and record calls."""
    calls = []

    def _xor(data: bytes, mask: bytes) -> bytes:
        return bytes(b ^ mask[i % len(mask)] for i, b in enumerate(data))

    def runner(cmd, *, capture_output=True, text=True, env=None, timeout=None, check=False, shell=False):
        calls.append({"cmd": list(cmd), "env": dict(env or {}), "shell": shell, "timeout": timeout})
        script = cmd[-1]
        in_bytes = base64.b64decode(env["_DPAPI_INPUT_B64"], validate=True)
        scope = env.get("_DPAPI_SCOPE", "CurrentUser").encode("utf-8")
        entropy = base64.b64decode(env["_DPAPI_ENTROPY_B64"], validate=True) if env.get("_DPAPI_ENTROPY_B64") else b"default"
        mask = b"DPAPI-MOCK:" + scope + b":" + entropy
        if "::Protect(" in script:
            out_bytes = b"V1|" + _xor(in_bytes, mask)
        elif "::Unprotect(" in script:
            if not in_bytes.startswith(b"V1|"):
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="bad ciphertext")
            out_bytes = _xor(in_bytes[3:], mask)
        else:
            return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="unknown script")
        return subprocess.CompletedProcess(cmd, 0, stdout=base64.b64encode(out_bytes).decode("ascii"), stderr="")

    return runner, calls


def _fake_request(ip: str = "198.51.100.10", headers: dict | None = None, method: str = "POST"):
    norm_headers = {k.lower(): v for k, v in (headers or {}).items()}
    return types.SimpleNamespace(
        method=method,
        headers=norm_headers,
        client=types.SimpleNamespace(host=ip),
    )


async def _asgi_get(asgi_app, path: str) -> tuple[int, dict[str, str], bytes]:
    """Invoke a FastAPI ASGI app directly with stdlib asyncio (no TestClient/httpx needed)."""
    status_code = 0
    resp_headers: dict[str, str] = {}
    body_chunks: list[bytes] = []

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "https",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [(b"host", b"testserver")],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 443),
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = message["status"]
            for k, v in message.get("headers", []):
                resp_headers[k.decode("latin-1").lower()] = v.decode("latin-1")
        elif message["type"] == "http.response.body":
            body_chunks.append(message.get("body", b""))

    await asgi_app(scope, receive, send)
    return status_code, resp_headers, b"".join(body_chunks)


class TestDPAPIVault(unittest.TestCase):
    def test_seal_and_unseal_roundtrip_without_command_interpolation(self):
        runner, calls = _make_fake_dpapi_runner()
        secret = "sk-ğüşöçıİ-$(whoami)\"';|sharpeye_99"
        entropy = "sharpeye-vault-entropy"

        sealed = dpapi_vault.seal_secret(secret, scope="CurrentUser", entropy=entropy, runner=runner)
        self.assertTrue(sealed.startswith(dpapi_vault.DPAPI_PREFIX))
        self.assertNotIn(secret, sealed)

        unsealed = dpapi_vault.unseal_secret(sealed, scope="CurrentUser", entropy=entropy, runner=runner)
        self.assertEqual(unsealed, secret)

        # Ensure zero command-line interpolation and shell=False on both calls.
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertFalse(call["shell"])
            joined_cmd = " ".join(call["cmd"])
            self.assertNotIn(secret, joined_cmd)
            self.assertNotIn(entropy, joined_cmd)
            self.assertNotIn(sealed[len(dpapi_vault.DPAPI_PREFIX):], joined_cmd)
            self.assertIn("_DPAPI_INPUT_B64", call["env"])
            self.assertEqual(call["env"]["_DPAPI_SCOPE"], "CurrentUser")

    def test_dpapi_validation_and_error_handling(self):
        runner, _ = _make_fake_dpapi_runner()
        with self.assertRaises(ValueError):
            dpapi_vault.seal_secret("", runner=runner)
        with self.assertRaises(ValueError):
            dpapi_vault.seal_secret("abc", scope="InvalidScope", runner=runner)
        with self.assertRaises(ValueError):
            dpapi_vault.unseal_secret("plain-secret", runner=runner)
        with self.assertRaises(ValueError):
            dpapi_vault.unseal_secret("dpapi:v1:!!!not-base64!!!", runner=runner)

        secret_probe = "TOP_SECRET_SHARPEYE_DO_NOT_LEAK"

        def failing_runner(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=f"leak {secret_probe}")

        with self.assertRaises(dpapi_vault.DPAPIError) as ctx:
            dpapi_vault.seal_secret(secret_probe, runner=failing_runner)
        self.assertNotIn(secret_probe, str(ctx.exception))

        with patch.object(dpapi_vault.sys, "platform", "linux"):
            with self.assertRaises(dpapi_vault.DPAPIError):
                dpapi_vault.seal_secret("abc")

    def test_load_secret_and_env_unseal(self):
        runner, _ = _make_fake_dpapi_runner()
        sealed = dpapi_vault.seal_secret("my-sharpeye-key", runner=runner)
        env = {"SHARPEYE_API_KEY": sealed, "GEMINI_API_KEY": "plain-gemini-key"}

        self.assertEqual(dpapi_vault.load_secret("SHARPEYE_API_KEY", env=env, runner=runner), "my-sharpeye-key")
        self.assertEqual(dpapi_vault.load_secret("GEMINI_API_KEY", env=env, runner=runner), "plain-gemini-key")
        self.assertEqual(dpapi_vault.load_secret("MISSING_KEY", default="fallback", env=env, runner=runner), "fallback")

        count = app_mod.unseal_dpapi_env_secrets(env=env, runner=runner)
        self.assertEqual(count, 1)
        self.assertEqual(env["SHARPEYE_API_KEY"], "my-sharpeye-key")
        self.assertEqual(env["GEMINI_API_KEY"], "plain-gemini-key")


class TestAuthAndRateLimiting(unittest.TestCase):
    def setUp(self):
        self._orig_auth_limiter = app_mod.auth_failure_limiter
        self._orig_key = os.environ.get("SHARPEYE_API_KEY")
        app_mod.limiter.hits.clear()

    def tearDown(self):
        app_mod.auth_failure_limiter = self._orig_auth_limiter
        if self._orig_key is None:
            os.environ.pop("SHARPEYE_API_KEY", None)
        else:
            os.environ["SHARPEYE_API_KEY"] = self._orig_key
        app_mod.limiter.hits.clear()

    def test_public_backward_compatibility_when_api_key_unset(self):
        os.environ.pop("SHARPEYE_API_KEY", None)
        req = _fake_request(ip="203.0.113.1")
        app_mod.guard(req)  # Must succeed with no Authorization / X-API-Key header

    def test_constant_time_auth_headers_and_dpapi_key_support(self):
        runner, _ = _make_fake_dpapi_runner()
        sealed = dpapi_vault.seal_secret("sharpeye-secret-token", runner=runner)
        os.environ["SHARPEYE_API_KEY"] = sealed

        with patch.object(dpapi_vault.sys, "platform", "win32"), patch.object(
            dpapi_vault.subprocess, "run", side_effect=runner
        ):
            # Missing header -> 401
            with self.assertRaises(app_mod.HTTPException) as ctx:
                app_mod.guard(_fake_request(ip="203.0.113.2"))
            self.assertEqual(ctx.exception.status_code, 401)

            # Wrong key -> 401
            with self.assertRaises(app_mod.HTTPException) as ctx2:
                app_mod.guard(_fake_request(ip="203.0.113.2", headers={"Authorization": "Bearer wrong"}))
            self.assertEqual(ctx2.exception.status_code, 401)

            # Valid Bearer -> passes and resets failure history
            app_mod.guard(_fake_request(ip="203.0.113.2", headers={"Authorization": "Bearer sharpeye-secret-token"}))
            # Valid X-API-Key -> passes
            app_mod.guard(_fake_request(ip="203.0.113.2", headers={"X-API-Key": "sharpeye-secret-token"}))

    def test_malformed_dpapi_key_fails_closed_503(self):
        os.environ["SHARPEYE_API_KEY"] = "dpapi:v1:!!!corrupt!!!"
        with self.assertRaises(app_mod.HTTPException) as ctx:
            app_mod.verify_api_key(_fake_request(ip="203.0.113.9", headers={"X-API-Key": "anything"}))
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertNotIn("corrupt", str(ctx.exception.detail))

    def test_sliding_window_per_ip_429_and_bounded_memory(self):
        now = [1000.0]
        limiter = app_mod.AuthFailureRateLimiter(
            max_failures=3,
            window_seconds=60.0,
            max_tracked_ips=5,
            clock=lambda: now[0],
        )
        app_mod.auth_failure_limiter = limiter
        os.environ["SHARPEYE_API_KEY"] = "valid-sharpeye-123"

        # 3 failed attempts from IP A -> 401 each
        for _ in range(3):
            with self.assertRaises(app_mod.HTTPException) as ctx:
                app_mod.guard(_fake_request(ip="198.51.100.1", headers={"Authorization": "Bearer bad"}))
            self.assertEqual(ctx.exception.status_code, 401)

        # 4th attempt from IP A (even with valid key!) -> 429 with Retry-After
        with self.assertRaises(app_mod.HTTPException) as ctx_429:
            app_mod.guard(_fake_request(ip="198.51.100.1", headers={"Authorization": "Bearer valid-sharpeye-123"}))
        self.assertEqual(ctx_429.exception.status_code, 429)
        self.assertGreaterEqual(int(ctx_429.exception.headers["Retry-After"]), 1)

        # Different IP B is isolated and succeeds
        app_mod.guard(_fake_request(ip="198.51.100.2", headers={"Authorization": "Bearer valid-sharpeye-123"}))

        # Advance clock past sliding window -> IP A unblocks
        now[0] += 61.0
        app_mod.guard(_fake_request(ip="198.51.100.1", headers={"Authorization": "Bearer valid-sharpeye-123"}))

        # Bounded memory test: flood 20 unique IPs -> len stays <= max_tracked_ips (5)
        for i in range(20):
            limiter.record_failure(f"192.0.2.{i}")
        self.assertLessEqual(len(limiter), 5)


class TestStrictCSPAndSecurityHeaders(unittest.TestCase):
    def test_csp_script_src_uses_sha256_without_unsafe_inline(self):
        csp = app_mod.SECURITY_HEADERS["Content-Security-Policy"]
        directives = {
            part.strip().split(" ", 1)[0]: part.strip()
            for part in csp.split(";")
            if part.strip()
        }
        self.assertIn("script-src", directives)
        script_src = directives["script-src"]
        self.assertIn("'sha256-", script_src)
        self.assertNotIn("'unsafe-inline'", script_src)
        self.assertNotIn("'unsafe-eval'", script_src)
        self.assertEqual(directives.get("object-src"), "object-src 'none'")
        self.assertEqual(directives.get("frame-ancestors"), "frame-ancestors 'none'")

    def test_asgi_middleware_emits_strict_security_headers(self):
        status, headers, _ = asyncio.run(_asgi_get(app_mod.app, "/healthz"))
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("x-content-type-options"), "nosniff")
        self.assertEqual(headers.get("x-frame-options"), "DENY")
        self.assertIn("strict-transport-security", headers)
        csp = headers.get("content-security-policy", "")
        script_src = next((p.strip() for p in csp.split(";") if p.strip().startswith("script-src")), "")
        self.assertIn("'sha256-", script_src)
        self.assertNotIn("'unsafe-inline'", script_src)


if __name__ == "__main__":
    unittest.main()
