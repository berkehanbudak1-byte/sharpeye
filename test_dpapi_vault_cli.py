"""Zero-dependency stdlib unittest suite for sharpeye dpapi_vault_cli.py."""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpapi_vault  # noqa: E402
import dpapi_vault_cli  # noqa: E402


def _make_fake_dpapi_runner():
    """Simulate Windows DPAPI Protect/Unprotect via XOR envelope and record calls."""
    calls: list[dict[str, object]] = []

    def _xor(data: bytes, mask: bytes) -> bytes:
        return bytes(b ^ mask[i % len(mask)] for i, b in enumerate(data))

    def runner(cmd, *, capture_output=True, text=True, env=None, timeout=None, check=False, shell=False):
        calls.append({"cmd": list(cmd), "env": dict(env or {}), "shell": shell, "timeout": timeout})
        script = cmd[-1]
        in_bytes = base64.b64decode(env["_DPAPI_INPUT_B64"], validate=True)
        scope = env.get("_DPAPI_SCOPE", "CurrentUser").encode("utf-8")
        entropy = (
            base64.b64decode(env["_DPAPI_ENTROPY_B64"], validate=True)
            if env.get("_DPAPI_ENTROPY_B64")
            else b"default"
        )
        mask = b"DPAPI-CLI-MOCK:" + scope + b":" + entropy
        if "::Protect(" in script:
            out_bytes = b"V1|" + _xor(in_bytes, mask)
        elif "::Unprotect(" in script:
            if not in_bytes.startswith(b"V1|"):
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="bad ciphertext")
            out_bytes = _xor(in_bytes[3:], mask)
        else:
            return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="unknown script")
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=base64.b64encode(out_bytes).decode("ascii"),
            stderr="",
        )

    return runner, calls


class TestSharpeyeDpapiVaultCli(unittest.TestCase):
    def test_safe_env_display_name_redacts_posix_and_windows_user_paths(self) -> None:
        self.assertEqual(
            dpapi_vault_cli._safe_env_display_name("/home/alice/projects/sharpeye/.env"),
            ".env",
        )
        self.assertEqual(
            dpapi_vault_cli._safe_env_display_name(r"C:\Users\alice\projects\sharpeye\prod.env"),
            "prod.env",
        )
        self.assertEqual(dpapi_vault_cli._safe_env_display_name("/home/alice/.."), ".env")
        self.assertEqual(dpapi_vault_cli._safe_env_display_name("/home/alice/."), ".env")
        self.assertEqual(dpapi_vault_cli._safe_env_display_name("C:"), ".env")
        self.assertEqual(dpapi_vault_cli._safe_env_display_name(""), ".env")

    def test_sensitive_key_detection_matches_secrets_and_skips_non_secret_config(self) -> None:
        for secret_name in (
            "SHARPEYE_API_KEY",
            "GEMINI_API_KEY",
            "XAI_API_KEY",
            "GROK_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "NEWS_API_KEY",
            "NEWS_TOKEN",
            "NEWS_SECRET",
            "OLLAMA_API_KEY",
            "OLLAMA_TOKEN",
            "OLLAMA_PASSWORD",
            "WEBHOOK_SECRET",
            "DB_PASSWORD",
        ):
            self.assertTrue(dpapi_vault_cli.is_sensitive_key(secret_name), secret_name)

        for non_secret in (
            "LLM_PROVIDER",
            "NEWS_WINDOW",
            "OLLAMA_URL",
            "OLLAMA_MODEL",
            "GEMINI_MODEL",
            "XAI_MODEL",
            "XAI_BASE_URL",
            "TRUST_PROXY",
            "ALLOWED_ORIGINS",
            "MAX_BODY_BYTES",
            "PUBLIC_KEY",
        ):
            self.assertFalse(dpapi_vault_cli.is_sensitive_key(non_secret), non_secret)

    def test_status_zero_leakage_reports_key_states_only(self) -> None:
        runner, _ = _make_fake_dpapi_runner()
        sealed_sharpeye = dpapi_vault.seal_secret("sharpeye-top-secret-alpha-01", runner=runner)
        plain_gemini = "AIzaSySharpeyePlainSecretBeta999"
        plain_grok = "xai-env-secret-gamma-777"
        plain_news = "news-api-secret-delta-555"
        plain_ollama_tok = "ollama-bearer-tok-epsilon-333"

        with tempfile.TemporaryDirectory(prefix="sharpeye-vault-status-") as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "# sharpeye local config",
                        "LLM_PROVIDER=gemini",
                        "NEWS_WINDOW=30d",
                        "OLLAMA_URL=http://127.0.0.1:11434",
                        "OLLAMA_MODEL=qwen2.5:7b",
                        f"SHARPEYE_API_KEY={sealed_sharpeye}",
                        f'GEMINI_API_KEY="{plain_gemini}"  # primary key',
                        "XAI_API_KEY=",
                        f"NEWS_API_KEY={plain_news}",
                        f"OLLAMA_TOKEN={plain_ollama_tok}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            env_map = {
                "GROK_API_KEY": plain_grok,
                "GITHUB_TOKEN": "ghp_unrelated_ci_token_must_not_appear",
            }

            stdout_json, stderr_json = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout_json), contextlib.redirect_stderr(stderr_json):
                rc_json = dpapi_vault_cli.main(
                    ["status", "--env-file", str(env_path), "--json"],
                    environ=env_map,
                )
            self.assertEqual(rc_json, 0)
            self.assertEqual(stderr_json.getvalue(), "")
            raw_json = stdout_json.getvalue()
            report = json.loads(raw_json)

            self.assertTrue(report["ok"])
            self.assertEqual(report["env_file"], ".env")
            self.assertTrue(report["env_file_exists"])
            self.assertTrue(report["has_plaintext"])
            self.assertEqual(
                report["keys"]["SHARPEYE_API_KEY"],
                {"env": "unset", "dotenv": "sealed", "effective": "sealed", "source": "dotenv"},
            )
            self.assertEqual(
                report["keys"]["GEMINI_API_KEY"],
                {"env": "unset", "dotenv": "plaintext", "effective": "plaintext", "source": "dotenv"},
            )
            self.assertEqual(
                report["keys"]["XAI_API_KEY"],
                {"env": "unset", "dotenv": "unset", "effective": "unset", "source": "none"},
            )
            self.assertEqual(
                report["keys"]["GROK_API_KEY"],
                {"env": "plaintext", "dotenv": "unset", "effective": "plaintext", "source": "env"},
            )
            self.assertEqual(report["keys"]["NEWS_API_KEY"]["dotenv"], "plaintext")
            self.assertEqual(report["keys"]["OLLAMA_TOKEN"]["dotenv"], "plaintext")

            # Non-secret config and unrelated host CI tokens must not be listed
            for excluded in ("LLM_PROVIDER", "NEWS_WINDOW", "OLLAMA_URL", "OLLAMA_MODEL", "GITHUB_TOKEN"):
                self.assertNotIn(excluded, report["keys"])

            stdout_txt, stderr_txt = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout_txt), contextlib.redirect_stderr(stderr_txt):
                rc_txt = dpapi_vault_cli.main(["status", "--env-file", str(env_path)], environ=env_map)
            self.assertEqual(rc_txt, 0)
            self.assertEqual(stderr_txt.getvalue(), "")
            raw_txt = stdout_txt.getvalue()
            self.assertIn("SHARPEYE_API_KEY: effective=sealed", raw_txt)
            self.assertIn("GEMINI_API_KEY: effective=plaintext", raw_txt)
            self.assertIn("NEWS_API_KEY: effective=plaintext", raw_txt)
            self.assertIn("OLLAMA_TOKEN: effective=plaintext", raw_txt)

            # Strict zero-leakage assertions: no secrets, no dpapi:v1:, no local user/temp path
            for forbidden in (
                "sharpeye-top-secret-alpha-01",
                plain_gemini,
                plain_grok,
                plain_news,
                plain_ollama_tok,
                sealed_sharpeye,
                dpapi_vault.DPAPI_PREFIX,
                tmp_dir,
                str(Path.home()),
                "ghp_unrelated_ci_token_must_not_appear",
            ):
                self.assertNotIn(forbidden, raw_json)
                self.assertNotIn(forbidden, raw_txt)

            # Duplicate key in .env with earlier plaintext line must still report plaintext
            dup_path = Path(tmp_dir) / "dup.env"
            dup_path.write_text(
                f"SHARPEYE_API_KEY=plain-leaked-earlier\nSHARPEYE_API_KEY={sealed_sharpeye}\n",
                encoding="utf-8",
            )
            dup_report = dpapi_vault_cli.collect_vault_status(dup_path, environ={})
            self.assertEqual(dup_report["keys"]["SHARPEYE_API_KEY"]["dotenv"], "plaintext")
            self.assertTrue(dup_report["has_plaintext"])

            # Missing .env file succeeds cleanly on status without leaking paths
            missing_env = Path(tmp_dir) / "does-not-exist.env"
            stdout_missing = io.StringIO()
            with contextlib.redirect_stdout(stdout_missing):
                rc_missing = dpapi_vault_cli.main(
                    ["status", "--env-file", str(missing_env), "--json"],
                    environ={"SHARPEYE_API_KEY": sealed_sharpeye},
                )
            self.assertEqual(rc_missing, 0)
            missing_out = stdout_missing.getvalue()
            self.assertNotIn(tmp_dir, missing_out)
            missing_report = json.loads(missing_out)
            self.assertFalse(missing_report["env_file_exists"])
            self.assertEqual(missing_report["env_file"], "does-not-exist.env")
            self.assertFalse(missing_report["has_plaintext"])
            self.assertEqual(missing_report["keys"]["SHARPEYE_API_KEY"]["effective"], "sealed")

    def test_seal_dry_run_previews_without_modifying_file_or_calling_dpapi(self) -> None:
        runner, calls = _make_fake_dpapi_runner()
        already_sealed = dpapi_vault.seal_secret("pre-sealed-sharpeye-val", runner=runner)
        calls.clear()

        plain_sharpeye = "sharpeye-dry-run-secret-111"
        plain_ollama = "ollama-dry-run-token-222"

        with tempfile.TemporaryDirectory(prefix="sharpeye-vault-dryrun-") as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            original_bytes = (
                f"LLM_PROVIDER=gemini\n"
                f"OLLAMA_URL=http://127.0.0.1:11434\n"
                f"SHARPEYE_API_KEY={plain_sharpeye}\n"
                f"GEMINI_API_KEY={already_sealed}\n"
                f"XAI_API_KEY=\n"
                f"OLLAMA_TOKEN={plain_ollama}\n"
            ).encode("utf-8")
            env_path.write_bytes(original_bytes)
            os.chmod(env_path, 0o644)

            stdout_json, stderr_json = io.StringIO(), io.StringIO()
            with mock.patch.object(dpapi_vault_cli, "is_dpapi_available", return_value=False), mock.patch.object(
                dpapi_vault, "seal_secret", side_effect=AssertionError("dry-run must not call seal_secret")
            ), mock.patch.object(
                dpapi_vault_cli.os, "chmod", side_effect=AssertionError("dry-run must not call chmod")
            ):
                with contextlib.redirect_stdout(stdout_json), contextlib.redirect_stderr(stderr_json):
                    rc = dpapi_vault_cli.main(["seal", "--env-file", str(env_path), "--dry-run", "--json"])

            self.assertEqual(rc, 0)
            self.assertEqual(stderr_json.getvalue(), "")
            raw_json = stdout_json.getvalue()
            res = json.loads(raw_json)
            self.assertTrue(res["ok"])
            self.assertTrue(res["dry_run"])
            self.assertFalse(res["changed"])
            self.assertEqual(res["env_file"], ".env")
            self.assertEqual(res["sealed_keys"], [])
            self.assertEqual(res["would_seal_keys"], ["SHARPEYE_API_KEY", "OLLAMA_TOKEN"])
            self.assertEqual(res["already_sealed_keys"], ["GEMINI_API_KEY"])
            self.assertIn("XAI_API_KEY", res["unset_keys"])
            self.assertIsNone(res["permissions"])

            # Text dry-run check
            stdout_txt, stderr_txt = io.StringIO(), io.StringIO()
            with mock.patch.object(dpapi_vault_cli, "is_dpapi_available", return_value=False):
                with contextlib.redirect_stdout(stdout_txt), contextlib.redirect_stderr(stderr_txt):
                    rc_txt = dpapi_vault_cli.main(["seal", "--env-file", str(env_path), "--dry-run"])
            self.assertEqual(rc_txt, 0)
            self.assertEqual(stderr_txt.getvalue(), "")
            raw_txt = stdout_txt.getvalue()
            self.assertIn("dry-run", raw_txt)
            self.assertIn("would_seal: SHARPEYE_API_KEY, OLLAMA_TOKEN", raw_txt)

            # File is byte-for-byte intact, permissions unchanged, zero temp files, zero leakage
            self.assertEqual(env_path.read_bytes(), original_bytes)
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(env_path.stat().st_mode), 0o644)
            self.assertEqual([p.name for p in Path(tmp_dir).iterdir()], [".env"])
            for forbidden in (plain_sharpeye, plain_ollama, already_sealed, dpapi_vault.DPAPI_PREFIX, tmp_dir):
                self.assertNotIn(forbidden, raw_json)
                self.assertNotIn(forbidden, raw_txt)

    def test_seal_atomic_fsync_0600_roundtrip_and_idempotency(self) -> None:
        runner, calls = _make_fake_dpapi_runner()
        plain_sharpeye = "sharpeye-ğüşöçıİ-secret-001"
        plain_gemini = "gem-plain#hash-secret-002"
        plain_grok = "grok-plain-secret-003"
        plain_news = "news-api-key-004"
        plain_custom = "custom-vault-entry-005"
        pre_sealed_xai = dpapi_vault.seal_secret("pre-existing-xai-006", runner=runner)
        calls.clear()

        with tempfile.TemporaryDirectory(prefix="sharpeye-vault-seal-") as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            # Include UTF-8 BOM and CRLF line endings to verify exact preservation
            env_content = (
                "# Header comment preserved\r\n"
                "LLM_PROVIDER=gemini\r\n"
                "NEWS_WINDOW=30d\r\n"
                "OLLAMA_URL=http://127.0.0.1:11434\r\n"
                f"SHARPEYE_API_KEY={plain_sharpeye}\r\n"
                f'GEMINI_API_KEY="{plain_gemini}"  # prod key\r\n'
                f"XAI_API_KEY={pre_sealed_xai}\r\n"
                f"export GROK_API_KEY='{plain_grok}'\r\n"
                f"NEWS_API_KEY={plain_news}\r\n"
                f"CUSTOM_VAULT_VALUE={plain_custom}\r\n"
                "ANTHROPIC_API_KEY=\r\n"
            )
            env_path.write_bytes(env_content.encode("utf-8-sig"))
            os.chmod(env_path, 0o644)

            events: list[str] = []
            real_mkstemp = tempfile.mkstemp
            real_chmod = os.chmod
            real_fsync = os.fsync
            real_replace = os.replace

            def spy_mkstemp(*args, **kwargs):
                events.append("mkstemp")
                return real_mkstemp(*args, **kwargs)

            def spy_chmod(path, mode, *args, **kwargs):
                events.append(f"chmod:{oct(mode)}")
                return real_chmod(path, mode, *args, **kwargs)

            def spy_fsync(fd):
                events.append("fsync")
                return real_fsync(fd)

            def spy_replace(src, dst):
                events.append("replace")
                return real_replace(src, dst)

            stdout_seal, stderr_seal = io.StringIO(), io.StringIO()
            with mock.patch.object(dpapi_vault_cli.tempfile, "mkstemp", side_effect=spy_mkstemp), mock.patch.object(
                dpapi_vault_cli.os, "chmod", side_effect=spy_chmod
            ), mock.patch.object(dpapi_vault_cli.os, "fsync", side_effect=spy_fsync), mock.patch.object(
                dpapi_vault_cli.os, "replace", side_effect=spy_replace
            ):
                with contextlib.redirect_stdout(stdout_seal), contextlib.redirect_stderr(stderr_seal):
                    rc = dpapi_vault_cli.main(
                        [
                            "seal",
                            "--env-file",
                            str(env_path),
                            "--scope",
                            "LocalMachine",
                            "--key",
                            "CUSTOM_VAULT_VALUE",
                            "--json",
                        ],
                        runner=runner,
                    )

            self.assertEqual(rc, 0)
            self.assertEqual(stderr_seal.getvalue(), "")
            seal_out = stdout_seal.getvalue()
            result = json.loads(seal_out)

            self.assertTrue(result["ok"])
            self.assertFalse(result["dry_run"])
            self.assertTrue(result["changed"])
            self.assertEqual(result["env_file"], ".env")
            self.assertEqual(result["permissions"], "0600")
            self.assertEqual(
                result["sealed_keys"],
                ["SHARPEYE_API_KEY", "GEMINI_API_KEY", "GROK_API_KEY", "CUSTOM_VAULT_VALUE", "NEWS_API_KEY"],
            )
            self.assertEqual(result["already_sealed_keys"], ["XAI_API_KEY"])
            self.assertEqual(result["unset_keys"], ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"])

            # Verify atomic write sequence: mkstemp -> chmod(0o600) -> fsync -> replace -> chmod(0o600)
            self.assertEqual(events, ["mkstemp", "chmod:0o600", "fsync", "replace", "chmod:0o600"])
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(env_path.stat().st_mode), 0o600)
            self.assertEqual([p.name for p in Path(tmp_dir).iterdir()], [".env"])

            # Verify BOM & CRLF preserved, non-secrets intact, zero plaintext on disk, zero leaks in output
            raw_disk = env_path.read_bytes()
            self.assertTrue(raw_disk.startswith(b"\xef\xbb\xbf"))
            disk_text = raw_disk.decode("utf-8-sig")
            self.assertIn("\r\n", disk_text)
            self.assertIn("# Header comment preserved\r\n", disk_text)
            self.assertIn("LLM_PROVIDER=gemini\r\n", disk_text)
            self.assertIn("NEWS_WINDOW=30d\r\n", disk_text)
            self.assertIn("OLLAMA_URL=http://127.0.0.1:11434\r\n", disk_text)
            self.assertIn("  # prod key\r\n", disk_text)
            self.assertIn("export GROK_API_KEY=dpapi:v1:", disk_text)
            self.assertIn(f"XAI_API_KEY={pre_sealed_xai}\r\n", disk_text)

            for secret in (plain_sharpeye, plain_gemini, plain_grok, plain_news, plain_custom):
                self.assertNotIn(secret, disk_text)
                self.assertNotIn(secret, seal_out)
            self.assertNotIn(dpapi_vault.DPAPI_PREFIX, seal_out)
            self.assertNotIn(tmp_dir, seal_out)

            # Verify every runner call used shell=False and LocalMachine scope with no secret in cmd
            self.assertEqual(len(calls), 5)
            for call in calls:
                self.assertFalse(call["shell"])
                self.assertEqual(call["env"]["_DPAPI_SCOPE"], "LocalMachine")
                joined_cmd = " ".join(call["cmd"])
                for secret in (plain_sharpeye, plain_gemini, plain_grok, plain_news, plain_custom):
                    self.assertNotIn(secret, joined_cmd)

            # Verify full unseal round-trip compatibility with dpapi_vault.load_secret / unseal_secret
            parsed = dpapi_vault_cli.parse_env_file_values(env_path)
            self.assertEqual(
                dpapi_vault.load_secret("SHARPEYE_API_KEY", env=parsed, scope="LocalMachine", runner=runner),
                plain_sharpeye,
            )
            self.assertEqual(
                dpapi_vault.unseal_secret(parsed["GEMINI_API_KEY"], scope="LocalMachine", runner=runner),
                plain_gemini,
            )
            self.assertEqual(
                dpapi_vault.unseal_secret(parsed["GROK_API_KEY"], scope="LocalMachine", runner=runner),
                plain_grok,
            )
            self.assertEqual(
                dpapi_vault.unseal_secret(parsed["NEWS_API_KEY"], scope="LocalMachine", runner=runner),
                plain_news,
            )
            self.assertEqual(
                dpapi_vault.unseal_secret(parsed["CUSTOM_VAULT_VALUE"], scope="LocalMachine", runner=runner),
                plain_custom,
            )
            self.assertEqual(
                dpapi_vault.unseal_secret(parsed["XAI_API_KEY"], scope="CurrentUser", runner=runner),
                "pre-existing-xai-006",
            )

            # Second seal pass is strictly idempotent: zero runner calls, byte-for-byte identical
            calls.clear()
            stdout_idem, stderr_idem = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout_idem), contextlib.redirect_stderr(stderr_idem):
                rc_idem = dpapi_vault_cli.main(
                    ["seal", "--env-file", str(env_path), "--key", "CUSTOM_VAULT_VALUE", "--json"],
                    runner=runner,
                )
            self.assertEqual(rc_idem, 0)
            self.assertEqual(stderr_idem.getvalue(), "")
            self.assertEqual(len(calls), 0)
            idem = json.loads(stdout_idem.getvalue())
            self.assertFalse(idem["changed"])
            self.assertEqual(idem["sealed_keys"], [])
            self.assertEqual(
                idem["already_sealed_keys"],
                [
                    "SHARPEYE_API_KEY",
                    "GEMINI_API_KEY",
                    "XAI_API_KEY",
                    "GROK_API_KEY",
                    "CUSTOM_VAULT_VALUE",
                    "NEWS_API_KEY",
                ],
            )
            self.assertEqual(env_path.read_bytes(), raw_disk)

            # Status after sealing confirms zero plaintext remaining
            status_after = dpapi_vault_cli.collect_vault_status(env_path, environ={})
            self.assertFalse(status_after["has_plaintext"])
            self.assertEqual(status_after["keys"]["SHARPEYE_API_KEY"]["dotenv"], "sealed")
            self.assertEqual(status_after["keys"]["NEWS_API_KEY"]["dotenv"], "sealed")
            self.assertEqual(status_after["keys"]["CUSTOM_VAULT_VALUE"]["dotenv"], "sealed")

    def test_seal_fails_closed_preserving_original_bytes_without_leaks(self) -> None:
        plain_first = "sharpeye-first-secret-alpha-111"
        plain_second = "gemini-second-secret-beta-222"

        with tempfile.TemporaryDirectory(prefix="sharpeye-vault-fail-") as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            original_bytes = (
                f"SHARPEYE_API_KEY={plain_first}\n"
                f"GEMINI_API_KEY={plain_second}\n"
            ).encode("utf-8")
            env_path.write_bytes(original_bytes)
            os.chmod(env_path, 0o644)

            # 1. DPAPI unavailable and no runner -> fails closed, file bytes AND mode untouched
            out1, err1 = io.StringIO(), io.StringIO()
            with mock.patch.object(dpapi_vault_cli, "is_dpapi_available", return_value=False):
                with contextlib.redirect_stdout(out1), contextlib.redirect_stderr(err1):
                    rc1 = dpapi_vault_cli.main(["seal", "--env-file", str(env_path)])
            self.assertEqual(rc1, 1)
            self.assertEqual(out1.getvalue(), "")
            self.assertIn("unavailable", err1.getvalue().lower())
            self.assertEqual(env_path.read_bytes(), original_bytes)
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(env_path.stat().st_mode), 0o644)
            self.assertEqual([p.name for p in Path(tmp_dir).iterdir()], [".env"])
            for forbidden in (plain_first, plain_second, tmp_dir, "Traceback"):
                self.assertNotIn(forbidden, err1.getvalue())

            # 2. Mid-batch failure: 1st key seals, 2nd key crashes with secret + path -> original untouched
            good_runner, _ = _make_fake_dpapi_runner()
            attempt = {"n": 0}

            def flaky_runner(cmd, **kwargs):
                attempt["n"] += 1
                if attempt["n"] == 1:
                    return good_runner(cmd, **kwargs)
                raise RuntimeError(f"boom {plain_second} at /home/alice/secret/.env")

            out2, err2 = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out2), contextlib.redirect_stderr(err2):
                rc2 = dpapi_vault_cli.main(["seal", "--env-file", str(env_path)], runner=flaky_runner)
            self.assertEqual(rc2, 1)
            self.assertEqual(out2.getvalue(), "")
            self.assertIn("GEMINI_API_KEY", err2.getvalue())
            self.assertEqual(env_path.read_bytes(), original_bytes)
            self.assertEqual([p.name for p in Path(tmp_dir).iterdir()], [".env"])
            for forbidden in (plain_first, plain_second, "/home/alice", str(Path.home()), tmp_dir, "Traceback"):
                self.assertNotIn(forbidden, err2.getvalue())

            # 3. Runner returns non-zero with secret in stderr -> fail closed, zero leakage
            def failing_proc_runner(cmd, **kwargs):
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=f"leak {plain_first}")

            out3, err3 = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out3), contextlib.redirect_stderr(err3):
                rc3 = dpapi_vault_cli.main(["seal", "--env-file", str(env_path)], runner=failing_proc_runner)
            self.assertEqual(rc3, 1)
            self.assertEqual(env_path.read_bytes(), original_bytes)
            for forbidden in (plain_first, plain_second, tmp_dir, "Traceback"):
                self.assertNotIn(forbidden, err3.getvalue())

            # 4. Atomic os.replace failure cleans up temp file and preserves original bytes
            out4, err4 = io.StringIO(), io.StringIO()
            with mock.patch.object(
                dpapi_vault_cli.os,
                "replace",
                side_effect=OSError(f"replace failed for {tmp_dir}/.env"),
            ):
                with contextlib.redirect_stdout(out4), contextlib.redirect_stderr(err4):
                    rc4 = dpapi_vault_cli.main(["seal", "--env-file", str(env_path)], runner=good_runner)
            self.assertEqual(rc4, 1)
            self.assertEqual(out4.getvalue(), "")
            self.assertEqual(env_path.read_bytes(), original_bytes)
            self.assertEqual([p.name for p in Path(tmp_dir).iterdir()], [".env"])
            for forbidden in (plain_first, plain_second, tmp_dir, "Traceback"):
                self.assertNotIn(forbidden, err4.getvalue())

    def test_missing_directory_symlink_invalid_utf8_and_bad_key_errors(self) -> None:
        runner, _ = _make_fake_dpapi_runner()
        with tempfile.TemporaryDirectory(prefix="sharpeye-vault-err-") as tmp_dir:
            # Missing file on seal and seal --dry-run
            missing = Path(tmp_dir) / "missing.env"
            for extra_flag in ([], ["--dry-run"]):
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    rc = dpapi_vault_cli.main(["seal", "--env-file", str(missing), *extra_flag], runner=runner)
                self.assertEqual(rc, 1)
                self.assertEqual(out.getvalue(), "")
                self.assertIn("not found", err.getvalue().lower())
                self.assertNotIn(tmp_dir, err.getvalue())
                self.assertNotIn("Traceback", err.getvalue())

            # Directory passed to both status and seal
            for subcmd in ("status", "seal"):
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    rc = dpapi_vault_cli.main([subcmd, "--env-file", tmp_dir], runner=runner)
                self.assertEqual(rc, 1)
                self.assertEqual(out.getvalue(), "")
                self.assertIn("regular file", err.getvalue().lower())
                self.assertNotIn(tmp_dir, err.getvalue())
                self.assertNotIn("Traceback", err.getvalue())

            # Symlink refused on seal and target file untouched
            real_file = Path(tmp_dir) / "real.env"
            real_bytes = b"SHARPEYE_API_KEY=symlink-target-secret\n"
            real_file.write_bytes(real_bytes)
            link_path = Path(tmp_dir) / "link.env"
            try:
                link_path.symlink_to(real_file)
            except OSError:
                link_path = None
            if link_path is not None:
                out_sym, err_sym = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out_sym), contextlib.redirect_stderr(err_sym):
                    rc_sym = dpapi_vault_cli.main(["seal", "--env-file", str(link_path)], runner=runner)
                self.assertEqual(rc_sym, 1)
                self.assertIn("symlink", err_sym.getvalue().lower())
                self.assertEqual(real_file.read_bytes(), real_bytes)
                self.assertNotIn(tmp_dir, err_sym.getvalue())

            # Invalid UTF-8 file rejected cleanly on both status and seal
            bad_utf8 = Path(tmp_dir) / "binary.env"
            bad_bytes = b"\xff\xfe\x00invalid"
            bad_utf8.write_bytes(bad_bytes)
            for subcmd in ("status", "seal"):
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    rc = dpapi_vault_cli.main([subcmd, "--env-file", str(bad_utf8)], runner=runner)
                self.assertEqual(rc, 1)
                self.assertIn("utf-8", err.getvalue().lower())
                self.assertEqual(bad_utf8.read_bytes(), bad_bytes)
                self.assertNotIn(tmp_dir, err.getvalue())

            # Invalid --key name rejected cleanly without echoing untrusted value
            valid_env = Path(tmp_dir) / "ok.env"
            valid_env.write_text("SHARPEYE_API_KEY=\n", encoding="utf-8")
            bad_key = "BAD KEY=/home/alice/secret"
            out_k, err_k = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out_k), contextlib.redirect_stderr(err_k):
                rc_k = dpapi_vault_cli.main(
                    ["status", "--env-file", str(valid_env), "--key", bad_key],
                    environ={},
                )
            self.assertEqual(rc_k, 1)
            self.assertIn("invalid", err_k.getvalue().lower())
            self.assertNotIn("/home/alice", err_k.getvalue())

    def test_standalone_subprocess_cli_zero_deps_and_zero_leaks(self) -> None:
        repo_root = Path(__file__).resolve().parent
        cli_script = repo_root / "dpapi_vault_cli.py"
        plain_secret = "sharpeye-subproc-secret-999888"
        user_home = str(Path.home())

        with tempfile.TemporaryDirectory(prefix="sharpeye-vault-subproc-") as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            original_bytes = f"SHARPEYE_API_KEY={plain_secret}\nOLLAMA_URL=http://127.0.0.1:11434\n".encode("utf-8")
            env_path.write_bytes(original_bytes)

            child_env = {
                k: v
                for k, v in os.environ.items()
                if not dpapi_vault_cli.is_sensitive_key(k)
            }
            child_env["PYTHONDONTWRITEBYTECODE"] = "1"

            # 1. status --json
            proc_status = subprocess.run(
                [sys.executable, "-B", str(cli_script), "status", "--env-file", str(env_path), "--json"],
                capture_output=True,
                text=True,
                env=child_env,
                check=False,
                timeout=15,
            )
            self.assertEqual(proc_status.returncode, 0, proc_status.stderr)
            self.assertEqual(proc_status.stderr, "")
            status_payload = json.loads(proc_status.stdout)
            self.assertEqual(status_payload["env_file"], ".env")
            self.assertEqual(status_payload["keys"]["SHARPEYE_API_KEY"]["effective"], "plaintext")
            for forbidden in (plain_secret, tmp_dir, user_home, "Traceback"):
                self.assertNotIn(forbidden, proc_status.stdout)
                self.assertNotIn(forbidden, proc_status.stderr)

            # 2. seal --dry-run --json
            proc_dry = subprocess.run(
                [sys.executable, "-B", str(cli_script), "seal", "--env-file", str(env_path), "--dry-run", "--json"],
                capture_output=True,
                text=True,
                env=child_env,
                check=False,
                timeout=15,
            )
            self.assertEqual(proc_dry.returncode, 0, proc_dry.stderr)
            dry_payload = json.loads(proc_dry.stdout)
            self.assertTrue(dry_payload["dry_run"])
            self.assertEqual(dry_payload["would_seal_keys"], ["SHARPEYE_API_KEY"])
            self.assertEqual(env_path.read_bytes(), original_bytes)

            # 3. seal without DPAPI on non-Windows fails closed and preserves bytes
            if sys.platform != "win32":
                proc_seal = subprocess.run(
                    [sys.executable, "-B", str(cli_script), "seal", "--env-file", str(env_path)],
                    capture_output=True,
                    text=True,
                    env=child_env,
                    check=False,
                    timeout=15,
                )
                self.assertEqual(proc_seal.returncode, 1)
                self.assertEqual(proc_seal.stdout, "")
                self.assertEqual(env_path.read_bytes(), original_bytes)
                for forbidden in (plain_secret, tmp_dir, user_home, "Traceback"):
                    self.assertNotIn(forbidden, proc_seal.stderr)

            # 4. unknown subcommand exits 2 without leaking absolute script path or traceback
            proc_bad = subprocess.run(
                [sys.executable, "-B", str(cli_script), "unknown-cmd"],
                capture_output=True,
                text=True,
                env=child_env,
                check=False,
                timeout=15,
            )
            self.assertEqual(proc_bad.returncode, 2)
            self.assertNotIn(str(repo_root), proc_bad.stderr)
            self.assertNotIn(user_home, proc_bad.stderr)
            self.assertNotIn("Traceback", proc_bad.stderr)


if __name__ == "__main__":
    unittest.main()
