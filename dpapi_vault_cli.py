#!/usr/bin/env python3
"""Zero-dependency Windows DPAPI vault CLI for sharpeye secrets.

Inspects whether sensitive environment and `.env` keys are sealed (`dpapi:v1:`),
plaintext, or unset, and atomically seals plaintext keys in-place inside `.env`
using Windows DPAPI (`dpapi_vault.py`) with `0o600` file permissions. Never
prints or logs secret values, DPAPI ciphertexts, local user paths, or raw
tracebacks.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from collections.abc import Iterable, Mapping, Sequence
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Callable, NoReturn

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import dpapi_vault  # noqa: E402

DPAPI_PREFIX = dpapi_vault.DPAPI_PREFIX

# Aligned with `app._DPAPI_SECRET_ENV_NAMES` and `core._SECRET_ENV_NAMES`.
SECRET_ENV_KEYS: tuple[str, ...] = (
    "SHARPEYE_API_KEY",
    "GEMINI_API_KEY",
    "XAI_API_KEY",
    "GROK_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
)

_REPO_ENV_PREFIXES: tuple[str, ...] = (
    "SHARPEYE_",
    "NEWS_",
    "OLLAMA_",
)

_ENV_KEY_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SENSITIVE_KEY_RE = re.compile(
    r"(?:^|_)(?:"
    r"API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|PASSPHRASE|"
    r"PRIVATE_KEY|ACCESS_KEY|AUTH_KEY|SIGNING_KEY|ENCRYPTION_KEY|MASTER_KEY|"
    r"CREDENTIAL|CREDENTIALS|BEARER"
    r")$",
    re.IGNORECASE,
)
_ASSIGN_RE = re.compile(r"^(\s*(?:export\s+)?)([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)(.*)$")
_INLINE_COMMENT_RE = re.compile(r"(\s+#.*)$")
_WIN_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\)[^\s'\"`,;()]+")
_POSIX_PATH_RE = re.compile(r"(?:(?<=^)|(?<=[\s'\"=(]))/(?:[^\s'\"`,;()]+)(?:/[^\s'\"`,;()]*)*")


class VaultCLIError(RuntimeError):
    """Controlled, leak-free error raised by the DPAPI vault CLI."""


def is_dpapi_available() -> bool:
    """Return True when Windows DPAPI sealing via powershell.exe is available."""
    return dpapi_vault.sys.platform == "win32" and bool(shutil.which("powershell.exe"))


def _safe_env_display_name(env_file: str | Path) -> str:
    """Return a basename-only display label so local user paths never leak."""
    raw = str(env_file or "").strip()
    if not raw:
        return ".env"
    segments = [part for part in re.split(r"[/\\]+", raw) if part]
    if not segments:
        return ".env"
    candidate = segments[-1]
    if candidate in (".", "..") or ":" in candidate or any(ord(ch) < 32 for ch in candidate):
        return ".env"
    return candidate


def _redact_sensitive_text(text: str) -> str:
    """Redact API keys, Bearer tokens, DPAPI tokens, and URL credentials from errors."""
    s = str(text or "")
    s = re.sub(r"(https?://)([^/\s:@]+:[^/\s@]+@)", r"\1***@", s, flags=re.IGNORECASE)
    s = re.sub(
        r"([?&](?:api_key|apikey|token|access_token|secret|password|key)=)[^&\s]+",
        r"\1***",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"\b(Bearer\s+)[A-Za-z0-9._~+/=-]{6,}", r"\1[REDACTED]", s, flags=re.IGNORECASE)
    s = re.sub(r"\bdpapi:v1:[A-Za-z0-9+/=]+", "dpapi:v1:[REDACTED]", s)
    s = re.sub(r"\b(?:sk|xai|gsk|AIza)[-_A-Za-z0-9]{8,}\b", "[REDACTED_KEY]", s)
    return s


def _sanitize_output_text(text: str) -> str:
    """Redact secrets and local filesystem paths from diagnostic messages."""
    cleaned = _redact_sensitive_text(str(text or ""))
    cleaned = _WIN_PATH_RE.sub("[PATH]", cleaned)
    cleaned = _POSIX_PATH_RE.sub("[PATH]", cleaned)
    return cleaned


def _validate_key_names(keys: Iterable[str]) -> tuple[str, ...]:
    validated: list[str] = []
    seen: set[str] = set()
    for raw_key in keys:
        key = str(raw_key or "").strip()
        if not key or not _ENV_KEY_NAME_RE.fullmatch(key):
            raise VaultCLIError("Invalid environment variable name.")
        if key not in seen:
            seen.add(key)
            validated.append(key)
    return tuple(validated)


def is_sensitive_key(key: str, *, extra_keys: Iterable[str] = ()) -> bool:
    """Return True if an environment key holds a known or pattern-matched secret."""
    cleaned = str(key or "").strip()
    if not cleaned or not _ENV_KEY_NAME_RE.fullmatch(cleaned):
        return False
    upper = cleaned.upper()
    if upper in SECRET_ENV_KEYS:
        return True
    if any(cleaned == extra or upper == str(extra).strip().upper() for extra in extra_keys):
        return True
    return bool(_SENSITIVE_KEY_RE.search(cleaned))


def _split_line_ending(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    if line.endswith("\r"):
        return line[:-1], "\r"
    return line, ""


def _parse_rhs_value(rhs: str) -> tuple[str, str]:
    """Parse a `.env` value expression into `(value, trailing_suffix)`."""
    stripped = rhs.strip()
    if not stripped:
        return "", ""
    if stripped[0] in ('"', "'"):
        quote = stripped[0]
        chars: list[str] = []
        idx = 1
        escaped = False
        closed = False
        while idx < len(stripped):
            ch = stripped[idx]
            if quote == '"' and escaped:
                chars.append(ch)
                escaped = False
                idx += 1
                continue
            if quote == '"' and ch == "\\":
                escaped = True
                idx += 1
                continue
            if ch == quote:
                closed = True
                idx += 1
                break
            chars.append(ch)
            idx += 1
        if closed:
            return "".join(chars), stripped[idx:]
        return stripped, ""
    if stripped.startswith("#"):
        return "", rhs
    match = _INLINE_COMMENT_RE.search(rhs)
    if match:
        return rhs[: match.start()].strip(), match.group(1)
    return stripped, ""


def classify_secret_value(value: str | None) -> str:
    """Classify a secret value as `'sealed'`, `'plaintext'`, or `'unset'`."""
    if value is None:
        return "unset"
    raw = str(value).strip()
    if not raw:
        return "unset"
    if raw.startswith(DPAPI_PREFIX):
        return "sealed" if raw[len(DPAPI_PREFIX) :].strip() else "unset"
    return "plaintext"


def _read_env_bytes(target: Path) -> bytes:
    if target.exists() and not target.is_file():
        raise VaultCLIError("Target .env path must be a regular file.")
    return target.read_bytes()


def parse_env_file_entries(env_file: str | Path) -> list[tuple[str, str]]:
    """Read ordered `(key, value)` assignments from a `.env` file using stdlib only."""
    target = Path(env_file)
    if not target.exists():
        return []
    raw_bytes = _read_env_bytes(target)
    try:
        raw_text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise VaultCLIError("Target .env file is not valid UTF-8.") from None
    entries: list[tuple[str, str]] = []
    for line in raw_text.splitlines():
        body, _ = _split_line_ending(line)
        match = _ASSIGN_RE.match(body)
        if not match:
            continue
        _, key, _, rhs = match.groups()
        val, _ = _parse_rhs_value(rhs)
        entries.append((key, val))
    return entries


def parse_env_file_values(env_file: str | Path) -> dict[str, str]:
    """Read key-value assignments from a `.env` file (last assignment wins)."""
    values: dict[str, str] = {}
    for key, val in parse_env_file_entries(env_file):
        values[key] = val
    return values


def _aggregate_dotenv_states(entries: Sequence[tuple[str, str]]) -> dict[str, str]:
    """Summarize `.env` state per key; any plaintext occurrence wins over sealed."""
    states: dict[str, str] = {}
    for key, val in entries:
        state = classify_secret_value(val)
        prev = states.get(key)
        if prev == "plaintext" or state == "plaintext":
            states[key] = "plaintext"
        elif state == "sealed" or prev == "sealed":
            states[key] = "sealed"
        else:
            states[key] = "unset"
    return states


def _discover_status_keys(
    entries: Sequence[tuple[str, str]],
    env_source: Mapping[str, str],
    *,
    explicit_keys: Sequence[str] | None,
    extra_keys: Sequence[str],
) -> list[str]:
    extra_valid = _validate_key_names(extra_keys)
    if explicit_keys is not None:
        return list(_validate_key_names((*explicit_keys, *extra_valid)))

    ordered: list[str] = []
    seen: set[str] = set()

    def _add(k: str) -> None:
        if k not in seen:
            seen.add(k)
            ordered.append(k)

    for k in SECRET_ENV_KEYS:
        _add(k)
    for k in extra_valid:
        _add(k)
    for key, val in entries:
        if is_sensitive_key(key, extra_keys=extra_valid) or classify_secret_value(val) == "sealed":
            _add(key)
    for key in sorted(env_source.keys()):
        if not _ENV_KEY_NAME_RE.fullmatch(str(key)):
            continue
        upper = key.upper()
        if any(upper.startswith(pref) for pref in _REPO_ENV_PREFIXES) and is_sensitive_key(
            key, extra_keys=extra_valid
        ):
            _add(key)
    return ordered


def collect_vault_status(
    env_file: str | Path = ".env",
    *,
    keys: Sequence[str] | None = None,
    extra_keys: Sequence[str] = (),
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Inspect secret key protection state across process env and `.env`."""
    target = Path(env_file)
    if target.exists() and not target.is_file():
        raise VaultCLIError("Target .env path must be a regular file.")

    env_source = os.environ if environ is None else environ
    env_file_exists = target.is_file()
    entries = parse_env_file_entries(target) if env_file_exists else []
    dotenv_states = _aggregate_dotenv_states(entries)
    inspected_keys = _discover_status_keys(
        entries,
        env_source,
        explicit_keys=keys,
        extra_keys=extra_keys,
    )

    key_statuses: dict[str, dict[str, str]] = {}
    has_plaintext = False
    counts = {"sealed": 0, "plaintext": 0, "unset": 0}
    for key in inspected_keys:
        env_state = classify_secret_value(env_source.get(key))
        dotenv_state = dotenv_states.get(key, "unset")
        if env_state != "unset":
            effective = env_state
            source = "env"
        elif dotenv_state != "unset":
            effective = dotenv_state
            source = "dotenv"
        else:
            effective = "unset"
            source = "none"

        if env_state == "plaintext" or dotenv_state == "plaintext":
            has_plaintext = True

        counts[effective] += 1
        key_statuses[key] = {
            "env": env_state,
            "dotenv": dotenv_state,
            "effective": effective,
            "source": source,
        }

    return {
        "ok": True,
        "dpapi_available": bool(is_dpapi_available()),
        "env_file": _safe_env_display_name(target),
        "env_file_exists": env_file_exists,
        "has_plaintext": has_plaintext,
        "counts": counts,
        "keys": key_statuses,
    }


def _validate_sealed_output(sealed_val: Any, plaintext: str, key: str) -> str:
    if (
        not isinstance(sealed_val, str)
        or not sealed_val.startswith(DPAPI_PREFIX)
        or sealed_val == plaintext
    ):
        raise VaultCLIError(f"DPAPI sealing produced an invalid sealed value for {key}.")
    payload = sealed_val[len(DPAPI_PREFIX) :].strip()
    if not payload or (len(plaintext) >= 4 and plaintext in payload):
        raise VaultCLIError(f"DPAPI sealing produced an invalid sealed value for {key}.")
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise VaultCLIError(f"DPAPI sealing produced an invalid Base64 payload for {key}.") from None
    if not decoded:
        raise VaultCLIError(f"DPAPI sealing produced an empty payload for {key}.")
    return f"{DPAPI_PREFIX}{payload}"


def seal_env_file(
    env_file: str | Path,
    *,
    keys: Sequence[str] | None = None,
    extra_keys: Sequence[str] = (),
    scope: str = "CurrentUser",
    entropy: str | None = None,
    dry_run: bool = False,
    runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Seal plaintext sensitive keys in `.env` atomically with `0o600` permissions."""
    if scope not in dpapi_vault.ALLOWED_SCOPES:
        raise VaultCLIError(
            f"Unsupported DPAPI scope; expected one of {', '.join(sorted(dpapi_vault.ALLOWED_SCOPES))}."
        )

    extra_valid = _validate_key_names(extra_keys)
    explicit_valid = _validate_key_names((*keys, *extra_valid)) if keys is not None else None
    explicit_set = set(explicit_valid) if explicit_valid is not None else None

    target = Path(env_file)
    if target.is_symlink():
        raise VaultCLIError("Refusing to seal symlinked .env path.")
    if not target.exists():
        raise FileNotFoundError("Target .env file does not exist")
    if not target.is_file():
        raise VaultCLIError("Target .env path must be a regular file.")

    # Fail closed immediately before any modification when DPAPI is unavailable.
    if not dry_run and runner is None and not is_dpapi_available():
        raise VaultCLIError("Windows DPAPI is unavailable; cannot seal .env file.")

    raw_bytes = target.read_bytes()
    has_bom = raw_bytes.startswith(b"\xef\xbb\xbf")
    try:
        raw_text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise VaultCLIError("Target .env file is not valid UTF-8.") from None

    ordered_keys: list[str] = []
    seen_keys: set[str] = set()

    def _track_key(k: str) -> None:
        if k not in seen_keys:
            seen_keys.add(k)
            ordered_keys.append(k)

    base_keys = explicit_valid if explicit_valid is not None else (*SECRET_ENV_KEYS, *extra_valid)
    for k in base_keys:
        _track_key(k)

    sealed_set: set[str] = set()
    would_seal_set: set[str] = set()
    already_sealed_set: set[str] = set()
    new_lines: list[str] = []
    changed = False

    for line in raw_text.splitlines(keepends=True):
        body, eol = _split_line_ending(line)
        match = _ASSIGN_RE.match(body)
        if not match:
            new_lines.append(line)
            continue

        prefix, key, eq_part, rhs = match.groups()
        val, suffix = _parse_rhs_value(rhs)
        state = classify_secret_value(val)

        if explicit_set is not None:
            managed = key in explicit_set
        else:
            managed = is_sensitive_key(key, extra_keys=extra_valid) or state == "sealed"

        if not managed:
            new_lines.append(line)
            continue

        _track_key(key)
        if state == "unset":
            new_lines.append(line)
            continue
        if state == "sealed":
            already_sealed_set.add(key)
            new_lines.append(line)
            continue

        if dry_run:
            would_seal_set.add(key)
            new_lines.append(line)
            continue

        try:
            sealed_val = dpapi_vault.seal_secret(
                val,
                scope=scope,
                entropy=entropy,
                runner=runner,
            )
        except Exception as exc:
            raise VaultCLIError(f"DPAPI sealing failed for {key} ({type(exc).__name__}).") from None

        validated_sealed = _validate_sealed_output(sealed_val, val, key)
        new_lines.append(f"{prefix}{key}{eq_part}{validated_sealed}{suffix}{eol}")
        sealed_set.add(key)
        changed = True

    if not dry_run:
        if changed:
            output_bytes = "".join(new_lines).encode("utf-8-sig" if has_bom else "utf-8")
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=str(target.parent),
            )
            tmp_path = Path(tmp_name)
            fd_closed = False
            try:
                os.chmod(tmp_path, 0o600)
                with os.fdopen(fd, "wb") as handle:
                    fd_closed = True
                    handle.write(output_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_path, target)
                os.chmod(target, 0o600)
            except Exception:
                if not fd_closed:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
        else:
            os.chmod(target, 0o600)

    sealed_keys = [k for k in ordered_keys if k in sealed_set]
    would_seal_keys = [k for k in ordered_keys if k in would_seal_set]
    already_sealed_keys = [
        k
        for k in ordered_keys
        if k in already_sealed_set and k not in sealed_set and k not in would_seal_set
    ]
    unset_keys = [
        k
        for k in ordered_keys
        if k not in sealed_set and k not in would_seal_set and k not in already_sealed_set
    ]

    return {
        "ok": True,
        "dry_run": bool(dry_run),
        "dpapi_available": bool(is_dpapi_available()),
        "env_file": _safe_env_display_name(target),
        "changed": False if dry_run else changed,
        "would_seal_keys": would_seal_keys,
        "sealed_keys": sealed_keys,
        "already_sealed_keys": already_sealed_keys,
        "unset_keys": unset_keys,
        "permissions": None if dry_run else "0600",
    }


def _format_status_text(report: dict[str, Any]) -> str:
    lines = [
        f"Windows DPAPI available: {report['dpapi_available']}",
        f"Env file: {report['env_file']} ({'present' if report['env_file_exists'] else 'absent'})",
    ]
    for key, info in report["keys"].items():
        lines.append(
            f"  {key}: effective={info['effective']} (source={info['source']}, "
            f"env={info['env']}, dotenv={info['dotenv']})"
        )
    return "\n".join(lines)


def _format_seal_text(result: dict[str, Any]) -> str:
    already = ", ".join(result["already_sealed_keys"]) if result["already_sealed_keys"] else "none"
    unset = ", ".join(result["unset_keys"]) if result["unset_keys"] else "none"
    if result["dry_run"]:
        would_seal = ", ".join(result["would_seal_keys"]) if result["would_seal_keys"] else "none"
        return (
            f"DPAPI seal dry-run ({result['env_file']}, changed=False, "
            f"dpapi_available={result['dpapi_available']})\n"
            f"  would_seal: {would_seal}\n"
            f"  already_sealed: {already}\n"
            f"  unset: {unset}"
        )
    sealed = ", ".join(result["sealed_keys"]) if result["sealed_keys"] else "none"
    return (
        f"DPAPI seal complete ({result['env_file']}, permissions={result['permissions']}, "
        f"changed={result['changed']})\n"
        f"  sealed: {sealed}\n"
        f"  already_sealed: {already}\n"
        f"  unset: {unset}"
    )


class _SafeArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that sanitizes error output to prevent path or secret leakage."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        safe_msg = _sanitize_output_text(message)
        self.exit(2, f"{self.prog}: error: {safe_msg}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        prog="dpapi_vault_cli.py",
        description="Inspect or seal sharpeye secrets with Windows DPAPI.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser(
        "status",
        help="Report whether sensitive keys are sealed, plaintext, or unset",
    )
    p_status.add_argument("--env-file", default=".env", help="Path to .env file (default: .env)")
    p_status.add_argument(
        "--key",
        action="append",
        default=[],
        dest="extra_keys",
        help="Additional sensitive key name to inspect (repeatable)",
    )
    p_status.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON summary")

    p_seal = sub.add_parser(
        "seal",
        help="Seal plaintext sensitive keys in-place inside a .env file",
    )
    p_seal.add_argument("--env-file", default=".env", help="Path to .env file (default: .env)")
    p_seal.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which keys would be sealed without modifying the .env file",
    )
    p_seal.add_argument(
        "--scope",
        default="CurrentUser",
        choices=sorted(dpapi_vault.ALLOWED_SCOPES),
        help="Windows DPAPI protection scope (default: CurrentUser)",
    )
    p_seal.add_argument(
        "--key",
        action="append",
        default=[],
        dest="extra_keys",
        help="Additional sensitive key name to seal (repeatable)",
    )
    p_seal.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON summary")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable[..., Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else 1

    try:
        if args.cmd == "status":
            report = collect_vault_status(
                args.env_file,
                extra_keys=args.extra_keys,
                environ=environ,
            )
            if args.as_json:
                print(json.dumps(report, indent=2, ensure_ascii=False))
            else:
                print(_format_status_text(report))
            return 0

        if args.cmd == "seal":
            result = seal_env_file(
                args.env_file,
                extra_keys=args.extra_keys,
                scope=args.scope,
                dry_run=args.dry_run,
                runner=runner,
            )
            if args.as_json:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print(_format_seal_text(result))
            return 0
    except FileNotFoundError:
        print("Error: target .env file not found.", file=sys.stderr)
        return 1
    except PermissionError:
        print("Error: permission denied accessing target .env file.", file=sys.stderr)
        return 1
    except IsADirectoryError:
        print("Error: target .env path must be a regular file.", file=sys.stderr)
        return 1
    except VaultCLIError as exc:
        print(f"Error: {_sanitize_output_text(str(exc))}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Error: failed to access .env file ({type(exc).__name__}).", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: unexpected vault CLI failure ({type(exc).__name__}).", file=sys.stderr)
        return 1

    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Error: interrupted.", file=sys.stderr)
        raise SystemExit(130)
