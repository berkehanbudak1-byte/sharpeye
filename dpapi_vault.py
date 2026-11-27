"""Windows DPAPI (`dpapi:v1:`) local secret vault helper.

Protects and unprotects local secrets on Windows using
`System.Security.Cryptography.ProtectedData` without ever interpolating secret
material into shell or PowerShell command strings. Secrets and optional entropy
are passed strictly via Base64-encoded environment variables to a static
PowerShell script with `shell=False`.
"""
from __future__ import annotations

import base64
import binascii
import os
import subprocess
import sys
from typing import Any, Callable, Mapping, Optional

DPAPI_PREFIX = "dpapi:v1:"
ALLOWED_SCOPES = frozenset({"CurrentUser", "LocalMachine"})

_ENV_INPUT_B64 = "_DPAPI_INPUT_B64"
_ENV_ENTROPY_B64 = "_DPAPI_ENTROPY_B64"
_ENV_SCOPE = "_DPAPI_SCOPE"

_PS_PROTECT_SCRIPT = (
    "Add-Type -AssemblyName System.Security; "
    "$ErrorActionPreference = 'Stop'; "
    "$inBytes = [Convert]::FromBase64String($env:_DPAPI_INPUT_B64); "
    "$entropyBytes = $null; "
    "if ($env:_DPAPI_ENTROPY_B64) { $entropyBytes = [Convert]::FromBase64String($env:_DPAPI_ENTROPY_B64) }; "
    "$scope = if ($env:_DPAPI_SCOPE -eq 'LocalMachine') { "
    "[System.Security.Cryptography.DataProtectionScope]::LocalMachine } else { "
    "[System.Security.Cryptography.DataProtectionScope]::CurrentUser }; "
    "$outBytes = [System.Security.Cryptography.ProtectedData]::Protect($inBytes, $entropyBytes, $scope); "
    "[Console]::Out.Write([Convert]::ToBase64String($outBytes))"
)

_PS_UNPROTECT_SCRIPT = (
    "Add-Type -AssemblyName System.Security; "
    "$ErrorActionPreference = 'Stop'; "
    "$inBytes = [Convert]::FromBase64String($env:_DPAPI_INPUT_B64); "
    "$entropyBytes = $null; "
    "if ($env:_DPAPI_ENTROPY_B64) { $entropyBytes = [Convert]::FromBase64String($env:_DPAPI_ENTROPY_B64) }; "
    "$scope = if ($env:_DPAPI_SCOPE -eq 'LocalMachine') { "
    "[System.Security.Cryptography.DataProtectionScope]::LocalMachine } else { "
    "[System.Security.Cryptography.DataProtectionScope]::CurrentUser }; "
    "$outBytes = [System.Security.Cryptography.ProtectedData]::Unprotect($inBytes, $entropyBytes, $scope); "
    "[Console]::Out.Write([Convert]::ToBase64String($outBytes))"
)


class DPAPIError(RuntimeError):
    """Raised when a Windows DPAPI operation fails or is unavailable."""


def is_dpapi_secret(value: Optional[str]) -> bool:
    """Return True when value carries the `dpapi:v1:` sealed secret prefix."""
    return isinstance(value, str) and value.strip().startswith(DPAPI_PREFIX)


def _validate_scope(scope: str) -> str:
    if not isinstance(scope, str) or scope not in ALLOWED_SCOPES:
        raise ValueError(f"Unsupported DPAPI scope; expected one of {sorted(ALLOWED_SCOPES)}")
    return scope


def _run_dpapi_script(
    script: str,
    input_b64: str,
    *,
    scope: str,
    entropy: Optional[str],
    timeout: float,
    runner: Optional[Callable[..., Any]],
) -> str:
    if runner is None and sys.platform != "win32":
        raise DPAPIError("Windows DPAPI is only available on Windows (win32).")
    if entropy is not None and not isinstance(entropy, str):
        raise ValueError("entropy must be a string or None")

    child_env = dict(os.environ)
    child_env[_ENV_INPUT_B64] = input_b64
    child_env[_ENV_SCOPE] = scope
    if entropy:
        child_env[_ENV_ENTROPY_B64] = base64.b64encode(entropy.encode("utf-8")).decode("ascii")
    else:
        child_env.pop(_ENV_ENTROPY_B64, None)

    cmd = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script]
    exec_fn = runner if runner is not None else subprocess.run
    try:
        proc = exec_fn(
            cmd,
            capture_output=True,
            text=True,
            env=child_env,
            timeout=timeout,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DPAPIError(f"DPAPI runner failed ({type(exc).__name__}).") from None

    returncode = getattr(proc, "returncode", None)
    if returncode != 0:
        raise DPAPIError(f"DPAPI operation failed with exit code {returncode}.")

    stdout = str(getattr(proc, "stdout", "") or "").strip()
    if not stdout:
        raise DPAPIError("DPAPI operation returned empty output.")
    return stdout


def seal_secret(
    plaintext: str,
    *,
    scope: str = "CurrentUser",
    entropy: Optional[str] = None,
    timeout: float = 15.0,
    runner: Optional[Callable[..., Any]] = None,
) -> str:
    """Seal a UTF-8 secret with Windows DPAPI and return a `dpapi:v1:<b64>` string."""
    if not isinstance(plaintext, str) or not plaintext:
        raise ValueError("plaintext secret must be a non-empty string")
    valid_scope = _validate_scope(scope)
    input_b64 = base64.b64encode(plaintext.encode("utf-8")).decode("ascii")
    out_b64 = _run_dpapi_script(
        _PS_PROTECT_SCRIPT,
        input_b64,
        scope=valid_scope,
        entropy=entropy,
        timeout=timeout,
        runner=runner,
    )
    try:
        raw_protected = base64.b64decode(out_b64, validate=True)
    except (binascii.Error, ValueError):
        raise DPAPIError("DPAPI protect returned invalid Base64 output.") from None
    if not raw_protected:
        raise DPAPIError("DPAPI protect returned empty ciphertext.")
    canonical = base64.b64encode(raw_protected).decode("ascii")
    return f"{DPAPI_PREFIX}{canonical}"


def unseal_secret(
    sealed_value: str,
    *,
    scope: str = "CurrentUser",
    entropy: Optional[str] = None,
    timeout: float = 15.0,
    runner: Optional[Callable[..., Any]] = None,
) -> str:
    """Unseal a `dpapi:v1:<b64>` token back to its original UTF-8 plaintext secret."""
    if not isinstance(sealed_value, str):
        raise ValueError("sealed_value must be a string")
    token = sealed_value.strip()
    if not token.startswith(DPAPI_PREFIX):
        raise ValueError(f"sealed_value must start with '{DPAPI_PREFIX}'")
    payload_b64 = token[len(DPAPI_PREFIX):].strip()
    if not payload_b64:
        raise ValueError("sealed_value payload must not be empty")
    try:
        raw_cipher = base64.b64decode(payload_b64, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("sealed_value contains invalid Base64 payload") from None
    if not raw_cipher:
        raise ValueError("sealed_value payload must not be empty")

    valid_scope = _validate_scope(scope)
    canonical_in = base64.b64encode(raw_cipher).decode("ascii")
    out_b64 = _run_dpapi_script(
        _PS_UNPROTECT_SCRIPT,
        canonical_in,
        scope=valid_scope,
        entropy=entropy,
        timeout=timeout,
        runner=runner,
    )
    try:
        raw_plain = base64.b64decode(out_b64, validate=True)
        return raw_plain.decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        raise DPAPIError("DPAPI unprotect returned invalid UTF-8 payload.") from None


def load_secret(
    env_name: str,
    *,
    default: str = "",
    env: Optional[Mapping[str, str]] = None,
    scope: str = "CurrentUser",
    entropy: Optional[str] = None,
    timeout: float = 15.0,
    runner: Optional[Callable[..., Any]] = None,
) -> str:
    """Load an environment secret, transparently unsealing `dpapi:v1:` values."""
    source = env if env is not None else os.environ
    raw = (source.get(env_name) or "").strip()
    if not raw:
        return default
    if raw.startswith(DPAPI_PREFIX):
        return unseal_secret(
            raw,
            scope=scope,
            entropy=entropy,
            timeout=timeout,
            runner=runner,
        )
    return raw
