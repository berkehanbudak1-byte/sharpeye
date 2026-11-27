"""Çalıştır: uvicorn app:app   (önbellek ve istek sınırı süreç içi olduğundan tek worker kullan)"""
import asyncio
import base64
from collections import OrderedDict, deque
import hashlib
import hmac
import math
import os
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

import core
import dpapi_vault
import swarm_orchestrator

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=500)
_INDEX_PATH = Path(__file__).parent / "index.html"


def _script_hashes(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return [
        "'sha256-" + base64.b64encode(hashlib.sha256(m.encode("utf-8")).digest()).decode() + "'"
        for m in re.findall(r"<script>(.*?)</script>", text, re.S)
    ]


_CSP = "; ".join([
    "default-src 'self'",
    "script-src " + " ".join(["'self'", *_script_hashes(_INDEX_PATH)]),
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
    "font-src 'self' https://fonts.gstatic.com data:",
    "img-src 'self' data: blob: https:",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])

SECURITY_HEADERS = {
    "Strict-Transport-Security": "max-age=31536000",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    "Content-Security-Policy": _CSP,
}


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    for k, v in SECURITY_HEADERS.items():
        response.headers[k] = v
    return response


class AuthFailureRateLimiter:
    """Bounded sliding-window per-IP rate limiter for failed API-key attempts."""

    def __init__(
        self,
        max_failures: int = 5,
        window_seconds: float = 60.0,
        max_tracked_ips: int = 4096,
        clock=time.monotonic,
    ):
        if max_failures < 1:
            raise ValueError("max_failures must be >= 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if max_tracked_ips < 1:
            raise ValueError("max_tracked_ips must be >= 1")
        self.max_failures = int(max_failures)
        self.window_seconds = float(window_seconds)
        self.max_tracked_ips = int(max_tracked_ips)
        self.clock = clock
        self._failures: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _normalize_ip(ip: str) -> str:
        cleaned = str(ip or "").strip()[:128]
        return cleaned or "?"

    def _prune_bucket(self, bucket: deque[float], now: float) -> None:
        cutoff = now - self.window_seconds
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

    def _evict_if_needed(self, now: float) -> None:
        if len(self._failures) < self.max_tracked_ips:
            return
        cutoff = now - self.window_seconds
        expired = [k for k, q in self._failures.items() if not q or q[-1] <= cutoff]
        for k in expired:
            self._failures.pop(k, None)
        while len(self._failures) >= self.max_tracked_ips:
            self._failures.popitem(last=False)

    def check_allowed(self, ip: str) -> tuple[bool, int]:
        """Return `(allowed, retry_after_seconds)` for an authentication attempt."""
        key = self._normalize_ip(ip)
        now = float(self.clock())
        with self._lock:
            bucket = self._failures.get(key)
            if bucket is None:
                return True, 0
            self._prune_bucket(bucket, now)
            if not bucket:
                self._failures.pop(key, None)
                return True, 0
            if len(bucket) >= self.max_failures:
                retry_after = max(1, int(math.ceil(self.window_seconds - (now - bucket[0]))))
                return False, retry_after
            return True, 0

    def is_blocked(self, ip: str) -> bool:
        allowed, _ = self.check_allowed(ip)
        return not allowed

    def retry_after(self, ip: str) -> int:
        allowed, retry = self.check_allowed(ip)
        return 0 if allowed else retry

    def record_failure(self, ip: str) -> int:
        """Record a failed auth attempt for `ip` and return active window failure count."""
        key = self._normalize_ip(ip)
        now = float(self.clock())
        with self._lock:
            bucket = self._failures.get(key)
            if bucket is not None:
                self._prune_bucket(bucket, now)
                self._failures.move_to_end(key)
            else:
                self._evict_if_needed(now)
                bucket = deque(maxlen=self.max_failures)
                self._failures[key] = bucket
            bucket.append(now)
            return len(bucket)

    def record_success(self, ip: str) -> None:
        """Clear tracked failure history for `ip` after a successful authentication."""
        key = self._normalize_ip(ip)
        with self._lock:
            self._failures.pop(key, None)

    def reset(self) -> None:
        with self._lock:
            self._failures.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._failures)


limiter = core.RateLimiter(limit=12, window=60)
auth_failure_limiter = AuthFailureRateLimiter(max_failures=5, window_seconds=60.0, max_tracked_ips=4096)

_DPAPI_SECRET_ENV_NAMES = (
    "SHARPEYE_API_KEY",
    "GEMINI_API_KEY",
    "XAI_API_KEY",
    "GROK_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
)


class Query(BaseModel):
    query: str = Field(min_length=3, max_length=200)
    window: str = Field(default=core.DEFAULT_WINDOW, max_length=3)
    keywords: str = Field(default="", max_length=200)
    signal: str = Field(default="", max_length=20)


class ExportRequest(Query):
    format: str = Field(default="csv", max_length=10)


class AlertRule(BaseModel):
    name: str = Field(default="", max_length=120)
    entities: list[str] | str = Field(default_factory=list)
    keywords: list[str] | str = Field(default_factory=list)
    signal: str = Field(default="", max_length=20)
    signals: list[str] | str = Field(default_factory=list)
    min_impact: int = Field(default=0, ge=0, le=100)
    match_all: bool = False


class AlertsRequest(BaseModel):
    query: str = Field(default="", max_length=200)
    window: str = Field(default=core.DEFAULT_WINDOW, max_length=3)
    keywords: str = Field(default="", max_length=200)
    signal: str = Field(default="", max_length=20)
    rules: list[AlertRule | str] = Field(default_factory=list, max_length=50)
    items: list[dict] | None = Field(default=None, max_length=200)


class SwarmSliceInput(BaseModel):
    slice_id: str = Field(min_length=1, max_length=64)
    role: str = Field(default="news_auditor", max_length=64)
    items: list[dict] = Field(default_factory=list, max_length=swarm_orchestrator.MAX_SLICE_ITEMS)
    owned_keys: list[str] = Field(default_factory=list, max_length=swarm_orchestrator.MAX_SLICE_ITEMS)
    requested_tools: list[str] = Field(
        default_factory=lambda: ["score_signals", "build_digest", "detect_injection"],
        max_length=16,
    )
    requested_actions: list[str] = Field(default_factory=list, max_length=16)
    timeout_seconds: float | None = Field(default=None, gt=0, le=swarm_orchestrator.MAX_SLICE_TIMEOUT)


class SwarmRequest(BaseModel):
    query: str = Field(min_length=3, max_length=swarm_orchestrator.MAX_QUERY_CHARS)
    window: str = Field(default=core.DEFAULT_WINDOW, max_length=3)
    provider: str = Field(default="auto", max_length=20)
    max_workers: int = Field(default=swarm_orchestrator.DEFAULT_WORKERS, ge=1, le=swarm_orchestrator.MAX_WORKERS)
    slice_timeout_seconds: float = Field(
        default=swarm_orchestrator.DEFAULT_SLICE_TIMEOUT,
        gt=0,
        le=swarm_orchestrator.MAX_SLICE_TIMEOUT,
    )
    slices: list[SwarmSliceInput] | None = Field(default=None, max_length=swarm_orchestrator.MAX_SLICES)
    items: list[dict] | None = Field(default=None, max_length=100)
    requested_actions: list[str] = Field(default_factory=list, max_length=16)
    llm_enabled: bool = False


def client_ip(request: Request):
    xff = request.headers.get("x-forwarded-for", "")
    if os.getenv("TRUST_PROXY") == "1" and xff:  # Render vb. proxy arkasında
        return xff.split(",")[-1].strip()
    return request.client.host if request.client else "?"


def win(body: Query):
    return body.window if body.window in core.WINDOWS else core.DEFAULT_WINDOW


def unseal_dpapi_env_secrets(env_names=_DPAPI_SECRET_ENV_NAMES, env=None, runner=None) -> int:
    """Unseal any `dpapi:v1:` secrets in `env` in-place; returns count unsealed."""
    target = os.environ if env is None else env
    unsealed_count = 0
    for name in env_names:
        raw = (target.get(name) or "").strip()
        if dpapi_vault.is_dpapi_secret(raw):
            target[name] = dpapi_vault.unseal_secret(raw, runner=runner)
            unsealed_count += 1
    return unsealed_count


def get_configured_api_key(env=None, runner=None) -> str:
    """Return optional SHARPEYE_API_KEY (unsealing `dpapi:v1:` if sealed), or '' when disabled."""
    return dpapi_vault.load_secret("SHARPEYE_API_KEY", default="", env=env, runner=runner)


def _extract_api_key(request: Request) -> str:
    """Read API key strictly from HTTP headers (`Authorization: Bearer` or `X-API-Key`)."""
    auth_header = (request.headers.get("authorization") or "").strip()
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        if token:
            return token
    return (request.headers.get("x-api-key") or "").strip()


def verify_api_key(request: Request):
    """Constant-time optional API-key check with per-IP 429 auth failure limiting.

    Disabled when `SHARPEYE_API_KEY` is unset so default public behavior stays backward-compatible.
    """
    raw_env = (os.environ.get("SHARPEYE_API_KEY") or "").strip()
    if not raw_env:
        return

    ip = client_ip(request)
    allowed, retry_after = auth_failure_limiter.check_allowed(ip)
    if not allowed:
        raise HTTPException(
            429,
            "Çok fazla başarısız kimlik doğrulama denemesi. Lütfen daha sonra tekrar deneyin.",
            headers={"Retry-After": str(retry_after)},
        )

    try:
        expected = get_configured_api_key()
    except (dpapi_vault.DPAPIError, ValueError):
        raise HTTPException(503, "API anahtarı yapılandırması geçersiz.") from None
    if not expected:
        return

    provided = _extract_api_key(request)
    if not provided or not hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8")):
        auth_failure_limiter.record_failure(ip)
        raise HTTPException(
            401,
            "Geçersiz veya eksik API anahtarı.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    auth_failure_limiter.record_success(ip)


def verify_origin(request: Request):
    """Tarayıcı kaynaklı POST/PUT/PATCH/DELETE isteklerinde Origin başlığını doğrular."""
    if getattr(request, "method", "GET").upper() not in ("POST", "PUT", "PATCH", "DELETE"):
        return
    origin = (request.headers.get("origin") or "").strip()
    if not origin:
        return
    p = urlsplit(origin)
    if p.scheme not in ("http", "https") or not p.netloc:
        raise HTTPException(403, "Geçersiz istek kaynağı (Origin).")
    host = (request.headers.get("host") or "").strip().lower()
    allowed_env = {o.strip().lower().rstrip("/") for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()}
    origin_norm = f"{p.scheme.lower()}://{p.netloc.lower()}"
    if p.netloc.lower() != host and origin_norm not in allowed_env:
        raise HTTPException(403, "Geçersiz istek kaynağı (Origin).")


def guard(request: Request):
    verify_origin(request)
    verify_api_key(request)
    if not limiter.allow(client_ip(request)):
        raise HTTPException(429, "Çok fazla istek. Bir dakika sonra tekrar dene.")


async def run(coro):
    try:
        return await coro
    except core.NewsError as e:
        raise HTTPException(e.status, {"message": e.message, "suggestions": e.extra} if e.extra else e.message)


@app.post("/api/news")  # hızlı: çeviri + haber toplama. Sayfa başlıkları hemen gösterir.
async def news(body: Query, request: Request):
    guard(request)
    items = await run(core.get_news(body.query, win(body)))
    if body.keywords or body.signal:
        items = core.filter_items(items, body.keywords, body.signal)
    return {"query": body.query, "items": items}


@app.post("/api/export")  # anlık CSV / JSON istihbarat raporu dışa aktarımı
async def export_news(body: ExportRequest, request: Request):
    guard(request)
    fmt = body.format.strip().lower()
    if fmt not in ("csv", "json"):
        raise HTTPException(422, "Geçersiz dışa aktarma biçimi (csv veya json olmalı).")
    items = await run(core.get_news(body.query, win(body)))
    if body.keywords or body.signal:
        items = core.filter_items(items, body.keywords, body.signal)
    if fmt == "json":
        return Response(content=core.export_items_json(items), media_type="application/json; charset=utf-8")
    return Response(content=core.export_items_csv(items), media_type="text/csv; charset=utf-8")


@app.post("/api/digest")  # yönetici piyasa ve jeopolitik trend özeti (LLM harcamaz, anlık sinyal/tema analizi)
async def digest_news(body: Query, request: Request):
    guard(request)
    items = await run(core.get_news(body.query, win(body)))
    if body.keywords or body.signal:
        items = core.filter_items(items, body.keywords, body.signal)
    return {"query": body.query, "digest": core.build_trend_digest(items)}


@app.post("/api/alerts")  # izleme listesi ve özel alarm kuralları eşleştiricisi (LLM harcamaz)
async def alerts_news(body: AlertsRequest, request: Request):
    guard(request)
    if body.items is not None and not body.query.strip():
        items = list(body.items)
    else:
        if len(body.query.strip()) < 3:
            raise HTTPException(422, "Sorgu en az 3 karakter olmalı.")
        items = await run(core.get_news(body.query, win(body)))
    if body.keywords or body.signal:
        items = core.filter_items(items, body.keywords, body.signal)
    rules = [r.model_dump() if isinstance(r, BaseModel) else r for r in body.rules]
    return {"query": body.query, **core.evaluate_watchlist_alerts(items, rules)}


@app.post("/api/analyze")  # yavaş: Claude analizi (haberler önbellekten gelir)
async def analyze(body: Query, request: Request):
    guard(request)
    return {"query": body.query, "analysis": await run(core.get_analysis(body.query, win(body)))}


@app.get("/api/swarm")  # salt-okunur swarm politika ve yetenek bildirimi
async def swarm_capabilities():
    return swarm_orchestrator.get_swarm_capabilities()


@app.post("/api/swarm")  # paralel salt-okunur çoklu ajan doğrulama ve denetim orkestrasyonu
async def run_swarm(body: SwarmRequest, request: Request):
    guard(request)
    if body.slices is not None:
        slices = [
            swarm_orchestrator.SwarmSlice(
                slice_id=s.slice_id,
                role=s.role,
                items=s.items,
                owned_keys=tuple(s.owned_keys),
                requested_tools=tuple(s.requested_tools),
                requested_actions=tuple(s.requested_actions),
                timeout_seconds=s.timeout_seconds,
            )
            for s in body.slices
        ]
    else:
        if body.items is not None:
            raw_items = list(body.items)
        else:
            w = body.window if body.window in core.WINDOWS else core.DEFAULT_WINDOW
            raw_items = await run(core.get_news(body.query, w))
        slices = swarm_orchestrator.partition_items_into_slices(
            raw_items,
            num_slices=body.max_workers,
        )

    order = swarm_orchestrator.SwarmWorkOrder(
        query=body.query,
        slices=slices,
        provider=body.provider,
        max_workers=body.max_workers,
        slice_timeout_seconds=body.slice_timeout_seconds,
        requested_actions=tuple(body.requested_actions),
        llm_enabled=body.llm_enabled,
    )
    try:
        return await asyncio.to_thread(swarm_orchestrator.run_swarm_orchestration, order)
    except swarm_orchestrator.SwarmPolicyError as exc:
        raise HTTPException(422, str(exc)) from exc
    except core.NewsError as exc:
        raise HTTPException(exc.status, exc.message) from exc


@app.get("/healthz")  # Render'ın sağlık kontrolü için
async def healthz():
    return {"ok": True}


@app.get("/api/ideas")  # keşfet: bugünün gündeminden merak uyandıran fikirler (30 dk önbellek)
async def ideas(request: Request):
    guard(request)
    return {"ideas": await run(core.get_ideas())}


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "index.html", headers={"Cache-Control": "no-cache"})
