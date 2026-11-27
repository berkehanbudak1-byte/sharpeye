"""Çalıştır: uvicorn app:app   (önbellek ve istek sınırı süreç içi olduğundan tek worker kullan)"""
import asyncio
import base64
import hashlib
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

import core
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


limiter = core.RateLimiter(limit=12, window=60)


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
    return {
        **swarm_orchestrator.get_swarm_capabilities(),
        "recent_runs": swarm_orchestrator.get_swarm_history(),
    }


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
