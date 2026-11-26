"""Çalıştır: uvicorn app:app   (önbellek ve istek sınırı süreç içi olduğundan tek worker kullan)"""
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

import core

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=500)
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
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
    items: list[dict] | None = None


def client_ip(request: Request):
    xff = request.headers.get("x-forwarded-for", "")
    if os.getenv("TRUST_PROXY") == "1" and xff:  # Render vb. proxy arkasında
        return xff.split(",")[-1].strip()
    return request.client.host if request.client else "?"


def win(body: Query):
    return body.window if body.window in core.WINDOWS else core.DEFAULT_WINDOW


def guard(request: Request):
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
    items = await run(core.get_news(body.query, win(body)))
    if body.keywords or body.signal:
        items = core.filter_items(items, body.keywords, body.signal)
    if body.format.lower() == "json":
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
