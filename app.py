"""Çalıştır: uvicorn app:app   (önbellek ve istek sınırı süreç içi olduğundan tek worker kullan)"""
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import core

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=500)
limiter = core.RateLimiter(limit=12, window=60)


class Query(BaseModel):
    query: str = Field(min_length=3, max_length=200)
    window: str = Field(default=core.DEFAULT_WINDOW, max_length=3)


def client_ip(request: Request):
    xff = request.headers.get("x-forwarded-for", "")
    if os.getenv("TRUST_PROXY") == "1" and xff:  # Render vb. proxy arkasında
        return xff.split(",")[0].strip()
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
    return {"query": body.query, "items": await run(core.get_news(body.query, win(body)))}


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
