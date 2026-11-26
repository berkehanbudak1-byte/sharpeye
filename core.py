"""Haber toplama ve analiz çekirdeği. FastAPI'den bağımsız, tek başına test edilebilir."""
import asyncio, calendar, ipaddress, json, logging, os, re, socket, time
from collections import OrderedDict, deque
from html import unescape
from urllib.parse import quote, urljoin, urlsplit

import feedparser
import httpx

log = logging.getLogger("haber")
# Sağlayıcı: gemini (ücretsiz kota) | ollama (yerel, tamamen ücretsiz) | anthropic (ücretli)
PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
_g, _o = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"), os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
MODELS = {"gemini": (_g, _g), "ollama": (_o, _o),  # (ana model, çeviri modeli)
          "anthropic": (os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"),
                        os.getenv("TRANSLATE_MODEL", "claude-haiku-4-5-20251001"))}
llm_http = httpx.AsyncClient(timeout=180.0)  # yerel modeller yavaş olabilir
http = httpx.AsyncClient(timeout=httpx.Timeout(6.0, connect=3.0), follow_redirects=True,
                         headers={"User-Agent": "Mozilla/5.0 (haber-karsilastir)"})

LANGS = {"tr": ("tr", "TR", "TR:tr"), "en": ("en-US", "US", "US:en"), "de": ("de", "DE", "DE:de"),
         "fr": ("fr", "FR", "FR:fr"), "ru": ("ru", "RU", "RU:ru"), "ar": ("ar", "AE", "AE:ar"),
         "es": ("es", "ES", "ES:es"),
         # aynı dilin farklı ülke sürümleri: daha çeşitli bakış açısı
         "en-gb": ("en-GB", "GB", "GB:en"), "en-in": ("en-IN", "IN", "IN:en"),
         "ar-eg": ("ar", "EG", "EG:ar"), "es-mx": ("es-419", "MX", "MX:es-419")}
BASE = sorted({l.split("-")[0] for l in LANGS})
WINDOWS = {"1d", "7d", "30d", "1y", ""}  # Google News "when:" aralığı; "" = tüm zamanlar
DEFAULT_WINDOW = os.getenv("NEWS_WINDOW", "30d")
WINDOW_SECS = {"1d": 86400, "7d": 7 * 86400, "30d": 30 * 86400, "1y": 365 * 86400}
BG_LIMIT, BG_KEEP = 30, 12  # olayın geçmişi için: edisyon başına çekilen / analize giren eski haber
PER_LANG, TTL, LEVELS = 8, 15 * 60, ("Yüksek", "Orta", "Düşük")


class NewsError(Exception):
    def __init__(self, message, status, extra=None):
        super().__init__(message)
        self.message, self.status, self.extra = message, status, extra


class Memo:
    """TTL'li, sınırlı önbellek. Aynı anahtar için eşzamanlı istekleri tek işe birleştirir."""
    def __init__(self, ttl, maxsize=200):
        self.ttl, self.maxsize, self.data, self.inflight = ttl, maxsize, OrderedDict(), {}

    async def get(self, key, factory):
        hit = self.data.get(key)
        if hit and time.time() - hit[0] < self.ttl:
            self.data.move_to_end(key)
            return hit[1]
        task = self.inflight.get(key)
        if task is None:
            task = self.inflight[key] = asyncio.ensure_future(factory())
            task.add_done_callback(lambda t, k=key: self._done(k, t))
        return await asyncio.shield(task)  # istemci vazgeçse bile iş bitip önbelleğe girer

    def _done(self, key, task):
        self.inflight.pop(key, None)
        if not task.cancelled() and task.exception() is None:
            self.data[key] = (time.time(), task.result())
            while len(self.data) > self.maxsize:
                self.data.popitem(last=False)


class RateLimiter:
    def __init__(self, limit, window, clock=time.monotonic):
        self.limit, self.window, self.clock, self.hits = limit, window, clock, {}

    def allow(self, key):
        now = self.clock()
        q = self.hits.setdefault(key, deque())
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        if len(self.hits) > 5000:
            self.hits = {k: v for k, v in self.hits.items() if v and now - v[-1] <= self.window}
        return True


news_memo, analysis_memo = Memo(TTL), Memo(TTL)
norm = lambda q: re.sub(r"\s+", " ", q.strip().replace("İ", "i").lower())


async def _gemini(model, system, user, max_tokens):
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:  # anahtar yoksa ham KeyError yerine anlaşılır hata (503)
        raise NewsError("Gemini anahtarı tanımlı değil. GEMINI_API_KEY ortam değişkenini ayarlayıp sunucuyu yeni bir terminalden başlatın.", 503)
    r = await llm_http.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key},
        json={"systemInstruction": {"parts": [{"text": system}]},
              "contents": [{"role": "user", "parts": [{"text": user}]}],
              "generationConfig": {"maxOutputTokens": max_tokens * 2, "responseMimeType": "application/json"}})
    r.raise_for_status()
    return "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])


async def _ollama(model, system, user, max_tokens):
    r = await llm_http.post(f"{OLLAMA_URL}/api/chat", json={
        "model": model, "stream": False, "format": "json",
        "options": {"num_predict": max_tokens, "num_ctx": 8192},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
    r.raise_for_status()
    return r.json()["message"]["content"]


_ac = None


async def _anthropic(model, system, user, max_tokens):
    global _ac
    if _ac is None:
        import anthropic  # yalnızca bu sağlayıcı seçilirse gerekir
        _ac = anthropic.AsyncAnthropic(timeout=60.0, max_retries=2)
    r = await _ac.messages.create(model=model, max_tokens=max_tokens, system=system,
                                  messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in r.content if b.type == "text")


async def ask(kind, system, user, max_tokens):  # kind: "main" | "fast"
    main, fast = MODELS[PROVIDER]
    call = {"gemini": _gemini, "ollama": _ollama, "anthropic": _anthropic}[PROVIDER]
    return await call(fast if kind == "fast" else main, system, user, max_tokens)


def parse_json(raw):
    raw = re.sub(r"```json|```", "", raw)
    return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])


TR_SYSTEM = """Kullanıcı bir haber konusu yazdı. SADECE JSON döndür.
Kişi, kurum, yer, olay, ürün adları ve somut konular geçerlidir; bunları istenen dillere çevir:
{"vague":false,"tr":"...","en":"..."}
Metin haber aranamayacak kadar belirsizse (tek soru kelimesi: "nedir","ne","kim"; anlamsız ya da aşırı genel metin) çevirme, şunu döndür:
{"vague":true,"why":"Türkçe tek cümle: neden yetersiz","ideas":["yazılana yakın 4 somut haber konusu"]}"""


async def translate(query):
    """(çeviriler, sorun). Konu belirsizse çeviriler None, sorun {why, ideas} olur."""
    try:
        d = parse_json(await ask("fast", TR_SYSTEM, f"Diller: {BASE}\nKonu: {query}", 700))
    except Exception:
        log.warning("çeviri başarısız, orijinal sorgu kullanılıyor")
        return {l: query for l in LANGS}, None
    if d.get("vague") is True:
        ideas = d.get("ideas")
        return None, {"why": str(d.get("why", "")), "ideas": [str(x) for x in ideas[:4]] if isinstance(ideas, list) else []}
    return {l: d.get(l.split("-")[0]) or query for l in LANGS}, None


_SIGNALS = {
    "acil": ("son dakika", "acil", "breaking", "urgent", "flash", "alarm", "deprem", "earthquake", "patlama", "explosion", "saldırı", "attack", "olağanüstü", "emergency"),
    "kritik": ("kritik", "critical", "savaş", "war", "kriz", "crisis", "yaptırım", "sanction", "çöküş", "collapse", "iflas", "bankruptcy", "tehlike", "catastrophe", "rekor", "record"),
    "pozitif": ("anlaşma", "barış", "peace", "büyüme", "growth", "başarı", "success", "rekor artış", "iyileşme", "recovery", "onay", "approved", "yatırım", "investment", "işbirliği"),
    "negatif": ("düşüş", "kayıp", "loss", "ölüm", "dead", "killed", "gerilim", "tension", "çatışma", "conflict", "enflasyon", "inflation", "yasak", "ban", "tehdit", "threat", "zarar"),
}


def score_item(title: str) -> dict:
    """Başlıktan hızlı etki/aciliyet sinyali (acil, kritik, pozitif, negatif) ve 0-100 etki puanı üretir."""
    low = norm(title or "")
    tags = []
    for sig, words in _SIGNALS.items():
        if any(w in low for w in words):
            tags.append(sig)
    score = 20
    if "acil" in tags:
        score += 45
    if "kritik" in tags:
        score += 25
    if "negatif" in tags:
        score += 10
    if "pozitif" in tags:
        score += 10
    return {"signal": tags[0] if tags else "normal", "signals": tags or ["normal"], "impact_score": min(100, score)}


def filter_items(items, keywords="", signal=""):
    """Çoklu anahtar kelime (virgül/boşluk ayrılmış) ve sinyal etiketine göre haberleri süzer."""
    kws = [norm(k) for k in re.split(r"[,;]+", keywords or "") if norm(k)]
    sig = norm(signal or "")
    out = []
    for it in items or []:
        if sig and sig != "all":
            it_sigs = it.get("signals") or [it.get("signal", "normal")]
            if sig not in it_sigs and it.get("signal") != sig:
                continue
        if kws:
            hay = norm(f"{it.get('title', '')} {it.get('source', '')} {it.get('lang', '')}")
            if not any(k in hay for k in kws):
                continue
        out.append(it)
    return out


def export_items_csv(items) -> str:
    """Haber listesini UTF-8 CSV metnine dönüştürür."""
    import csv
    import io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["n", "lang", "source", "date", "signal", "impact_score", "title", "link"])
    for it in items or []:
        sc = score_item(it.get("title", "")) if "signal" not in it else it
        writer.writerow([
            it.get("n", ""),
            it.get("lang", ""),
            it.get("source", ""),
            it.get("date", ""),
            sc.get("signal", "normal"),
            sc.get("impact_score", 20),
            it.get("title", ""),
            it.get("link", ""),
        ])
    return buf.getvalue()


def export_items_json(items) -> str:
    """Haber listesini yapılandırılmış JSON raporuna dönüştürür."""
    enriched = []
    for it in items or []:
        row = dict(it)
        if "signal" not in row or "impact_score" not in row:
            row.update(score_item(row.get("title", "")))
        enriched.append(row)
    return json.dumps({"count": len(enriched), "items": enriched}, ensure_ascii=False, indent=2)


_DIGEST_STOP = {
    "ve", "ile", "için", "bir", "bu", "da", "de", "ama", "ne", "kadar", "gibi", "ise", "olan", "olarak",
    "daha", "en", "çok", "sonra", "önce", "mi", "mı", "mu", "mü", "ya", "yani",
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were", "will", "into", "over", "after",
    "new", "news", "der", "die", "das", "und", "von", "mit", "les", "des", "une", "dans", "pour", "del", "las", "los",
    "por", "con", "para", "haber", "son", "dakika", "göre", "üzerine", "hakkında",
}


def detect_breaking_surges(items, min_sources: int = 2, min_avg_impact: float = 35.0) -> list:
    """Farklı bağımsız kaynakların/alan adlarının aynı varlık veya anahtar kelime kümesini eşzamanlı ve yüksek etki puanıyla haberleştirdiği anlık dalgaları (breaking surge) tespit eder."""
    active = [dict(it) for it in (items or []) if not it.get("bg")]
    if not active and items:
        active = [dict(it) for it in items]
    if not active:
        return []

    clusters = {}
    for it in active:
        if "signal" not in it or "impact_score" not in it or "signals" not in it:
            it.update(score_item(it.get("title", "")))
        src = (it.get("source") or "").strip()
        if not src or src == "?":
            host = urlsplit(str(it.get("link") or "")).hostname or ""
            src = host.lower().removeprefix("www.") or "unknown"
        lang = (it.get("lang") or "").strip()
        impact = int(it.get("impact_score", 20))
        words = {
            w for w in re.findall(r"[a-zçğıöşü0-9]+", norm(it.get("title", "")))
            if len(w) >= 3 and w not in _DIGEST_STOP and not w.isdigit()
        }
        for w in words:
            entry = clusters.setdefault(w, {"sources": set(), "langs": set(), "impacts": [], "items": 0})
            entry["sources"].add(src)
            if lang:
                entry["langs"].add(lang)
            entry["impacts"].append(impact)
            entry["items"] += 1

    surges = []
    for kw, data in clusters.items():
        src_count = len(data["sources"])
        if src_count < max(2, int(min_sources)):
            continue
        impacts = data["impacts"]
        avg_imp = round(sum(impacts) / len(impacts), 2) if impacts else 0.0
        max_imp = max(impacts) if impacts else 0
        confirmed = bool(src_count >= max(2, int(min_sources)) and avg_imp >= float(min_avg_impact))
        surges.append({
            "keyword": kw,
            "confirmed_surge": confirmed,
            "source_count": src_count,
            "sources": sorted(data["sources"]),
            "lang_count": len(data["langs"]),
            "item_count": data["items"],
            "avg_impact": avg_imp,
            "max_impact": max_imp,
        })

    surges.sort(key=lambda s: (not s["confirmed_surge"], -s["source_count"], -s["avg_impact"], s["keyword"]))
    return surges[:8]


def build_trend_digest(items) -> dict:
    """Haber listesinden yönetici piyasa ve jeopolitik trend özeti (temalar, duygu dengesi, kritik uyarı oranı, en yüksek etkili 3 başlık, çok kaynaklı kırılma dalgaları) üretir."""
    from collections import Counter
    active = [dict(it) for it in (items or []) if not it.get("bg")]
    if not active and items:
        active = [dict(it) for it in items]
    if not active:
        return {
            "total_items": 0,
            "sentiment_balance": "neutral",
            "sentiment_label_tr": "notr",
            "sentiment_counts": {"pozitif": 0, "negatif": 0, "notr": 0},
            "critical_alert_count": 0,
            "critical_alert_ratio": 0.0,
            "top_themes": [],
            "top_headlines": [],
            "surges": [],
        }

    word_counts = Counter()
    pos_count = 0
    neg_count = 0
    neu_count = 0
    crit_count = 0

    for it in active:
        if "signal" not in it or "impact_score" not in it or "signals" not in it:
            it.update(score_item(it.get("title", "")))
        sigs = set(it.get("signals") or [it.get("signal", "normal")])
        if "acil" in sigs or "kritik" in sigs:
            crit_count += 1
        is_pos = "pozitif" in sigs
        is_neg = "negatif" in sigs
        if is_pos and not is_neg:
            pos_count += 1
        elif is_neg and not is_pos:
            neg_count += 1
        else:
            neu_count += 1

        words = [w for w in re.findall(r"[a-zçğıöşü0-9]+", norm(it.get("title", ""))) if len(w) >= 3 and w not in _DIGEST_STOP and not w.isdigit()]
        for w in set(words):
            word_counts[w] += 1

    total = len(active)
    if pos_count > neg_count:
        balance, label_tr = "bullish", "pozitif"
    elif neg_count > pos_count:
        balance, label_tr = "bearish", "negatif"
    else:
        balance, label_tr = "neutral", "notr"

    top_themes = [{"theme": w, "count": c} for w, c in word_counts.most_common(5)]
    ranked = sorted(active, key=lambda x: (-int(x.get("impact_score", 20)), int(x.get("n") or 0)))
    top_headlines = [
        {
            "n": it.get("n", i + 1),
            "lang": it.get("lang", ""),
            "source": it.get("source", ""),
            "title": it.get("title", ""),
            "signal": it.get("signal", "normal"),
            "impact_score": it.get("impact_score", 20),
            "link": it.get("link", ""),
        }
        for i, it in enumerate(ranked[:3])
    ]

    return {
        "total_items": total,
        "sentiment_balance": balance,
        "sentiment_label_tr": label_tr,
        "sentiment_counts": {"pozitif": pos_count, "negatif": neg_count, "notr": neu_count},
        "critical_alert_count": crit_count,
        "critical_alert_ratio": round(crit_count / total, 4),
        "top_themes": top_themes,
        "top_headlines": top_headlines,
        "surges": detect_breaking_surges(active),
    }


def _alert_norm(text: str) -> str:
    return norm(str(text or "")).replace("\u0307", "")


def _parse_rule_terms(val) -> list[str]:
    if not val:
        return []
    if isinstance(val, str):
        raw = re.split(r"[,;]+", val)
    elif isinstance(val, (list, tuple, set)):
        raw = [str(x) for x in val]
    else:
        raw = [str(val)]
    out = []
    seen = set()
    for term in raw:
        cleaned = re.sub(r"\s+", " ", str(term or "")).strip()
        key = _alert_norm(cleaned)
        if key and key not in seen:
            seen.add(key)
            out.append(cleaned)
    return out


def _term_in_text(term_norm: str, hay_norm: str) -> bool:
    if not term_norm or not hay_norm:
        return False
    if len(term_norm) <= 3 and re.fullmatch(r"[a-z0-9]+", term_norm):
        return bool(re.search(rf"(?<!\w){re.escape(term_norm)}(?!\w)", hay_norm))
    return term_norm in hay_norm


def evaluate_watchlist_alerts(items, rules) -> dict:
    """Haber listesini varlık/anahtar kelime, sinyal ve minimum etki puanı kurallarına göre değerlendirip yapılandırılmış alarm bildirimleri döndürür."""
    active = [dict(it) for it in (items or []) if not it.get("bg")]
    if not active and items:
        active = [dict(it) for it in items]

    enriched = []
    for idx, it in enumerate(active):
        row = dict(it)
        if "signal" not in row or "impact_score" not in row or "signals" not in row:
            row.update(score_item(row.get("title", "")))
        row["_idx"] = idx
        row["_hay"] = _alert_norm(f"{row.get('title', '')} {row.get('source', '')} {row.get('summary', '')}")
        row["_sigs"] = {_alert_norm(s) for s in (row.get("signals") or [row.get("signal", "normal")]) if s}
        if row.get("signal"):
            row["_sigs"].add(_alert_norm(str(row["signal"])))
        enriched.append(row)

    rule_list = [rules] if isinstance(rules, (dict, str)) else list(rules or [])
    normalized_rules = []
    for idx, raw_rule in enumerate(rule_list):
        if isinstance(raw_rule, str):
            terms = _parse_rule_terms(raw_rule)
            if not terms:
                continue
            normalized_rules.append({
                "name": terms[0],
                "entities": terms,
                "keywords": [],
                "signals": [],
                "min_impact": 0,
                "match_all": False,
            })
        elif isinstance(raw_rule, dict):
            entities = _parse_rule_terms(raw_rule.get("entities") or raw_rule.get("entity"))
            keywords = _parse_rule_terms(raw_rule.get("keywords") or raw_rule.get("keyword"))
            sig_terms = _parse_rule_terms(raw_rule.get("signals") or raw_rule.get("signal"))
            allowed_sigs = [_alert_norm(s) for s in sig_terms if _alert_norm(s) and _alert_norm(s) != "all"]
            raw_min = raw_rule.get("min_impact", raw_rule.get("min_impact_score", 0))
            min_impact = max(0, min(100, int(float(raw_min or 0))))
            name = str(raw_rule.get("name") or raw_rule.get("rule") or raw_rule.get("label") or "").strip()
            if not name:
                label_parts = entities[:2] + keywords[:2]
                name = ", ".join(label_parts) if label_parts else f"rule-{idx + 1}"
            normalized_rules.append({
                "name": name,
                "entities": entities,
                "keywords": keywords,
                "signals": allowed_sigs,
                "min_impact": min_impact,
                "match_all": bool(raw_rule.get("match_all", False)),
            })

    alerts = []
    unique_matched_indices = set()
    for rule in normalized_rules:
        matches = []
        rule_entities_hit = set()
        rule_keywords_hit = set()
        rule_signals_hit = set()

        for row in enriched:
            impact = int(row.get("impact_score", 20))
            if impact < rule["min_impact"]:
                continue
            if rule["signals"] and not (row["_sigs"] & set(rule["signals"])):
                continue

            matched_ent = [e for e in rule["entities"] if _term_in_text(_alert_norm(e), row["_hay"])]
            if rule["entities"]:
                if rule["match_all"] and len(matched_ent) != len(rule["entities"]):
                    continue
                if not matched_ent:
                    continue

            matched_kw = [k for k in rule["keywords"] if _term_in_text(_alert_norm(k), row["_hay"])]
            if rule["keywords"]:
                if rule["match_all"] and len(matched_kw) != len(rule["keywords"]):
                    continue
                if not matched_kw:
                    continue

            unique_matched_indices.add(row["_idx"])
            rule_entities_hit.update(matched_ent)
            rule_keywords_hit.update(matched_kw)
            rule_signals_hit.update(s for s in (row.get("signals") or [row.get("signal", "normal")]) if s)
            matches.append({
                "n": row.get("n", row["_idx"] + 1),
                "lang": row.get("lang", ""),
                "source": row.get("source", ""),
                "title": row.get("title", ""),
                "signal": row.get("signal", "normal"),
                "signals": row.get("signals", ["normal"]),
                "impact_score": impact,
                "link": row.get("link", ""),
                "matched_entities": matched_ent,
                "matched_keywords": matched_kw,
            })

        if matches:
            matches.sort(key=lambda m: (-int(m["impact_score"]), int(m.get("n") or 0)))
            highest_impact = max(int(m["impact_score"]) for m in matches)
            alerts.append({
                "rule": rule["name"],
                "match_count": len(matches),
                "highest_impact": highest_impact,
                "matched_entities": sorted(rule_entities_hit),
                "matched_keywords": sorted(rule_keywords_hit),
                "signals": sorted(rule_signals_hit),
                "matches": matches,
            })

    alerts.sort(key=lambda a: (-a["highest_impact"], -a["match_count"], a["rule"]))
    return {
        "triggered": bool(alerts),
        "alerts": alerts,
        "summary": {
            "total_items": len(enriched),
            "total_rules": len(normalized_rules),
            "triggered_rules": len(alerts),
            "matched_items": len(unique_matched_indices),
            "highest_impact": max((a["highest_impact"] for a in alerts), default=0),
            "critical_alerts": sum(1 for a in alerts if a["highest_impact"] >= 65),
        },
    }


def clean_title(title, source):
    title = re.sub(r"\s+", " ", title or "").strip()
    suffix = f" - {source}"
    return title[: -len(suffix)] if source and title.endswith(suffix) else title


async def fetch_rss(q, lang, window="", limit=PER_LANG):
    hl, gl, ceid = LANGS[lang]
    try:
        r = await http.get(f"https://news.google.com/rss/search?q={quote(q + (' when:' + window if window else ''))}&hl={hl}&gl={gl}&ceid={ceid}")
        r.raise_for_status()
    except Exception as e:
        log.warning("RSS alınamadı (%s): %s", lang, type(e).__name__)
        return []
    out = []
    for e in feedparser.parse(r.content).entries[:limit]:
        src = getattr(e, "source", None)
        source = src.get("title", "?") if src else "?"
        pp = getattr(e, "published_parsed", None)
        title = clean_title(e.get("title", ""), source)
        sig = score_item(title)
        out.append({"lang": lang, "q": q, "title": title, "source": source,
                    "date": time.strftime("%d.%m", pp) if pp else "", "ts": calendar.timegm(pp) if pp else 0,
                    "link": e.get("link", ""), **sig})
    return out


async def _background(queries, window, seen):
    """Seçilen aralıktan DAHA ESKİ haberlerden, zamana yayılmış örnek seçer (olayın geçmişi için)."""
    cutoff = time.time() - WINDOW_SECS[window]
    groups = await asyncio.gather(*(fetch_rss(q, l, "", BG_LIMIT) for l, q in queries.items()))
    old = {}
    for it in (i for g in groups for i in g):
        k = (it["source"], re.sub(r"\W+", "", it["title"].lower())[:60])
        if it["title"] and it["ts"] and it["ts"] < cutoff and k not in seen:
            seen.add(k)
            old[k] = it
    old = sorted(old.values(), key=lambda i: i["ts"])
    if len(old) > BG_KEEP:
        old = [old[round(i * (len(old) - 1) / (BG_KEEP - 1))] for i in range(BG_KEEP)]
    for it in old:
        it["bg"] = True
    return old


async def _load_news(query, window=""):
    queries, problem = await translate(query)
    if problem:
        raise NewsError(problem["why"] or "Bu bir haber konusu gibi görünmüyor; biraz daha somut yaz.", 422, problem["ideas"])
    groups = await asyncio.gather(*(fetch_rss(q, l, window) for l, q in queries.items()))  # 7 dil paralel
    seen, items = set(), []
    for it in (i for g in groups for i in g):
        k = (it["source"], re.sub(r"\W+", "", it["title"].lower())[:60])  # aynı kaynak+başlık tekrarı elenir;
        if it["title"] and k not in seen:                                 # farklı kaynakta aynı başlık kalır
            seen.add(k)
            items.append(it)
    if len(items) < 3:
        raise NewsError("Yeterli haber bulunamadı. Konuyu değiştirip ya da zaman aralığını genişletip tekrar dene.", 404)
    if window in WINDOW_SECS:
        items += await _background(queries, window, seen)
    for k, it in enumerate(items, 1):
        it["n"] = k  # analizdeki [n] numarasıyla aynı
    return items


SYSTEM = """Sen titiz bir haber doğrulama ve karşılaştırma analistisin. Farklı dillerdeki haber başlıkları verilecek.
Kurallar:
- SADECE verilen başlık ve metinlere dayan; olmayanı uydurma. "Kesin doğru" ilan etme, güven düzeyi belirt.
- Bazı haberlerde METİN (haberin ilk bölümü) verilir: iddiaları önce metinlerden çıkar, başlıkla metin çelişiyorsa belirt. Metni olmayan haberler yalnızca başlığa dayanır.
- Ortak iddia için bağımsız en az 2 kaynak, mümkünse 2 dil gerekir. Aynı ajansın (AP, Reuters, AFP) haberini yayımlayanlar tek kaynaktır.
- Ayrıntılı ve somut ol: isim, sayı, tarih, yer varsa yaz. Genel geçer cümle kurma.
- Özet ve her maddenin sonuna dayandığın kaynak numaralarını köşeli parantezle ekle, örn. [3,7,12] (numaralar [n] değerleridir).
- Metinlerdeki somut ayrıntıları (rakam, isim, yer, tarih, kısa alıntı) mutlaka aktar; başlıkta olmayan ayrıntıyı metinden al. "numbers" en çok 6 rakam içersin; kaynaklar farklı rakam veriyorsa value'da aralık yaz.
- "timeline", OLAYIN kendi kronolojisidir (haberlerin yayın zamanı DEĞİL): en çok 12 aşama, eskiden yeniye. iso = olayın tarihi (YYYY-MM-DD); metin/başlıktan çıkar, kesin değilse yayın tarihine bakarak tahmin et ve when'e "yaklaşık" yaz, çıkaramazsan iso'yu boş bırak. Her aşamanın sonuna kaynak numarası ekle.
- "ARKA PLAN" işaretli haberler olayın daha eski geçmişidir: timeline'ı olayın başlangıcına kadar geriye uzatmak ve "inside" kısmını yazmak için kullan; güncel ortak/çelişki analizine katma.
- "curious": okuyucunun merak edebileceği en çok 4 takip sorusu (neden/nasıl/başka ülke açısından) ve her biri için kısa arama ifadesi (query).
- Türkçe yaz. summary 3-4 cümle: ne oldu ve şu anki durum. "inside" 4-6 cümle: olayın iç yüzü — neden bu noktaya gelindi, kilit aktörler ve çıkarları, arka plan; kaynakta olmayanı ekleme, belirsizliği belirt. Listeler en fazla 8 madde, madde başına en fazla 40 kelime. SADECE şu JSON:
{"summary":"...","inside":"...","facts":["rakam/isim/tarih/yer içeren somut bilgi [n]"],"numbers":[{"value":"kısa rakam, örn. 1.200 veya %35","label":"neyin rakamı [n]"}],"timeline":[{"when":"12 Ekim 2026, 14:30","iso":"2026-10-12","what":"olay [n]"}],"common":["..."],
 "by_country":[{"who":"ülke/dil","says":"bu kaynakların ağırlıklı anlatısı [n]"}],
 "conflicts":[{"claim":"...","sides":[{"who":"kaynak/dil","says":"..."}]}],
 "framing":["..."],"unverified":["..."],"curious":[{"q":"okuyucunun merak edebileceği takip sorusu","query":"2-5 kelimelik arama ifadesi"}],"missing":["başlıklardan anlaşılamayan, ek araştırma gerektiren sorular"],
 "confidence":{"level":"Yüksek|Orta|Düşük","reason":"tek cümle"}}"""


def _strs(v):
    return [str(x) for x in v[:8]] if isinstance(v, list) else []


def _timeline(v):
    out = []
    for x in (v[:12] if isinstance(v, list) else []):
        if isinstance(x, dict):
            iso = str(x.get("iso", ""))
            out.append({"when": str(x.get("when", "")), "what": str(x.get("what", "")),
                        "iso": iso if re.fullmatch(r"\d{4}-\d{2}-\d{2}", iso) else ""})
    return out


def _pairs(v, k1, k2):
    return [{k1: str(x.get(k1, "")), k2: str(x.get(k2, ""))}
            for x in (v[:8] if isinstance(v, list) else []) if isinstance(x, dict)]


def normalize(a):
    if not isinstance(a, dict):
        raise ValueError("beklenmeyen biçim")
    conflicts = []
    for c in a.get("conflicts") or []:
        if isinstance(c, dict) and isinstance(c.get("sides"), list):
            sides = [{"who": str(s.get("who", "")), "says": str(s.get("says", ""))}
                     for s in c["sides"][:4] if isinstance(s, dict)]
            conflicts.append({"claim": str(c.get("claim", "")), "sides": sides})
    conf = a["confidence"] if isinstance(a.get("confidence"), dict) else {}
    return {"summary": str(a.get("summary", "")), "inside": str(a.get("inside", "")), "common": _strs(a.get("common")),
            "facts": _strs(a.get("facts")),
            "numbers": _pairs(a.get("numbers"), "value", "label"),
            "timeline": _timeline(a.get("timeline")),
            "by_country": _pairs(a.get("by_country"), "who", "says"),
            "missing": _strs(a.get("missing")),
            "curious": _pairs(a.get("curious"), "q", "query"),
            "conflicts": conflicts[:5], "framing": _strs(a.get("framing")),
            "unverified": _strs(a.get("unverified")),
            "confidence": {"level": conf.get("level") if conf.get("level") in LEVELS else "Düşük",
                           "reason": str(conf.get("reason", ""))}}


MAX_READ, EXCERPT_CHARS, READ_TIMEOUT = int(os.getenv("MAX_READ", "20")), int(os.getenv("EXCERPT_CHARS", "3000")), 40
text_http = httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=4.0), follow_redirects=False, headers={
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
    "Accept-Language": "en,tr;q=0.8"})


def _decode_sync(url):
    """Google News yönlendirme linkini yayıncının gerçek adresine çevirir; olmazsa None."""
    try:
        try:
            from gnews_decoder import decode
            r = decode(url)
        except ImportError:
            from googlenewsdecoder import gnewsdecoder
            r = gnewsdecoder(url)
    except Exception:
        return None
    if isinstance(r, dict):
        r = r.get("decoded_url") or r.get("url")
    return r if isinstance(r, str) and r.startswith("http") else None


def _extract(html):
    import trafilatura
    return (trafilatura.extract(html, include_comments=False, include_tables=False) or "").strip()


def _og_image(html, base):
    """Sayfanın og:image görselini döndürür. Yalnızca https kabul edilir."""
    m = re.search(r'<meta[^>]+(?:property|name)=["\'](?:og|twitter):image["\'][^>]*>', html[:60000], re.IGNORECASE)
    v = re.search(r'content=["\']([^"\']+)', m.group(0), re.IGNORECASE) if m else None
    u = urljoin(base, unescape(v.group(1))) if v else None
    return u if u and u.startswith("https://") else None


def pick_for_reading(items, n=MAX_READ):
    """Dilleri sırayla dolaşarak, her yayıncıdan en fazla 1 haber seçer (çeşitlilik için)."""
    by_lang, chosen, used = {}, [], set()
    for i in (x for x in items if not x.get("bg")):
        by_lang.setdefault(i["lang"], []).append(i)
    while len(chosen) < n and any(by_lang.values()):
        for lst in by_lang.values():
            while lst:
                it = lst.pop(0)
                if it["source"] not in used:
                    used.add(it["source"])
                    chosen.append(it)
                    break
            if len(chosen) >= n:
                break
    return chosen


def _host_public(host):
    """Ad yalnızca genel IP adreslerine çözülüyorsa True. Döngü (127.x), özel ağ ve bulut metadata adresleri reddedilir."""
    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError:
        return False
    return bool(infos) and all(ipaddress.ip_address(i[4][0]).is_global for i in infos)


async def _safe_get(url):
    """Yayıncı sayfası: yalnızca https ve genel adres. Yönlendirmeler tek tek denetlenir (en fazla 3). Olmazsa None."""
    for _ in range(4):
        p = urlsplit(url)
        try:
            port = p.port
        except ValueError:
            return None
        if p.scheme != "https" or not p.hostname or p.username or p.password or port not in (None, 443):
            return None
        if not await asyncio.to_thread(_host_public, p.hostname):
            return None
        r = await text_http.get(url, follow_redirects=False)
        if getattr(r, "is_redirect", False) or getattr(r, "status_code", 0) in (301, 302, 303, 307, 308):
            loc = r.headers.get("location", "")
            if not loc:
                return None
            url = urljoin(url, loc)
            continue
        return r
    return None


async def read_one(it, sem):
    """(metin, görsel, sonuç_kodu). Kodlar: ok, decode, fetch, type, short."""
    async with sem:
        try:
            url = await asyncio.to_thread(_decode_sync, it["link"])
            if not url:
                return None, None, "decode"
            r = await _safe_get(url)
            if r is None:
                return None, None, "fetch"
            r.raise_for_status()
            if "html" not in r.headers.get("content-type", "html").lower():
                return None, None, "type"
            text = await asyncio.to_thread(_extract, r.text)
            img = _og_image(r.text, str(getattr(r, "url", url)))
        except Exception as e:
            log.info("metin okunamadı (%s): %s", it["source"], type(e).__name__)
            return None, None, "fetch"
    if len(text) > 200:
        return text[:EXCERPT_CHARS], img, "ok"
    return None, img, "short"  # paywall/boş sayfalar


async def read_articles(items):
    """({no: metin}, {no: görsel_url}, istatistik). Okunamayanlar sessizce atlanır ama sayılır."""
    sem, chosen = asyncio.Semaphore(6), pick_for_reading(items)
    tasks = [asyncio.ensure_future(read_one(it, sem)) for it in chosen]
    done, pending = await asyncio.wait(tasks, timeout=READ_TIMEOUT)
    for p in pending:
        p.cancel()
    texts, images, why = {}, {}, {}
    for it, t in zip(chosen, tasks):
        if t not in done:
            why["timeout"] = why.get("timeout", 0) + 1
            continue
        txt, img, w = t.result()
        why[w] = why.get(w, 0) + 1
        if txt:
            texts[it["n"]] = txt
        if img:
            images[it["n"]] = img
    log.info("metin okuma: %s/%s %s", len(texts), len(chosen), why)
    return texts, images, {"tried": len(chosen), "read": len(texts), "why": why}


def _today():
    return time.strftime("%Y-%m-%d")


def _day(i):
    return time.strftime("%Y-%m-%d", time.gmtime(i["ts"])) if i.get("ts") else ""


async def _analyze(query, window=DEFAULT_WINDOW):
    items = await get_news(query, window)
    texts, images, stats = await read_articles(items)
    lines = []
    for i in items:
        lines.append(f"[{i['n']}] {'ARKA PLAN ' if i.get('bg') else ''}({i['lang']}) {i['source']} {_day(i)}: {i['title']}")
        if i["n"] in texts:
            lines.append(f"    METİN: {texts[i['n']]}")
    try:
        out = normalize(parse_json(await ask("main", SYSTEM, f"Bugünün tarihi: {_today()}\nKonu: {query}\n\n" + "\n".join(lines), 5000)))
        out["read"], out["images"], out["stats"] = sorted(texts), images, stats
        return out
    except NewsError:
        raise
    except Exception as e:
        if getattr(getattr(e, "response", None), "status_code", None) == 429:
            raise NewsError("Ücretsiz yapay zekâ kotası doldu. Biraz sonra tekrar dene.", 429)
        log.warning("analiz hatası: %s", type(e).__name__)
        raise NewsError("Analiz üretilemedi. Biraz sonra tekrar dene.", 502)


async def get_news(q, window=DEFAULT_WINDOW):
    return await news_memo.get(norm(q) + "|" + window, lambda: _load_news(q, window))


async def get_analysis(q, window=DEFAULT_WINDOW):
    return await analysis_memo.get(norm(q) + "|" + window, lambda: _analyze(q, window))


# ---- Keşfet: bugünün gündeminden merak uyandıran araştırma fikirleri
IDEAS_SYSTEM = """Farklı ülkelerin Google News ana sayfa başlıkları verilecek. Merak uyandıran 6 araştırma fikri seç:
tercihen birden çok ülkede geçen ya da ülkeye göre farklı çerçevelenen konular; siyaset, ekonomi, bilim, sağlık, teknoloji, toplum gibi çeşitli alanlardan.
Yalnızca verilen başlıklara dayan. Her fikir için: question (Türkçe, merak uyandıran kısa soru, en fazla 14 kelime), hook (Türkçe tek cümle: neden ilginç), query (Türkçe, 2-5 kelimelik arama ifadesi).
SADECE JSON: {"ideas":[{"question":"","hook":"","query":""}]}"""
ideas_memo = Memo(30 * 60)


async def fetch_top(lang):
    hl, gl, ceid = LANGS[lang]
    try:
        r = await http.get(f"https://news.google.com/rss?hl={hl}&gl={gl}&ceid={ceid}")
        r.raise_for_status()
    except Exception as e:
        log.warning("gündem alınamadı (%s): %s", lang, type(e).__name__)
        return []
    out = []
    for e in feedparser.parse(r.content).entries[:6]:
        src = getattr(e, "source", None)
        out.append(f"({lang}) " + clean_title(e.get("title", ""), src.get("title", "") if src else ""))
    return out


def _clean_ideas(d):
    raw = d.get("ideas") if isinstance(d, dict) else None
    out = []
    for x in (raw[:6] if isinstance(raw, list) else []):
        if isinstance(x, dict) and str(x.get("query", "")).strip() and str(x.get("question", "")).strip():
            out.append({k: str(x.get(k, "")).strip() for k in ("question", "hook", "query")})
    return out


async def _ideas():
    lines = [x for g in await asyncio.gather(*(fetch_top(l) for l in LANGS)) for x in g]
    if len(lines) < 8:
        raise NewsError("Gündem şu an alınamadı.", 503)
    try:
        ideas = _clean_ideas(parse_json(await ask("main", IDEAS_SYSTEM, "\n".join(lines), 1500)))
    except Exception:
        raise NewsError("Fikir üretilemedi.", 502)
    if not ideas:
        raise NewsError("Fikir üretilemedi.", 502)
    return ideas


async def get_ideas():
    return await ideas_memo.get("ideas", _ideas)
