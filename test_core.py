"""Çevrimdışı testler: anthropic/httpx/feedparser sahte modüllerle değiştirilir. Çalıştır: python test_core.py"""
import asyncio, calendar, json, os, re, sys, time, types, unittest
from unittest.mock import patch

LANGS = ["tr", "en", "de", "fr", "ru", "ar", "es"]


class Ctl:
    @classmethod
    def reset(c):
        c.tr = c.an = c.rss = 0
        c.tr_delay = c.an_delay = c.rss_delay = 0.0
        c.fail = set()
        c.posts = []
        c.tr_json = None
        c.urls = []
        c.post_body = {}
        c.analysis = json.dumps({"summary": "özet", "common": ["a"], "conflicts": [], "framing": [], "unverified": [],
                                 "confidence": {"level": "Orta", "reason": "r"}})
        c.feeds = {l: [{"title": f"H{l}{i} - S{l}{i}", "source": {"title": f"S{l}{i}"}, "link": "https://x/", "published_parsed": None}
                       for i in range(8)] for l in LANGS}


class Blk:
    def __init__(s, t): s.type, s.text = "text", t


class Msgs:
    async def create(self, model, max_tokens, system, messages):
        if "haiku" in model:
            Ctl.tr += 1; await asyncio.sleep(Ctl.tr_delay)
            return types.SimpleNamespace(content=[Blk(json.dumps({l: f"q-{l}" for l in LANGS}))])
        Ctl.an += 1; await asyncio.sleep(Ctl.an_delay)
        return types.SimpleNamespace(content=[Blk(Ctl.analysis)])


class Resp:
    def __init__(s, c): s.content = c
    def raise_for_status(s): pass


class Client:
    def __init__(s, **k): s.messages = Msgs()
    async def post(s, url, headers=None, json=None):
        Ctl.posts.append((url, json))
        return types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: Ctl.post_body)

    async def get(s, url):
        Ctl.rss += 1; Ctl.urls.append(url); await asyncio.sleep(Ctl.rss_delay)
        lang = re.search(r"ceid=\w+:(\w+)", url).group(1)
        if lang in Ctl.fail: raise RuntimeError("boom")
        return Resp(json.dumps(Ctl.feeds[lang]).encode())


class Ent(dict):
    def __getattr__(s, k):
        try: return s[k]
        except KeyError: raise AttributeError(k)


def fp_parse(c):
    es = [Ent(e) for e in json.loads(c)]
    for e in es:
        if e.get("published_parsed"): e["published_parsed"] = tuple(e["published_parsed"])
    return types.SimpleNamespace(entries=es)


_ORIG_MODULES = {}
_STUB_OBJS = {}
for _name, _attrs in {"httpx": {"AsyncClient": Client, "Timeout": lambda *a, **k: None},
                      "feedparser": {"parse": fp_parse}}.items():
    _ORIG_MODULES[_name] = sys.modules.get(_name)
    _m = types.ModuleType(_name)
    _m.__dict__.update(_attrs)
    _STUB_OBJS[_name] = _m
    sys.modules[_name] = _m
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import core
finally:
    for _name, _orig in _ORIG_MODULES.items():
        if _orig is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _orig


async def fake_ask(kind, system, user, max_tokens):
    if kind == "fast":
        Ctl.tr += 1; await asyncio.sleep(Ctl.tr_delay)
        return Ctl.tr_json or json.dumps({l: f"q-{l}" for l in LANGS})
    Ctl.an += 1; await asyncio.sleep(Ctl.an_delay)
    return Ctl.analysis


core.ask = fake_ask
REAL_HOST_PUBLIC = core._host_public
FAKE_HTTP = Client()
FAKE_LLM_HTTP = Client()
FAKE_FEEDPARSER = _STUB_OBJS["feedparser"]


def setUpModule():
    core.ask = fake_ask
    core.http = FAKE_HTTP
    core.llm_http = FAKE_LLM_HTTP
    core.feedparser = FAKE_FEEDPARSER


class T(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        Ctl.reset(); core._decode_sync = lambda link: None; core.news_memo = core.Memo(900); core.analysis_memo = core.Memo(900)
        core._host_public = lambda h: True  # sahte yayıncı adları çözülmez; gerçek kontrol ayrı testte

    async def test_shape_cleaning_per_lang_cap(self):
        Ctl.feeds["en"][0]["published_parsed"] = [2026, 9, 28, 10, 0, 0, 0, 271, 0]
        items = await core.get_news("Test")
        self.assertEqual(len(items), 7 * core.PER_LANG)
        self.assertTrue(all(" - S" not in i["title"] for i in items))
        self.assertEqual([i["date"] for i in items if i["lang"] == "en"][0], "28.09")
        self.assertEqual([i["ts"] for i in items if i["lang"] == "en"][0], calendar.timegm((2026, 9, 28, 10, 0, 0, 0, 271, 0)))

    async def test_dedup_same_source_only(self):
        e = lambda t, s: {"title": f"{t} - {s}", "source": {"title": s}, "link": "", "published_parsed": None}
        Ctl.feeds = {l: [] for l in LANGS}
        Ctl.feeds["tr"] = [e("Aynı", "A"), e("Aynı", "A"), e("Aynı", "B"), e("Diğer", "C")]
        self.assertEqual(len(await core.get_news("x y")), 3)

    async def test_partial_failure_ok(self):
        Ctl.fail = {"ru", "ar"}
        items = await core.get_news("Test")
        self.assertEqual({i["lang"] for i in items}, {"tr", "en", "de", "fr", "es"})

    async def test_total_failure_not_cached(self):
        Ctl.fail = set(LANGS)
        with self.assertRaises(core.NewsError) as c: await core.get_news("Test")
        self.assertEqual(c.exception.status, 404)
        Ctl.fail = set()
        self.assertTrue(await core.get_news("Test"))

    async def test_cache_and_coalescing(self):
        Ctl.tr_delay = Ctl.an_delay = 0.05
        qs = ["Test  Konu", "test konu", "İSTANBUL", "istanbul"] * 5
        res = await asyncio.gather(*(core.get_analysis("İstanbul") for _ in range(5)))
        res2 = await asyncio.gather(*(core.get_analysis("istanbul") for _ in range(5)))
        self.assertEqual((Ctl.tr, Ctl.an, Ctl.rss), (1, 1, 2 * len(core.LANGS)))
        self.assertTrue(all(r is res[0] for r in res + res2))

    async def test_bad_json_then_recovers(self):
        good = Ctl.analysis; Ctl.analysis = "json değil"
        with self.assertRaises(core.NewsError) as c: await core.get_analysis("Test")
        self.assertEqual(c.exception.status, 502)
        Ctl.analysis = "Tabii:\n```json\n" + good + "\n```"
        self.assertEqual((await core.get_analysis("Test"))["summary"], "özet")
        self.assertEqual(Ctl.an, 2)

    def test_normalize(self):
        a = core.normalize({"summary": "s", "confidence": {"level": "Çok Yüksek"}, "common": "x",
                            "conflicts": [{"claim": "c", "sides": [{"who": "w", "says": "z"}, "kötü"]}, "kötü", {"sides": 3}]})
        self.assertEqual(a["confidence"]["level"], "Düşük")
        self.assertEqual((a["common"], len(a["conflicts"]), len(a["conflicts"][0]["sides"])), ([], 1, 1))
        with self.assertRaises(ValueError): core.normalize([])
        n = core.normalize({"timeline": {"x": 1}, "by_country": [{"who": "TR"}, 5]})
        self.assertEqual((n["timeline"], n["by_country"], n["missing"]), ([], [{"who": "TR", "says": ""}], []))

    async def test_memo_eviction_and_ttl(self):
        m = core.Memo(900, maxsize=3)
        for k in "abcde": await m.get(k, lambda k=k: asyncio.sleep(0, k))
        self.assertEqual(list(m.data), ["c", "d", "e"])
        n, m = [0], core.Memo(0.05)
        async def f(): n[0] += 1; return n[0]
        await m.get("k", f); await asyncio.sleep(0.06); await m.get("k", f)
        self.assertEqual(n[0], 2)

    async def test_gemini_and_ollama_requests(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            with self.assertRaises(core.NewsError) as cm:
                await core._gemini("gemini-3.5-flash-lite", "sys", "usr", 100)
            self.assertEqual(cm.exception.status, 503)
        os.environ["GEMINI_API_KEY"] = "k"
        Ctl.post_body = {"candidates": [{"content": {"parts": [{"text": '{"a":1}'}]}}]}
        self.assertEqual(await core._gemini("gemini-3.5-flash-lite", "sys", "usr", 100), '{"a":1}')
        url, body = Ctl.posts[-1]
        self.assertTrue(url.endswith("gemini-3.5-flash-lite:generateContent"))
        self.assertEqual(body["generationConfig"]["maxOutputTokens"], 200)
        Ctl.post_body = {"message": {"content": "{}"}}
        self.assertEqual(await core._ollama("qwen2.5:7b", "sys", "usr", 100), "{}")
        url, body = Ctl.posts[-1]
        self.assertTrue(url.endswith("/api/chat"))
        self.assertEqual((body["format"], body["stream"]), ("json", False))

    async def test_read_articles_partial(self):
        class TH:
            async def get(s, url, **kw):
                if "bad" in url: raise RuntimeError("engel")
                return types.SimpleNamespace(raise_for_status=lambda: None, headers={"content-type": "TEXT/HTML; charset=utf-8"}, text='<meta property="og:image" content="https://img.example/a.jpg?x=1&amp;y=2"><p>x</p>')
        core.text_http, core._extract = TH(), lambda html: "A" * 3000
        core._decode_sync = lambda link: link.replace("gn://", "https://pub/")
        items = [{"lang": "tr", "source": f"S{i}", "title": "t", "link": f"gn://{'bad' if i == 1 else 'ok'}{i}", "n": i + 1}
                 for i in range(4)]
        got, imgs, st = await core.read_articles(items)
        self.assertEqual(sorted(got), [1, 3, 4])
        self.assertEqual((st["tried"], st["read"], st["why"]), (4, 3, {"ok": 3, "fetch": 1}))
        self.assertEqual(set(imgs.values()), {"https://img.example/a.jpg?x=1&y=2"})
        self.assertIsNone(core._og_image('<meta content="http://a/b.jpg" property="og:image">', "https://x"))
        self.assertEqual(core._og_image('<meta property="og:image" content="/i.png">', "https://x.com/a/b"), "https://x.com/i.png")
        self.assertTrue(all(len(v) == core.EXCERPT_CHARS for v in got.values()))
        core._decode_sync = lambda link: None
        an = await core.get_analysis("Metin testi")  # decode None -> yalnızca başlık, yine de çalışır
        self.assertEqual((an["read"], an["images"]), ([], {}))
        self.assertEqual(an["stats"]["read"], 0)
        self.assertIn("decode", an["stats"]["why"])

    def test_pick_for_reading_diverse(self):
        items = [{"lang": l, "source": s, "n": k} for k, (l, s) in enumerate(
            [("tr", "A"), ("tr", "A"), ("tr", "B"), ("en", "A"), ("en", "C")], 1)]
        got = core.pick_for_reading(items, n=4)
        self.assertEqual(len({i["source"] for i in got}), len(got))  # aynı yayıncı tekrarlanmaz
        self.assertEqual({i["lang"] for i in got}, {"tr", "en"})

    async def test_time_window(self):
        n = len(core.LANGS)
        await core.get_news("Pencere", "7d")           # güncel (when:7d) + arka plan (when yok)
        self.assertEqual(len(Ctl.urls), 2 * n)
        self.assertTrue(all("when%3A7d" in u for u in Ctl.urls[:n]))
        self.assertFalse(any("when" in u for u in Ctl.urls[n:]))
        await core.get_news("Pencere", "7d")           # önbellek
        self.assertEqual(Ctl.rss, 2 * n)
        await core.get_news("Pencere", "")             # tüm zamanlar: arka plan yok
        self.assertEqual(Ctl.rss, 3 * n)

    async def test_background_spread(self):
        now = time.time()
        mk = lambda i, ts: {"title": f"E{i} - S{i}", "source": {"title": f"S{i}"}, "link": "", "published_parsed": list(time.gmtime(ts)[:9])}
        Ctl.feeds["tr"] = [mk(i, now - 86400 * (100 + 30 * (20 - i))) for i in range(20)] + [mk(50 + i, now - 3600 * i) for i in range(3)]
        res = await core._background({"tr": "x"}, "30d", set())
        self.assertEqual(len(res), core.BG_KEEP)
        self.assertTrue(all(r["bg"] and r["ts"] < now - 30 * 86400 for r in res))   # güncel olanlar girmez
        self.assertEqual([r["ts"] for r in res], sorted(r["ts"] for r in res))      # eskiden yeniye
        self.assertEqual((res[0]["title"], res[-1]["title"]), ("E0", "E19"))        # uçlar korunur

    def test_timeline_normalize(self):
        tl = core.normalize({"timeline": [{"when": "x", "iso": "bozuk", "what": "y"},
                                          {"when": "a", "iso": "2026-10-12", "what": "b"}, 7]})["timeline"]
        self.assertEqual([e["iso"] for e in tl], ["", "2026-10-12"])

    async def test_vague_query_asks_for_clarification(self):
        Ctl.tr_json = json.dumps({"vague": True, "why": "Çok genel.", "ideas": ["a", "b"]})
        with self.assertRaises(core.NewsError) as c: await core.get_news("nedir")
        self.assertEqual((c.exception.status, c.exception.extra, Ctl.rss), (422, ["a", "b"], 0))

    async def test_ideas_from_top_stories(self):
        Ctl.analysis = json.dumps({"ideas": [{"question": "Neden X?", "hook": "h", "query": "x y"}, {"question": "eksik"}]})
        core.ideas_memo = core.Memo(900)
        self.assertEqual(await core.get_ideas(), [{"question": "Neden X?", "hook": "h", "query": "x y"}])
        await core.get_ideas()
        self.assertEqual(Ctl.an, 1)

    def test_rate_limiter(self):
        t = [0.0]; r = core.RateLimiter(12, 60, clock=lambda: t[0])
        self.assertEqual([r.allow("ip") for _ in range(13)], [True] * 12 + [False])
        self.assertTrue(r.allow("other"))
        t[0] = 61; self.assertTrue(r.allow("ip"))

    def test_host_public_ssrf_checks(self):
        orig_getaddrinfo = core.socket.getaddrinfo
        fake_dns = {
            "127.0.0.1": [(2, 1, 6, "", ("127.0.0.1", 443))],
            "10.0.0.1": [(2, 1, 6, "", ("10.0.0.1", 443))],
            "172.16.0.1": [(2, 1, 6, "", ("172.16.0.1", 443))],
            "192.168.1.1": [(2, 1, 6, "", ("192.168.1.1", 443))],
            "169.254.169.254": [(2, 1, 6, "", ("169.254.169.254", 443))],
            "::1": [(23, 1, 6, "", ("::1", 443, 0, 0))],
            "mixed.example": [(2, 1, 6, "", ("93.184.216.34", 443)), (2, 1, 6, "", ("10.0.0.2", 443))],
            "public.example": [(2, 1, 6, "", ("93.184.216.34", 443))],
        }
        def stub_dns(host, port, *a, **k):
            if host in fake_dns:
                return fake_dns[host]
            raise OSError("dns fail")
        core.socket.getaddrinfo = stub_dns
        try:
            for bad in ("127.0.0.1", "10.0.0.1", "172.16.0.1", "192.168.1.1", "169.254.169.254", "::1", "mixed.example", "missing.invalid"):
                self.assertFalse(REAL_HOST_PUBLIC(bad), bad)
            self.assertTrue(REAL_HOST_PUBLIC("public.example"))
        finally:
            core.socket.getaddrinfo = orig_getaddrinfo

    async def test_safe_get_ssrf_and_redirects(self):
        orig_getaddrinfo = core.socket.getaddrinfo
        orig_text_http = core.text_http
        fake_dns = {
            "127.0.0.1": [(2, 1, 6, "", ("127.0.0.1", 443))],
            "10.0.0.5": [(2, 1, 6, "", ("10.0.0.5", 443))],
            "192.168.1.10": [(2, 1, 6, "", ("192.168.1.10", 443))],
            "pub.example": [(2, 1, 6, "", ("93.184.216.34", 443))],
            "pub2.example": [(2, 1, 6, "", ("93.184.216.35", 443))],
        }
        calls = []
        class SafeTH:
            async def get(self, url, **kw):
                calls.append((url, kw.get("follow_redirects")))
                if url == "https://pub.example/ok":
                    return types.SimpleNamespace(is_redirect=False, status_code=200, headers={"content-type": "text/html"}, text="ok")
                if url == "https://pub.example/redir-ok":
                    return types.SimpleNamespace(is_redirect=True, status_code=302, headers={"location": "https://pub2.example/final"})
                if url == "https://pub2.example/final":
                    return types.SimpleNamespace(is_redirect=False, status_code=200, headers={"content-type": "text/html"}, text="done")
                if url == "https://pub.example/redir-private":
                    return types.SimpleNamespace(is_redirect=True, status_code=302, headers={"location": "https://127.0.0.1/secret"})
                if url == "https://pub.example/redir-http":
                    return types.SimpleNamespace(is_redirect=True, status_code=302, headers={"location": "http://pub2.example/downgrade"})
                if url.startswith("https://pub.example/loop"):
                    n = int(url[-1])
                    return types.SimpleNamespace(is_redirect=True, status_code=302, headers={"location": f"https://pub.example/loop{n + 1}"})
                raise AssertionError(f"unexpected url: {url}")
        core.socket.getaddrinfo = lambda host, port, *a, **k: fake_dns[host] if host in fake_dns else (_ for _ in ()).throw(OSError("dns"))
        core._host_public = REAL_HOST_PUBLIC
        core.text_http = SafeTH()
        try:
            for blocked in (
                "http://pub.example/insecure",
                "file:///etc/passwd",
                "javascript:alert(1)",
                "https://127.0.0.1/secret",
                "https://10.0.0.5/internal",
                "https://192.168.1.10/lan",
                "https://pub.example:8443/custom-port",
                "https://user:pass@pub.example/ok",
            ):
                self.assertIsNone(await core._safe_get(blocked), blocked)
            self.assertEqual(calls, [])

            r = await core._safe_get("https://pub.example/ok")
            self.assertIsNotNone(r)
            self.assertEqual(r.text, "ok")

            r2 = await core._safe_get("https://pub.example/redir-ok")
            self.assertIsNotNone(r2)
            self.assertEqual(r2.text, "done")

            self.assertIsNone(await core._safe_get("https://pub.example/redir-private"))
            self.assertIsNone(await core._safe_get("https://pub.example/redir-http"))
            self.assertIsNone(await core._safe_get("https://pub.example/loop0"))
            self.assertTrue(all(fr is False for _, fr in calls))
        finally:
            core.socket.getaddrinfo = orig_getaddrinfo
            core.text_http = orig_text_http

    async def test_timing_parallel_and_early_news(self):
        Ctl.tr_delay, Ctl.rss_delay, Ctl.an_delay = 0.05, 0.1, 0.5  # sahte gecikmeler
        t0 = time.perf_counter()
        an = asyncio.ensure_future(core.get_analysis("Zaman"))
        await core.get_news("Zaman"); t_news = time.perf_counter() - t0
        await an; t_all = time.perf_counter() - t0
        print(f"\n[simüle] haberler: {t_news:.2f}s, analiz dahil toplam: {t_all:.2f}s (7 RSS sıralı olsaydı >= 0.75s)")
        self.assertLess(t_news, 1.1)    # 11 sürüm x 2 tur x 0.1s sıralı olsaydı ~2.2s olurdu
        self.assertLess(t_news, t_all / 2)


class AppTest(unittest.TestCase):
    def test_app_endpoints_and_security_headers(self):
        from fastapi.testclient import TestClient
        import app as app_mod
        client = TestClient(app_mod.app)
        app_mod.limiter.hits.clear()

        r = client.get("/healthz")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"ok": True})
        self.assertEqual(r.headers.get("x-frame-options"), "DENY")
        self.assertEqual(r.headers.get("x-content-type-options"), "nosniff")

        r_home = client.get("/")
        self.assertEqual(r_home.status_code, 200)
        self.assertIn("Aynı haber, farklı ülkeler", r_home.text)

        with patch.dict(os.environ, {"TRUST_PROXY": "1"}):
            req = types.SimpleNamespace(
                headers={"x-forwarded-for": "1.2.3.4, 9.8.7.6"},
                client=types.SimpleNamespace(host="10.0.0.1"),
            )
            self.assertEqual(app_mod.client_ip(req), "9.8.7.6")

    def test_impact_scoring_filtering_and_export(self):
        from fastapi.testclient import TestClient
        import app as app_mod

        s_acil = core.score_item("SON DAKİKA: Kritik deprem ve patlama alarmı")
        self.assertEqual(s_acil["signal"], "acil")
        self.assertIn("kritik", s_acil["signals"])
        self.assertGreaterEqual(s_acil["impact_score"], 90)

        s_poz = core.score_item("Yeni barış anlaşması ve rekor büyüme yatırımı")
        self.assertIn("pozitif", s_poz["signals"])

        items = [
            {"n": 1, "lang": "tr", "source": "AA", "date": "12.10", "title": "Son dakika deprem uyarısı", "link": "https://a", **core.score_item("Son dakika deprem uyarısı")},
            {"n": 2, "lang": "en", "source": "Reuters", "date": "12.10", "title": "Peace investment growth", "link": "https://b", **core.score_item("Peace investment growth")},
            {"n": 3, "lang": "de", "source": "DW", "date": "12.10", "title": "Normaler Bericht", "link": "https://c", **core.score_item("Normaler Bericht")},
        ]
        self.assertEqual(len(core.filter_items(items, signal="acil")), 1)
        self.assertEqual(len(core.filter_items(items, keywords="reuters, dw")), 2)
        self.assertEqual(len(core.filter_items(items, keywords="deprem", signal="acil")), 1)

        csv_out = core.export_items_csv(items)
        self.assertIn("n,lang,source,date,signal,impact_score,title,link", csv_out)
        self.assertIn("Son dakika deprem uyarısı", csv_out)

        json_out = json.loads(core.export_items_json(items))
        self.assertEqual(json_out["count"], 3)
        self.assertEqual(json_out["items"][0]["signal"], "acil")

        client = TestClient(app_mod.app)
        app_mod.limiter.hits.clear()
        with patch.object(core, "get_news", return_value=items):
            r_csv = client.post("/api/export", json={"query": "deprem", "keywords": "deprem", "signal": "acil", "format": "csv"})
            self.assertEqual(r_csv.status_code, 200)
            self.assertIn("text/csv", r_csv.headers.get("content-type", ""))
            self.assertIn("Son dakika deprem uyarısı", r_csv.text)
            self.assertNotIn("Normaler Bericht", r_csv.text)

            r_json = client.post("/api/export", json={"query": "deprem", "format": "json"})
            self.assertEqual(r_json.status_code, 200)
            self.assertEqual(r_json.json()["count"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=1)
