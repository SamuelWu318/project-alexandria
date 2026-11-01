from __future__ import annotations
import json, re, glob, sys, os, signal, subprocess, time, contextlib
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# make the package modules (utils, search, query) importable no matter how this is launched
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from utils import SrcPaths
from utils import tags              # POV/Tense enums + the tone valence/dominance table (the VD emotion chart)
import search                       # unified read path (the one search() door + the D5 soft constants we override live)
import query                        # read-path front door: summary + soft WORD->coord (arc/emotion words -> tones curve)

# ---- local read-path test server (no LLM) ----
# A tiny stdlib HTTP server driving the LIVE Qdrant store so a browser front end can exercise the whole
# retrieval path. Three tabs: (1) TESTING — replay the gold test_queries over the whole corpus; (2) USER
# QUERY — a writer's request (summary + descriptors + pov/tense + intensity arc + up to 3 valence/dominance
# emotions) -> query.to_query_object -> top-20; (3) TWEAKING — the tuned blend + soft weights, editable live
# (the server swaps search.py's module constants per request — safe because this server is SERIAL).
# Run: `python -m webtest.server`, then open http://localhost:8765/ ; /api/* return JSON.

# ---- config ----

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
GOLD = HERE / "gold"
PORT = 8765

# The four writer-facing pov/tense combinations offered as one dropdown (label -> hard facets).
POV_TENSE = [
    {"label": "3rd · past",    "pov": "third", "tense": "past"},
    {"label": "3rd · present", "pov": "third", "tense": "present"},
    {"label": "1st · past",    "pov": "first", "tense": "past"},
    {"label": "1st · present", "pov": "first", "tense": "present"},
]

# ---- qdrant lock: last launch wins ----
# Local on-disk Qdrant is single-process. This harness is meant to be relaunched freely, so a NEW
# server forcibly evicts whoever holds the lock (an old server, or a stray REPL) then opens. Scoped
# strictly to THIS qdrant dir.

# ** LOCKED **
# PIDs (excluding self) holding this qdrant dir open, or running a sibling server (via lsof/pgrep).
def _lock_holders() -> set[int]:
    dirp = str(SrcPaths.QDRANT_DIR)
    pids: set[int] = set()
    probes = (
        ["lsof", "-t", "--", os.path.join(dirp, ".lock")],  # precise: the lock file
        ["lsof", "-t", "+D", dirp],                          # any handle under the dir
        ["pgrep", "-f", r"webtest[./]server"],               # script path or `-m webtest.server`
    )
    for cmd in probes:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
            pids.update(int(x) for x in out.split() if x.strip().isdigit())
        except Exception:
            pass
    pids.discard(os.getpid())
    return pids


# ** LOCKED **
# SIGTERM the lock holders, wait briefly, then SIGKILL any that survive.
def _evict(pids: set[int]) -> None:
    for p in pids:
        try: os.kill(p, signal.SIGTERM)
        except ProcessLookupError: pass
        except PermissionError: print(f"[webtest] cannot signal pid {p} (permission)")
    time.sleep(0.7)
    for p in pids:
        try: os.kill(p, signal.SIGKILL)
        except (ProcessLookupError, PermissionError): pass


# ** LOCKED **
# Open the store, evicting the current lock holder on contention (last wins); retries a few times.
def _open_qdrant(retries: int = 6) -> QdrantClient:
    last: Exception | None = None
    for i in range(retries):
        try:
            return QdrantClient(path=str(SrcPaths.QDRANT_DIR))
        except RuntimeError as e:
            last = e
            if "already accessed" not in str(e).lower():
                raise
            holders = _lock_holders()                        # who holds the lock?
            if holders:
                print(f"[webtest] qdrant lock held by {sorted(holders)} — evicting")
                _evict(holders)                              # kill them, then retry
            time.sleep(0.4 * (i + 1))   # wait out the RocksDB lock release
    raise RuntimeError(f"could not acquire qdrant lock after {retries} tries: {last}")


# ---- shared, opened once ----
_client = _open_qdrant()

# in-memory scene index: scene_id -> full record (text_html etc.), + book titles, + the distinct
# descriptor vocabulary (for the user-query autofill).
_scenes: dict[str, dict] = {}
_book_title: dict[str, str] = {}
_descriptors: list[str] = []        # distinct scene descriptors, most-common first (autofill source)


# Load every pg*-s.json into memory once (payload source for the reader + the descriptor vocab).
def _load_scenes() -> None:
    freq: dict[str, int] = {}
    for f in sorted(glob.glob(str(SrcPaths.SCENES_DIR / "pg*-s.json"))):
        recs = json.loads(Path(f).read_text(encoding="utf-8"))
        for r in recs:
            sid = r.get("scene_id")
            if sid:
                _scenes[sid] = r
            for d in (r.get("descriptors") or []):
                w = str(d).strip().lower()
                if w:
                    freq[w] = freq.get(w, 0) + 1
        if recs:
            bid = recs[0].get("book_id")
            _book_title[bid] = (recs[0].get("book_metadata") or {}).get("Title") or f"book {bid}"
    _descriptors[:] = sorted(freq, key=lambda w: (-freq[w], w))   # frequency desc, then alpha
    print(f"[webtest] loaded {len(_scenes)} scenes / {len(_book_title)} books / {len(_descriptors)} descriptors")


# ** LOCKED **
# text_html -> plain text, whitespace collapsed.
def _strip(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html or "")).strip()


# First ~`words` words of the scene prose, for the result-card teaser.
def _preview(rec: dict, words: int = 50) -> str:
    toks = _strip(rec.get("text_html", "")).split()
    return " ".join(toks[:words]) + ("…" if len(toks) > words else "")


# ** LOCKED **
# Dense per-book rank = the integer suffix of the scene_id, or None.
def _pos(scene_id: str):
    try:
        return int(str(scene_id).rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return None


# Lightweight scene card for a result list (payload merged over the on-disk record for text_html).
def _card(payload: dict, score: float | None = None) -> dict:
    sid = payload.get("scene_id")
    rec = {**_scenes.get(sid, {}), **payload}
    return {
        "scene_id": sid,
        "book_id": rec.get("book_id"),
        "book_title": _book_title.get(rec.get("book_id"), ""),
        "pos": rec.get("pos") if "pos" in rec else _pos(sid or ""),
        "score": round(score, 4) if score is not None else None,
        "chapter_title": rec.get("chapter_title"),
        "summary": rec.get("summary"),
        "moments": [m.get("sentence") for m in (rec.get("moments") or []) if isinstance(m, dict)],
        "descriptors": rec.get("descriptors"),
        "pov": rec.get("pov"),
        "tense": rec.get("tense"),
        "prose_register": rec.get("prose_register"),
        "dialogue_ratio": rec.get("dialogue_ratio"),
        "arc": rec.get("arc"),
        "word_count": rec.get("word_count"),
        "preview": _preview(rec),
    }


# ---- tuning: the tweaking tab's live weights ----
# The blend knobs (method_weights, combine, normalize, rrf_k) are search() kwargs. The soft knobs (LAMBDA +
# the soft-axis / affect-axis mixes) are search.py MODULE CONSTANTS (D5 — fixed in code). The tweaking tab
# ships full overrides; since this server is single-threaded we swap those globals around one search() call
# and restore them after, so the code default is untouched between requests.

# search.py soft constants overridden per request, grouped by the tweaking-tab control that sets them.
_SOFT_CONSTS = ("LAMBDA", "W_TONE", "W_PROSE", "W_DIA", "CURVE_WV", "CURVE_WD", "CURVE_WI")


# The live tuned defaults, read straight from the search module so this UI mirrors the real code constants.
def _tuning_defaults() -> dict:
    return {
        "method_weights": dict(search.DEFAULT_METHOD_WEIGHTS),   # scenes vs flavor RRF split (ratio)
        "combine": "sum", "normalize": "zscore", "rrf_k": 60,    # search() defaults
        "lambda": search.LAMBDA,                                 # global soft strength
        "soft_mix": {"tone": search.W_TONE, "prose": search.W_PROSE, "dialogue": search.W_DIA},
        "affect_mix": {"v": search.CURVE_WV, "d": search.CURVE_WD, "i": search.CURVE_WI},
    }


# Swap search.py's soft module constants to the tweaking tab's values for one call, then restore.
@contextlib.contextmanager
def _soft_overrides(t: dict):
    saved = {k: getattr(search, k) for k in _SOFT_CONSTS}
    try:
        if t.get("lambda") is not None:
            search.LAMBDA = float(t["lambda"])
        sm = t.get("soft_mix") or {}
        if sm:
            search.W_TONE, search.W_PROSE, search.W_DIA = float(sm["tone"]), float(sm["prose"]), float(sm["dialogue"])
        am = t.get("affect_mix") or {}
        if am:
            search.CURVE_WV, search.CURVE_WD, search.CURVE_WI = float(am["v"]), float(am["d"]), float(am["i"])
        yield
    finally:
        for k, v in saved.items():
            setattr(search, k, v)


# Coerce a UI normalize string to search's contract ("none"/"raw"/"" -> None; else pass through).
def _norm(v):
    if isinstance(v, str) and v.strip().lower() in ("none", "null", "raw", ""):
        return None
    return v


# The blend knobs from a tuning dict as search() kwargs (method_weights / combine / normalize / rrf_k).
def _blend_kwargs(t: dict) -> dict:
    kw = {"combine": t.get("combine") or "sum",
          "normalize": _norm(t.get("normalize", "zscore")),
          "rrf_k": int(t.get("rrf_k") or 60)}
    mw = t.get("method_weights")
    if mw:
        kw["method_weights"] = {"scenes": float(mw.get("scenes", 0)), "flavor": float(mw.get("flavor", 0))}
    return kw


# ---- search dispatch ----

# Run ONE gold query object over the corpus -> a result column. Never raises: errors ride back in-band.
def _run_gold(q: dict, limit: int, tuning: dict) -> dict:
    try:
        pts = search.search(
            _client,
            summary=q.get("summary") or None,
            moments=q.get("moments") or None,        # gold clause sentence(s) -> svos channel, when present
            descriptors=q.get("descriptors") or None,
            limit=limit, **_blend_kwargs(tuning),
        )
        results = [_card(p.payload, p.score) for p in pts]
        return {"label": q.get("label", ""), "target_book_id": q.get("target_book_id"),
                "meta": q.get("meta", {}), "results": results}
    except Exception as e:  # empty query, bad weights, etc. -> show it, don't 500 the batch
        return {"label": q.get("label", ""), "target_book_id": q.get("target_book_id"),
                "meta": q.get("meta", {}), "error": f"{type(e).__name__}: {e}", "results": []}


# Run a writer's request: summary + descriptors + hard pov/tense + intensity arc + ordered VD emotion words.
# The soft WORDS (arc -> intensity axis, emotions -> valence/dominance) become the numeric `tones` curve in
# query.to_query_object; the tweaking tab's soft weights are applied around the call.
def _run_user_query(body: dict) -> dict:
    tuning = body.get("tuning") or {}
    summary = body.get("summary") or ""
    descriptors = [d for d in (body.get("descriptors") or []) if str(d).strip()]
    pov = body.get("pov") or None
    tense = body.get("tense") or None
    arc = body.get("arc") or None
    emotions = [e for e in (body.get("emotions") or []) if e]   # ordered tone WORDS from the VD chart
    limit = int(body.get("limit", 20))
    try:
        obj = query.to_query_object(summary, pov=pov, tense=tense, arc=arc,   # arc + emotions -> tones curve
                                    vd=emotions or None, descriptors=descriptors or None)
    except ValueError as e:
        return {"error": str(e), "results": []}
    if not obj:
        return {"error": "empty query — add a summary, descriptors, a pov/tense, an arc, or an emotion",
                "results": []}
    try:
        with _soft_overrides(tuning):
            pts = search.search(_client, limit=limit, **obj, **_blend_kwargs(tuning))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "results": []}
    return {"results": [_card(p.payload, p.score) for p in pts],
            "query_object": {k: v for k, v in obj.items()}}


# ---- request handling ----

class Handler(BaseHTTPRequestHandler):
    # Silence the default per-request access logging.
    def log_message(self, *a):  # quiet
        pass

    # Write one HTTP response (status, body bytes, content type).
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # Send an object as a JSON response.
    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    # Send a file from disk, or a 404 JSON if absent.
    def _file(self, path: Path, ctype: str):
        if not path.is_file():
            return self._json({"error": "not found"}, 404)
        self._send(200, path.read_bytes(), ctype)

    # Route GET: index.html, /api/config (everything the UI needs), /api/scene (reader), static assets.
    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        if p in ("/", "/index.html"):
            return self._file(STATIC / "index.html", "text/html; charset=utf-8")
        if p == "/api/config":
            tq = json.loads((GOLD / "test_queries.json").read_text())["test_queries"]
            # live vocab (source of truth = the code): pov/tense enums, the 6 intensity ARCS (query.py),
            # the tone valence/dominance table (the emotion chart), the distinct descriptor autofill list,
            # and the tuned weights read straight off the search module.
            emotions = [{"word": t.value, "v": v, "d": d} for t, (v, d) in tags.TONE_VD.items()]
            return self._json({
                "books": [{"book_id": b, "title": t} for b, t in sorted(_book_title.items())],
                "test_queries": tq,
                "pov": [t.value for t in tags.POV], "tense": [t.value for t in tags.Tense],
                "pov_tense": POV_TENSE,
                "arcs": list(query.INTENSITY_ARCS.keys()),
                "emotions": emotions,
                "descriptors": _descriptors,
                "tuning": _tuning_defaults(),
            })
        if p == "/api/scene":
            sid = (parse_qs(u.query).get("id") or [""])[0]
            rec = _scenes.get(sid)
            if not rec:
                return self._json({"error": f"scene {sid!r} not found"}, 404)
            d = _card(rec, None)
            d.update({
                "text_html": rec.get("text_html", ""),
                "vdi_curve": rec.get("vdi_curve"),
                "prev_scene_id": rec.get("prev_scene_id"),
                "next_scene_id": rec.get("next_scene_id"),
            })
            return self._json(d)
        if p.startswith("/static/"):
            return self._file(STATIC / p[len("/static/"):], _ctype(p))
        return self._json({"error": f"no route {p}"}, 404)

    # Route POST: /api/search_batch (testing tab, gold replay), /api/user_query (writer request -> top 20).
    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if u.path == "/api/search_batch":
            limit = int(body.get("limit", 8))
            queries = body.get("queries", [])
            tuning = body.get("tuning") or {}
            with _soft_overrides(tuning):                          # gold columns share one live-tuned pass
                cols = [_run_gold(q, limit, tuning) for q in queries]
            return self._json({"columns": cols})
        if u.path == "/api/user_query":
            return self._json(_run_user_query(body))
        return self._json({"error": f"no route {u.path}"}, 404)


# ** LOCKED **
# Map a static file path to its Content-Type (css/js/html, else octet-stream).
def _ctype(path: str) -> str:
    if path.endswith(".css"): return "text/css"
    if path.endswith(".js"): return "application/javascript"
    if path.endswith(".html"): return "text/html; charset=utf-8"
    return "application/octet-stream"


# ---- server ----

# ** ENTRY ** — evict an older sibling, load scenes, then serve on PORT until Ctrl-C.
def main():
    srv = None
    try:
        # Do this even when opening the current Qdrant directory did not contend: an older server may
        # still own PORT after its database directory was moved or replaced underneath it.
        holders = _lock_holders()
        if holders:
            print(f"[webtest] older server held by {sorted(holders)} — evicting")
            _evict(holders)
        _load_scenes()                   # in-memory payload source + descriptor vocab
        srv = HTTPServer(("127.0.0.1", PORT), Handler)
        print(f"[webtest] serving on http://localhost:{PORT}/  (Ctrl-C to stop)")
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if srv is not None:
            srv.server_close()
        _client.close()


if __name__ == "__main__":
    main()
