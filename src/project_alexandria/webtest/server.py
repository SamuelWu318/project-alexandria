# FOR CLAUDE — Local read-path test server (no LLM).
# -----------------------------------------------------------------------------
# A tiny stdlib HTTP server (zero new deps) that drives the LIVE stores so the
# gold queries + a browser front end can exercise the whole retrieval path:
#   * Qdrant  (search.py)      -> vector search: fused / summary / combined / descriptors
#   * SQLite  (relational.py)  -> reading-order navigation for the book "wheel"
#   * scenes json (in memory)  -> full text_html + previews (the payload source of truth)
#
# It is deliberately SERIAL (HTTPServer, not ThreadingHTTPServer): the SQLite mirror is
# opened once and sqlite objects are single-thread; one local user clicking does not need
# concurrency, and serial keeps the Qdrant client + connection access trivially safe.
#
# Run:  python -m webtest.server        (from src/project_alexandria/, venv active)
# Then open http://localhost:8765/ .  Endpoints under /api/* return JSON.
# -----------------------------------------------------------------------------
from __future__ import annotations
import json, re, glob, sys, os, signal, subprocess, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# make the package modules (utils, search) importable no matter how this is launched
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from utils import SrcPaths
from utils import relational
import search

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
GOLD = HERE / "gold"
PORT = 8765

# --- qdrant lock: last launch wins --- #
# Local on-disk Qdrant is single-process — a second instance normally errors with
# "already accessed by another instance". This harness is meant to be relaunched
# freely, so a NEW server forcibly evicts whoever holds the lock (an old webtest
# server, or a stray python REPL) and then opens. Scoped strictly to THIS qdrant dir.

def _lock_holders() -> set[int]:
    """PIDs (excluding self) holding this qdrant dir open, or running a sibling server."""
    dirp = str(SrcPaths.QDRANT_DIR)
    pids: set[int] = set()
    probes = (
        ["lsof", "-t", "--", os.path.join(dirp, ".lock")],  # precise: the lock file
        ["lsof", "-t", "+D", dirp],                          # any handle under the dir
        ["pgrep", "-f", "webtest/server.py"],                # sibling servers by cmdline
    )
    for cmd in probes:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
            pids.update(int(x) for x in out.split() if x.strip().isdigit())
        except Exception:
            pass
    pids.discard(os.getpid())
    return pids


def _evict(pids: set[int]) -> None:
    """SIGTERM the holders, give them a moment, then SIGKILL any that survive."""
    for p in pids:
        try: os.kill(p, signal.SIGTERM)
        except ProcessLookupError: pass
        except PermissionError: print(f"[webtest] cannot signal pid {p} (permission)")
    time.sleep(0.7)
    for p in pids:
        try: os.kill(p, signal.SIGKILL)
        except (ProcessLookupError, PermissionError): pass


def _open_qdrant(retries: int = 6) -> QdrantClient:
    """Open the store, evicting the current lock holder on contention (last wins)."""
    last: Exception | None = None
    for i in range(retries):
        try:
            return QdrantClient(path=str(SrcPaths.QDRANT_DIR))
        except RuntimeError as e:
            last = e
            if "already accessed" not in str(e).lower():
                raise
            holders = _lock_holders()
            if holders:
                print(f"[webtest] qdrant lock held by {sorted(holders)} — evicting")
                _evict(holders)
            time.sleep(0.4 * (i + 1))   # wait out the RocksDB lock release
    raise RuntimeError(f"could not acquire qdrant lock after {retries} tries: {last}")


# --- shared, opened once --- #
_client = _open_qdrant()
_conn = relational.open_db(SrcPaths.DB_PATH)

# in-memory scene index: scene_id -> full record (text_html etc.), + book titles
_scenes: dict[str, dict] = {}
_book_title: dict[str, str] = {}


def _load_scenes() -> None:
    """Load every pg*-s.json into memory once (payload source for text + previews)."""
    for f in sorted(glob.glob(str(SrcPaths.SCENES_DIR / "pg*-s.json"))):
        recs = json.loads(Path(f).read_text(encoding="utf-8"))
        for r in recs:
            sid = r.get("scene_id")
            if sid:
                _scenes[sid] = r
        if recs:
            bid = recs[0].get("book_id")
            _book_title[bid] = (recs[0].get("book_metadata") or {}).get("Title") or f"book {bid}"
    print(f"[webtest] loaded {len(_scenes)} scenes across {len(_book_title)} books")


def _strip(html: str) -> str:
    """text_html -> plain text, whitespace collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html or "")).strip()


def _preview(rec: dict, words: int = 45) -> str:
    """First ~`words` words of the scene prose, for the result-card teaser."""
    toks = _strip(rec.get("text_html", "")).split()
    return " ".join(toks[:words]) + ("…" if len(toks) > words else "")


def _card(rec: dict, score: float | None = None) -> dict:
    """Lightweight scene card for a result column / book list (no full text)."""
    return {
        "scene_id": rec.get("scene_id"),
        "book_id": rec.get("book_id"),
        "book_title": _book_title.get(rec.get("book_id"), ""),
        "pos": rec.get("pos") if "pos" in rec else _pos(rec.get("scene_id", "")),
        "score": round(score, 4) if score is not None else None,
        "scene_title": rec.get("scene_title"),
        "chapter_title": rec.get("chapter_title"),
        "summary": rec.get("summary"),
        "dominant_tone": rec.get("dominant_tone"),
        "intensity": rec.get("intensity"),
        "arc": rec.get("arc"),
        "word_count": rec.get("word_count"),
        "preview": _preview(rec),
    }


def _pos(scene_id: str):
    try:
        return int(str(scene_id).rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return None


# --- search dispatch --- #

def _run_query(q: dict, limit: int, book_id: str | None) -> dict:
    """Run ONE query object -> a result column. Never raises: errors ride back in-band."""
    mode = q.get("mode", "fused")
    flt = search.book_filter(book_id) if book_id else None
    # anti-descriptors need matching weights; fill equal if the query gave only the list
    anti = q.get("anti_descriptors")
    aw = q.get("anti_weights")
    if anti and not aw:
        aw = [1.0 / len(anti)] * len(anti)
    try:
        if mode == "summary":
            pts = search.search_summary(_client, q.get("summary", ""), limit=limit, flt=flt)
        elif mode == "combined":
            pts = search.search_combined(
                _client, q.get("summary", ""), q.get("descriptors") or [],
                q.get("weights"), anti_descriptors=anti, anti_weights=aw,
                anti_strength=q.get("anti_strength", 1.0), limit=limit, flt=flt)
        elif mode == "descriptors":
            pts = search.search_weighted_descriptors(
                _client, q.get("descriptors") or [], q.get("weights"),
                anti_descriptors=anti, anti_weights=aw,
                anti_strength=q.get("anti_strength", 1.0), limit=limit, flt=flt)
        else:  # fused (default)
            frame = {k: q.get(k) for k in ("summary", "subject", "verb", "object", "setting")}
            frame["descriptors"] = q.get("descriptors") or []
            # optional per-field weight override from the UI sliders (percentages ->
            # search_fused renormalizes to 1.0 over the fields actually present in the frame)
            fw = q.get("field_weights") or None
            pts = search.search_fused(_client, frame, weights=fw, limit=limit, flt=flt)
        results = [_card(p.payload, p.score) for p in pts]
        return {"label": q.get("label", ""), "mode": mode, "meta": q.get("meta", {}),
                "target_book_id": q.get("target_book_id"), "results": results}
    except Exception as e:  # bad weights, empty frame, etc. -> show it, don't 500 the batch
        return {"label": q.get("label", ""), "mode": mode, "meta": q.get("meta", {}),
                "error": f"{type(e).__name__}: {e}", "results": []}


# --- request handling --- #

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def _file(self, path: Path, ctype: str):
        if not path.is_file():
            return self._json({"error": "not found"}, 404)
        self._send(200, path.read_bytes(), ctype)

    # ---- GET ----
    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        if p in ("/", "/index.html"):
            return self._file(STATIC / "index.html", "text/html; charset=utf-8")
        if p == "/api/datasets":
            tq = json.loads((GOLD / "test_queries.json").read_text())["test_queries"]
            dq = json.loads((GOLD / "descriptor_queries.json").read_text())["descriptor_queries"]
            return self._json({"test_queries": tq, "descriptor_queries": dq,
                               "books": [{"book_id": b, "title": t} for b, t in sorted(_book_title.items())]})
        if p == "/api/scene":
            sid = (parse_qs(u.query).get("id") or [""])[0]
            rec = _scenes.get(sid)
            if not rec:
                return self._json({"error": f"scene {sid!r} not found"}, 404)
            d = _card(rec, None)
            d.update({
                "text_html": rec.get("text_html", ""),
                "descriptors": rec.get("descriptors"),
                "subject": rec.get("subject"), "verb": rec.get("verb"),
                "object": rec.get("object"), "setting": rec.get("setting"),
                "prev_scene_id": rec.get("prev_scene_id"),
                "next_scene_id": rec.get("next_scene_id"),
            })
            return self._json(d)
        if p == "/api/book":
            bid = (parse_qs(u.query).get("id") or [""])[0]
            # reading order comes from the RELATIONAL store (proves the B-tree nav),
            # decorated with previews from the in-memory scene payloads.
            rows = relational.find(_conn, book_id=bid, order=True)
            out = []
            for row in rows:
                rec = _scenes.get(row["scene_id"], {})
                card = _card({**row, **{"text_html": rec.get("text_html", "")}})
                out.append(card)
            return self._json({"book_id": bid, "title": _book_title.get(bid, ""),
                               "count": len(out), "scenes": out})
        # fallback: other static assets
        if p.startswith("/static/"):
            return self._file(STATIC / p[len("/static/"):], _ctype(p))
        return self._json({"error": f"no route {p}"}, 404)

    # ---- POST ----
    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if u.path == "/api/search_batch":
            limit = int(body.get("limit", 8))
            book_id = body.get("book_id") or None
            queries = body.get("queries", [])
            cols = [_run_query(q, limit, book_id) for q in queries]
            return self._json({"columns": cols})
        return self._json({"error": f"no route {u.path}"}, 404)


def _ctype(path: str) -> str:
    if path.endswith(".css"): return "text/css"
    if path.endswith(".js"): return "application/javascript"
    if path.endswith(".html"): return "text/html; charset=utf-8"
    return "application/octet-stream"


def main():
    _load_scenes()
    srv = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[webtest] serving on http://localhost:{PORT}/  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _conn.close(); _client.close()


if __name__ == "__main__":
    main()
