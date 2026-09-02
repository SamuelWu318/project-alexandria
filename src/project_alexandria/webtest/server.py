# FOR CLAUDE — Local read-path test server (no LLM).
# -----------------------------------------------------------------------------
# A tiny stdlib HTTP server (zero new deps) that drives the LIVE stores so the
# gold queries + a browser front end can exercise the whole retrieval path:
#   * Qdrant  (search.py)      -> unified search(): what-happens (summary + svos) + flavor
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
from utils import SrcPaths, read_json
from utils import relational
from utils import subjects          # book-level subject trie (the folder pre-filter)
import search

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
GOLD = HERE / "gold"
PORT = 8765

# Retrieval-strategy switch for a subject-folder filter: at or below this many allowed
# books, force exact brute-force over the isolated set (Strategy 2 — fast + exact when few
# qualify); above it, use the filtered HNSW walk (Strategy 1 — efficient over a broad set).
# Counted in BOOKS (what the folder knows); the deeper cost is scenes, but books proxy it.
EXACT_BOOK_THRESHOLD = 10

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
subjects.ensure_table(_conn)         # subject folder tree lives as a sibling table in scenes.db

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


def _build_subject_tree() -> int:
    """Populate the folder tree for the SEARCHABLE books only (those whose scenes loaded).

    The trie derives from catalog subjects, but here we scope it to the books actually in
    the index so folder navigation never dead-ends on a book with no retrievable scenes.
    Scoping the TABLE (not just the queries) means count_branch / books_in_branch are
    inherently searchable-only. Rebuilt each boot — idempotent, cheap at test scale.
    """
    md = read_json(Path(SrcPaths.RECALL_DIR) / "metadata.json", {})
    md = {code: m for code, m in md.items() if code in _book_title}   # searchable subset
    n = subjects.upsert_many(_conn, md)
    print(f"[webtest] subject tree: {n} rows across {len(md)} searchable books")
    return n


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

def _run_query(q: dict, limit: int, flt, exact: bool = False) -> dict:
    """Run ONE query object -> a result column. Never raises: errors ride back in-band.

    `flt` is a prebuilt Qdrant filter (single-book, subject-folder set, or None for all
    books) computed once by the caller — the hard pre-filter that decides which books can
    surface before the vector search even runs. `exact` picks the retrieval strategy for
    that filter: brute-force the isolated set (few allowed books) vs the HNSW walk (many).
    """
    mode = q.get("mode", "search")   # echoed back for the UI column header; not a dispatch key
    # anti-descriptors need matching weights; fill equal if the query gave only the list
    anti = q.get("anti_descriptors")
    aw = q.get("anti_weights")
    if anti and not aw:
        aw = [1.0 / len(anti)] * len(anti)
    try:
        # ONE unified entry: search() activates the what-happens channels (summary + svos
        # moment sentences) and/or the flavor channel (descriptors), runs them over `flt`,
        # and merges (weighted RRF) when more than one is active. An empty field abstains.
        pts = search.search(
            _client,
            summary=q.get("summary") or None,
            moments=q.get("moments") or None,        # manual clause sentence(s) -> svos channel
            descriptors=q.get("descriptors") or None,
            weights=q.get("weights"),
            anti_descriptors=anti, anti_weights=aw,
            anti_strength=q.get("anti_strength", 1.0),
            method_weights=q.get("method_weights") or None,
            normalize=q.get("normalize", "zscore"),
            flt=flt, exact=exact, limit=limit,
        )
        results = [_card(p.payload, p.score) for p in pts]
        return {"label": q.get("label", ""), "mode": mode, "meta": q.get("meta", {}),
                "target_book_id": q.get("target_book_id"), "results": results}
    except Exception as e:  # bad weights, empty query, etc. -> show it, don't 500 the batch
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
                "moments": rec.get("moments"),        # [{sentence, subject, verb, object, setting}]
                "svos": rec.get("svos"),              # the embedded clause sentences
                "prev_scene_id": rec.get("prev_scene_id"),
                "next_scene_id": rec.get("next_scene_id"),
            })
            return self._json(d)
        if p == "/api/subject":
            # folder listing at a subject path: sub-folders (child terms + counts) and the
            # books in this branch (the "files"). path=Fiction&path=Italy -> ["Fiction","Italy"];
            # no path -> the roots. This is the pre-retrieval navigator, all from SQL indexes.
            path = parse_qs(u.query).get("path", [])
            folders = [{"term": t, "count": subjects.count_branch(_conn, path + [t])}
                       for t in subjects.children(_conn, path)]
            if path:
                bids = subjects.books_in_branch(_conn, path)
                books = [{"book_id": b, "title": _book_title.get(b, f"book {b}")} for b in bids]
                subject, count = " -- ".join(reversed(path)), len(bids)
            else:
                books, subject, count = [], None, len(_book_title)   # root: browse only
            return self._json({"path": path, "subject": subject, "folders": folders,
                               "books": books, "count": count})
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
            queries = body.get("queries", [])
            # hard pre-filter precedence: a subject-folder branch (subject_path) wins over a
            # single pinned book (book_id). The branch filters by the indexed `subject_paths`
            # label (one term, not an id-set); its book count comes from the SQL tree and drives
            # the exact-vs-walk switch. A branch with zero searchable books -> every column empty.
            subject_path = body.get("subject_path")   # reversed nav list, e.g. ["Fiction","Italy"]
            book_id = body.get("book_id") or None
            if subject_path:
                n = subjects.count_branch(_conn, subject_path)
                if n == 0:
                    empty = [{"label": q.get("label", ""), "mode": q.get("mode", "fused"),
                              "meta": q.get("meta", {}), "target_book_id": q.get("target_book_id"),
                              "results": []} for q in queries]
                    return self._json({"columns": empty})
                flt = search.subject_filter(subject_path)
                exact = n <= EXACT_BOOK_THRESHOLD              # few books -> brute-force the set
            elif book_id:
                flt = search.book_filter(book_id)
                exact = True                                   # one book is maximally selective
            else:
                flt = None
                exact = False
            cols = [_run_query(q, limit, flt, exact) for q in queries]
            return self._json({"columns": cols})
        return self._json({"error": f"no route {u.path}"}, 404)


def _ctype(path: str) -> str:
    if path.endswith(".css"): return "text/css"
    if path.endswith(".js"): return "application/javascript"
    if path.endswith(".html"): return "text/html; charset=utf-8"
    return "application/octet-stream"


def main():
    _load_scenes()
    _build_subject_tree()            # folder pre-filter tree, scoped to the loaded books
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
