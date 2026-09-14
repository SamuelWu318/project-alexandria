from __future__ import annotations
import json, math, time
from pathlib import Path
from pydantic import BaseModel, ValidationError, field_validator
from openai import pydantic_function_tool

from utils import (CLIENT, MODEL, MODEL_PARAMS, SELECT_PROMPT,
                   classify_llm_error, inject_retry_notes, read_json, write_json)
from utils import log
from utils.vectorstore import open_client   # the Qdrant client door (foundation; no feature-file dep)
import query                                # read-path front door: word->coord + search, one door with payloads

# ---- Phase 11: the similarity-gold builder (rough queries -> pooled candidates -> LLM top-5 gold) ----
# The product goal is a top-5 of CONTENT-similar scenes, so the durable benchmark is a per-query relevance
# set, not one iconic scene. This module builds that set reproducibly in two steps (PLAN §6):
#   1. POOL   — run each rough query through the real read path (query.run) over a few configs, dedupe, and
#               keep the top-POOL_DEPTH candidate scenes (scene_id + summary + facets — enough to judge) to
#               pool_candidates.json.
#   2. SELECT — read pool_candidates.json, chunk-feed each query + its candidates to the LLM judge
#               (SELECT_PROMPT, forced `output_selection`), and freeze the ranked top-TOP_K matches per query
#               to pool_queries.json — one entry per query in the queries.json format (the rough query fields
#               + a best-first `matches` list), the gold evals scores P@5/nDCG@5.
# RUN IT AS A FILE (`python -m similarity_gold`): main() pools if pool_candidates.json is missing, then selects.
# OWNERSHIP: SELECT_PROMPT (utils/llm.py) is the USER'S tuning surface — retune the judge to move the gold;
# this file is the plumbing (pool merge, chunked judge call + retry, write-back). The rough query set lives
# in webtest/gold/rough_queries.json; the pool and the gold are written beside it.

GOLD_DIR = Path(__file__).resolve().parent / "webtest" / "gold"
ROUGH_PATH = GOLD_DIR / "rough_queries.json"       # input: the hand-authored rough / similarity queries
POOL_PATH = GOLD_DIR / "pool_candidates.json"      # step-1 output / step-2 input: top-20 candidates per query
GOLD_PATH = GOLD_DIR / "pool_queries.json"         # step-2 output: queries.json-format gold (query + top-5 matches)

POOL_DEPTH = 20          # candidates pooled per query (PLAN §6 step 1)
TOP_K = 5                # gold size = the product goal's "top-5" (PLAN §6 step 2)
CHUNK = 20               # candidates per judge call; a deeper pool tournaments down to one finalist chunk
# candidate payload fields shown to the judge — what happens (summary) + the feel/style facets, enough to rank.
CANDIDATE_FIELDS = ("scene_id", "book_id", "summary", "descriptors", "pov", "tense",
                    "prose_register", "dialogue_ratio", "arc")


# ---- select tool schema (the judge's forced output; prompt = SELECT_PROMPT, the user's surface) ----

class Selection(BaseModel):
    # one picked scene: the id (copied from a candidate) + a one-clause why (kept in the gold for audit).
    scene_id: str
    reason: str = ""

    # Coerce a non-empty id to str; a blank id fails validation so the retry loop replays it.
    @field_validator("scene_id", mode="before")
    @classmethod
    def _clean_id(cls, v):
        v = str(v).strip() if v is not None else ""
        if not v:
            raise ValueError("scene_id must be non-empty")
        return v


class TopFive(BaseModel):
    # the judge's ranked pick, best match first (validated ⊆ the candidate pool by _run_select, not here).
    ranked: list[Selection]


SELECT_TOOL = pydantic_function_tool(
    TopFive, name="output_selection",
    description="Return up to five candidate scenes, the most CONTENT-similar to the query FIRST.")
# NON-STRICT for the same reason as enrich's tool: require_parameters drops strict-schema on some providers;
# pydantic + the candidate-membership check below validate the args instead.
SELECT_TOOL["function"]["strict"] = False


# ---- LLM judge: one forced tool call + retry (retry policy reused from utils.llm; loop owned here) ----

# System-prompt addendum for a RETRY (why the last pick was rejected, replayed); "" on the first attempt.
def _retry_note(notes: list[str]) -> str:
    if not notes:
        return ""
    lines = "\n".join(f"- attempt {i + 1}: {n}" for i, n in enumerate(notes))
    return ("\n\n# RETRY — FIX THE SELECTION\n"
            "Your previous answer was not a valid ranked pick from the candidate pool. Return "
            "output_selection with up to five scene_ids COPIED EXACTLY from the candidates, best match "
            f"first, no duplicates. Problems from earlier attempts:\n{lines}")


# One forced `output_selection` call validated into a TopFive whose ids all lie in `valid_ids`. Retries never
# abort on a bad answer (fresh convo + replayed reason, temperature climbs then freezes); only a fatal API error raises.
def _run_select(user_content: str, valid_ids: set) -> TopFive:
    TEMP_FREEZE_ATTEMPTS = 8
    notes: list[str] = []
    transient_tries = attempt = 0
    while True:
        temp = 0 if attempt == 0 else min(0.6, math.log(attempt ** 0.20) + 0.15)
        messages = [
            {"role": "system", "content": inject_retry_notes(SELECT_PROMPT, notes, _retry_note)},  # utils.llm splice
            {"role": "user", "content": user_content},
        ]
        try:
            response = CLIENT.chat.completions.create(
                model=MODEL, temperature=temp, tools=[SELECT_TOOL], messages=messages,
                tool_choice={"type": "function", "function": {"name": "output_selection"}},  # force the judge tool
                **MODEL_PARAMS,                                    # routing + reasoning only (utils/llm.py)
            )
        except Exception as e:
            if classify_llm_error(e) == "fatal":                  # utils.llm: non-retryable 4xx -> abort
                raise RuntimeError(f"output_selection fatal (no retry): {e}") from e
            transient_tries += 1
            sleep = min(2 ** transient_tries, 30)
            attempt = min(attempt + 1, TEMP_FREEZE_ATTEMPTS)
            log.warn(f"select transient retry {transient_tries} (sleep {sleep}s): {e}")
            time.sleep(sleep)
            continue

        choices = response.choices
        msg = choices[0].message if choices else None
        if not msg or not msg.tool_calls:
            reason = "did not call output_selection"
        else:
            try:
                data = TopFive.model_validate_json(msg.tool_calls[0].function.arguments)
            except (ValidationError, json.JSONDecodeError, ValueError) as e:
                reason = f"arguments failed schema validation: {e}"
            else:
                picked = [s.scene_id for s in data.ranked]
                unknown = [p for p in picked if p not in valid_ids]
                if not picked:
                    reason = "returned no scenes"
                elif unknown:
                    reason = f"scene_ids not in the candidate pool: {unknown[:5]}"
                elif len(picked) != len(set(picked)):
                    reason = "duplicate scene_ids in the ranking"
                else:
                    return data

        attempt = min(attempt + 1, TEMP_FREEZE_ATTEMPTS)
        notes.append(reason)
        log.warn(f"select validation retry (fresh convo, temp {temp}): {reason[:120]}")


# ---- the builder ----

class SimilarityGold:
    # Reproducible top-5 similarity-gold builder: pool -> chunked LLM select -> write-back. One instance owns
    # the paths + an optional shared Qdrant client (opened lazily, closed by close()). `build()` = pool then select.
    def __init__(self, client=None, *, rough_path: Path = ROUGH_PATH,
                 pool_path: Path = POOL_PATH, gold_path: Path = GOLD_PATH):
        self.rough_path = Path(rough_path)
        self.pool_path = Path(pool_path)
        self.gold_path = Path(gold_path)
        self._client = client
        self._owns_client = client is None

    # The Qdrant client (opened lazily on first pool; foundation open_client).
    @property
    def client(self):
        if self._client is None:
            self._client = open_client()
        return self._client

    # Close the client iff this instance opened it (a caller-supplied client is left alone).
    def close(self):
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    # The rough query entries (the "rough_queries" list of the input json).
    def _queries(self) -> list[dict]:
        return read_json(self.rough_path, {}).get("rough_queries", [])

    # One rough entry's intuition sliders as WORDS (tones/prose/dialogue) for query.run + the judge's view; {} if none.
    @staticmethod
    def _slider_words(e: dict) -> dict:
        return {f: e[f] for f in ("tones", "prose", "dialogue") if e.get(f) is not None}

    # The pooling configs for one query: semantic-only under both blends + (if set) the intuition sliders —
    # diverse enough that a strong scene no single config surfaces still enters the pool (the pooling caveat).
    def _pool_configs(self, e: dict) -> list[dict]:
        cfgs = [{}, {"combine": "max"}]                        # semantic only, two blends
        sliders = self._slider_words(e)
        if sliders:
            cfgs.append(sliders)                               # + the intuition-slider lane
        return cfgs

    # ** MAIN ** — the CLI + build() pool here (step 1)
    # Pool the top-`depth` candidate scenes per rough query: run each config through query.run, round-robin
    # merge so EVERY config contributes, dedupe by scene_id, keep the judge-facing fields. Writes pool_path.
    def pool(self, *, depth: int = POOL_DEPTH, limit: int = POOL_DEPTH) -> dict:
        out: dict = {}
        for e in self._queries():
            runs = [query.run(self.client, e["summary"], limit=limit, **cfg)   # read path: word->coord + search
                    for cfg in self._pool_configs(e)]
            picked: dict = {}                                  # scene_id -> payload, round-robin across configs
            for rank in range(max((len(r) for r in runs), default=0)):
                if len(picked) >= depth:
                    break
                for r in runs:
                    if rank < len(r):
                        sid = r[rank].payload.get("scene_id")
                        if sid and sid not in picked:
                            picked[sid] = r[rank].payload
                            if len(picked) >= depth:
                                break
            cands = [{f: pl.get(f) for f in CANDIDATE_FIELDS} for pl in picked.values()]
            out[e["id"]] = {"summary": e["summary"], "sliders": self._slider_words(e), "candidates": cands}
            log.info(f"pooled {e['id']}: {len(cands)} candidates from {len(runs)} configs")
        write_json(self.pool_path, out)                        # read_write: atomic json
        log.done(f"pooled {len(out)} queries -> {self.pool_path}")
        return out

    # Judge one chunk of candidates: build the {query, candidates} payload, force the select tool, return the
    # ranked Selections (<= top_k, best-first) — the id AND its one-clause reason, so reasons survive into the gold.
    def _select_chunk(self, summary: str, sliders: dict, chunk: list[dict], top_k: int) -> list[Selection]:
        payload = json.dumps({
            "query": {"summary": summary, **({"leanings": sliders} if sliders else {})},
            "candidates": chunk,
        }, ensure_ascii=False)
        data = _run_select(payload, {c["scene_id"] for c in chunk})   # forced judge call + retry
        return data.ranked[:top_k]

    # Reduce a candidate list to the ranked top_k via the judge: one call when it fits CHUNK, else a tournament
    # (shortlist top_k per chunk, re-judge the survivors) so an arbitrarily deep pool still collapses to one ranking.
    def _judge(self, summary: str, sliders: dict, cands: list[dict], top_k: int) -> list[Selection]:
        pool = cands
        while True:
            chunks = [pool[i:i + CHUNK] for i in range(0, len(pool), CHUNK)]
            if len(chunks) <= 1:
                return self._select_chunk(summary, sliders, pool, top_k)
            survivors: list[dict] = []
            for ch in chunks:
                by_id = {c["scene_id"]: c for c in ch}
                survivors.extend(by_id[s.scene_id] for s in self._select_chunk(summary, sliders, ch, top_k)
                                 if s.scene_id in by_id)
            pool = survivors

    # One gold match from a pooled candidate + the judge's reason: scene id / book / summary (for audit) + why.
    @staticmethod
    def _match(cand: dict, reason: str) -> dict:
        return {"scene_id": cand.get("scene_id"), "book_id": cand.get("book_id"),
                "summary": cand.get("summary"), "reason": reason}

    # ** MAIN ** — the file run + build() select here (step 2)
    # Judge each pooled query and write pool_queries.json in the QUERIES.JSON format: one entry per query =
    # the rough query's own fields + a best-first `matches` list ({scene_id, book_id, summary, reason}). Reads
    # pool_candidates.json when no pool is passed. `summary` + `reason` are kept so the gold is auditable; the
    # evals scorer reads only the match scene_ids.
    def select(self, pool: dict | None = None, *, top_k: int = TOP_K) -> dict:
        pool = pool if pool is not None else read_json(self.pool_path, {})
        if not pool:
            raise RuntimeError(f"no pool at {self.pool_path}; run pool() first (or run this file to pool + select)")
        rough_by_id = {e["id"]: e for e in self._queries()}    # the original query fields, to mirror queries.json
        entries: list[dict] = []
        for qid, item in pool.items():
            cands = item.get("candidates") or []
            entry = dict(rough_by_id.get(qid) or {"id": qid, "summary": item.get("summary")})
            if not cands:
                entry["matches"] = []
                entries.append(entry)
                log.warn(f"select {qid}: empty pool -> no matches")
                continue
            by_id = {c["scene_id"]: c for c in cands}
            ranked = self._judge(item.get("summary") or entry.get("summary", ""),
                                 item.get("sliders") or {}, cands, top_k)
            entry["matches"] = [self._match(by_id[s.scene_id], s.reason)
                                for s in ranked if s.scene_id in by_id]
            entries.append(entry)
            log.info(f"selected {qid}: {[s.scene_id for s in ranked]}")
        write_json(self.gold_path, {"pool_queries": entries})
        log.done(f"froze top-{top_k} matches for {len(entries)} queries -> {self.gold_path}")
        return {"pool_queries": entries}

    # ** MAIN ** — one call makes more gold (pool then select)
    def build(self, *, depth: int = POOL_DEPTH, limit: int = POOL_DEPTH, top_k: int = TOP_K) -> dict:
        return self.select(self.pool(depth=depth, limit=limit), top_k=top_k)


# ---- run as a file ----

# ** ENTRY **
# Build the Phase-11 similarity gold end to end — run this file (`python -m similarity_gold`), no arguments:
# pool the candidates if pool_candidates.json is missing, then read it, LLM-select the top-5 matches per query,
# and write pool_queries.json (queries.json format). Tune SELECT_PROMPT (utils/llm.py) to change the judge.
def main():
    g = SimilarityGold()
    try:
        if not g.pool_path.exists():
            log.step(f"no {g.pool_path.name} yet -> pooling candidates first")
            g.pool()                                           # step 1: retrieve + write pool_candidates.json
        g.select()                                             # step 2: read pool_candidates.json -> pool_queries.json
    finally:
        g.close()


if __name__ == "__main__":
    main()
