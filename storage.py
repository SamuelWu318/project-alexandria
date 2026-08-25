# FOR CLAUDE — Centralized filesystem layer.
# -----------------------------------------------------------------------------
# The shared foundation every other module imports: (1) every on-disk path the
# pipeline uses, (2) the only functions that read/write JSON or text on disk, (3) the
# single load_dotenv() call — importing storage populates os.environ for everyone —
# (4) the shared scene-tag vocabulary (Tone / Intensity / Arc), and (5) the shared LLM
# client + model id + error policy + SCHEMA_VERSION. Everything that touches the
# filesystem — data.py, process.py, embed.py, tests.py — imports from here instead of
# hand-rolling `json.dumps(...) + write_text(...)`. Edit a path or the on-disk layout
# in ONE place.
#
# Why this exists / invariants to preserve:
#   * write_json / write_text are ATOMIC: content goes to a sibling <file>.tmp then
#     os.replace()s into place, so a crash mid-write can never leave a half-written
#     or corrupt file. Every writer in the pipeline relies on this (recall cache,
#     scene records, resumable checkpoints, status file).
#   * read_json returns `default` on a missing OR corrupt file. Callers lean on this
#     for their "recompute if unreadable" policy (caches and checkpoints).
#   * os.replace is atomic only within one filesystem; the .tmp always lives in the
#     destination's own directory, so this holds.
#
# NOT here (owned elsewhere on purpose): the Qdrant collection name / vector config
# (search.py owns the vector store) and the prompts (process.py / embed.py own those).
# The tag-vocab enums + the LLM client / model / SCHEMA_VERSION DO live here, shared.
# -----------------------------------------------------------------------------
from __future__ import annotations
import json, os
import openai
from enum import Enum
from openai import OpenAI
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

# Load .env ONCE, here, at import time. Every module that needs configuration imports
# storage (for paths / IO), so importing it populates os.environ for all of them — no
# other module calls load_dotenv() itself.
load_dotenv()


# --- LLM client + shared config (model / error policy / schema version) --- #
# The single OpenRouter client + model id + error policy, shared by both LLM stages
# (segmentation in process.py, enrichment + query distillation in embed.py). It lives
# here because storage already loads .env, so os.environ["OPENROUTER_KEY"] is ready.
# MODEL and the retry policy stay the user's tuning surface; SCHEMA_VERSION stamps the
# record shape (embed.py reads it).

SCHEMA_VERSION = 2   # bump when the scene-record shape changes (embed.py reads it).
                     # v2: + decomposed frame fields (subject/verb/object/setting).

MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"

CLIENT = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_KEY"],
)


def _classify_error(e: Exception) -> str:
    """Classify an API error: "transient" (retry with backoff) vs "fatal" (raise now).

    transient = network / 429 / 5xx; fatal = other 4xx where retrying won't help.
    """
    if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError)):
        return "transient"
    if isinstance(e, openai.APIStatusError):
        return "transient" if (e.status_code == 429 or e.status_code >= 500) else "fatal"
    return "transient"  # unknown network-ish -> limited retry


# --- paths (single source of truth; the whole corpus + outputs live under master/) --- #

CATALOG_PATH    = "master/pg_catalog.csv"            # Gutenberg metadata catalog (CSV)
DATA_PATH       = "master/data"                      # source book archives, pg{code}-h.zip
RECALL_PATH     = "master/recall"                    # parse cache: metadata.json + books.json
SCENES_PATH     = "master/scenes"                    # per-book scene records, pg{code}-s.json
CHECKPOINT_DIR  = "master/checkpoints"               # segmentation checkpoints (resumable)
ENRICH_CKPT_DIR = "master/checkpoints/enrich"        # enrichment checkpoints (resumable)
STATUS_PATH     = "master/checkpoints/status.json"   # {book_id: "completed"} -> skip on rerun
TEST_PATH       = "test"                             # directory for testing all of project alexandria


def read_json(path: str | Path, default: Any = None) -> Any:
    """Load JSON from `path`; return `default` if the file is missing or corrupt."""
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def write_json(path: str | Path, obj: Any, indent: int | None = 2) -> None:
    """Atomically write `obj` as UTF-8 JSON (non-ASCII preserved, parents created)."""
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent))


def read_text(path: str | Path, default: str | None = None) -> str | None:
    """Read a UTF-8 text file; return `default` if it does not exist."""
    p = Path(path)
    if not p.exists():
        return default
    return p.read_text(encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    """Atomically write `text`: parents created, temp file written then os.replaced()."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


# --- scene tag vocabulary (shared enrichment enums; the user's to tune) --- #

# Rigid flavor tags for the goal: fetch scenes by emotional FLAVOR, then inject
# that flavor into the user's own prose. A scene is ONE tone (see HOW TO CUT), so
# these tags describe exactly ONE dominant feeling. The enrichment LLM call fills
# them on the fully-stitched scene text; a record holds None until then.

class Tone(str, Enum):
    # CONTROLLED VOCABULARY — the single dominant feeling of a scene, and the rigid
    # facet Mode-1 search filters on (Mode-2 transition pairs are built from it too).
    #
    # Laid out on the empirically-derived 4-D affective space of Fontaine, Scherer,
    # Roesch & Ellsworth (2007), "The World of Emotions is not Two-Dimensional"
    # (Psychological Science). The four axes, in order of importance, are VALENCE
    # (pleasant<->unpleasant), POTENCY/CONTROL (weak<->dominant), AROUSAL
    # (calm<->activated) and NOVELTY (expected<->sudden). Named tones are drawn from
    # Scherer's Geneva Emotion Wheel (20 emotion families on a valence x control
    # wheel), pruned + extended for literary scene-flavor.
    #
    # INTENSITY and ARC are SEPARATE facets, so a tone names the QUALITY of a
    # feeling, never its strength (magnitude lives in Intensity). The sections tile
    # the Valence x Arousal quadrants, then the Novelty axis, then blended tones.
    # Edit freely to retune search; keep values lowercase. NOTE: changing this
    # vocabulary invalidates already-enriched tones — re-enrich to stay consistent.

    # negative · high-arousal — threat, conflict, agitation
    DREAD = "dread"              # anticipatory fear (low potency)
    TERROR = "terror"           # acute, overwhelming fear
    ANXIETY = "anxiety"         # restless worry / unease
    MENACE = "menace"           # outward threat, intimidation (high potency)
    RAGE = "rage"               # hot anger (high potency)
    DEFIANCE = "defiance"       # hostile resistance (high potency)
    DISGUST = "disgust"         # revulsion
    CONTEMPT = "contempt"       # cold scorn / disdain

    # negative · low-arousal — loss, sorrow, withdrawal
    GRIEF = "grief"             # acute mourning
    MELANCHOLY = "melancholy"   # pensive, settled sadness
    DESPAIR = "despair"         # hopelessness
    LONELINESS = "loneliness"   # isolation
    SHAME = "shame"             # self-directed disgrace
    GUILT = "guilt"             # remorse over a wrong done
    REGRET = "regret"           # wishing the past undone
    RESIGNATION = "resignation" # bleak, defeated acceptance

    # positive · high-arousal — energy, uplift, victory
    JOY = "joy"                 # bright happiness
    DELIGHT = "delight"         # lively, playful pleasure
    EXCITEMENT = "excitement"   # eager anticipation / thrill
    TRIUMPH = "triumph"         # exultant victory (high potency)
    HOPE = "hope"               # forward-looking optimism
    PASSION = "passion"         # ardor, desire, romantic heat
    AMUSEMENT = "amusement"     # mirth, comic pleasure
    WONDER = "wonder"           # marvel at something new (novelty+)

    # positive · low-arousal — calm, warmth, connection
    SERENITY = "serenity"       # tranquil peace
    CONTENTMENT = "contentment" # settled satisfaction
    TENDERNESS = "tenderness"   # gentle, protective care
    AFFECTION = "affection"     # fond, steady love
    RELIEF = "relief"           # tension released
    GRATITUDE = "gratitude"     # thankfulness
    COMPASSION = "compassion"   # sympathy for another's pain
    PRIDE = "pride"             # quiet self-worth (high potency)

    # novelty axis — expectation violated (valence-ambiguous)
    SURPRISE = "surprise"       # sudden astonishment
    SUSPENSE = "suspense"       # tense, held-breath anticipation
    CURIOSITY = "curiosity"     # drawn-in intrigue / interest
    AWE = "awe"                 # reverent, overwhelmed vastness

    # complex / blended — mixed-valence literary flavors
    BITTERSWEET = "bittersweet" # joy and sorrow at once
    NOSTALGIA = "nostalgia"     # wistful ache for the past
    LONGING = "longing"         # yearning for the absent / distant
    FOREBODING = "foreboding"   # ominous sense of coming ill
    IRONY = "irony"             # detached incongruity
    SATIRE = "satire"           # mocking social critique
    WHIMSY = "whimsy"           # light, fanciful playfulness
    SOLEMNITY = "solemnity"     # grave, ceremonial dignity


class Intensity(str, Enum):
    LOW = "low"            # a faint wash of the feeling, mostly beneath the surface
    MODERATE = "moderate"  # clearly present, colours the scene
    HIGH = "high"          # the feeling dominates every line


class Arc(str, Enum):
    # trajectory of the feeling ACROSS the scene. Powers the box-to-box sequence
    # search ("as tension builds") and scaffolds a summary that names direction.
    RISING = "rising"      # feeling intensifies toward the end
    STEADY = "steady"      # feeling holds at one level throughout
    FALLING = "falling"    # feeling releases / subsides toward the end
    TURN = "turn"          # feeling flips or pivots partway through
