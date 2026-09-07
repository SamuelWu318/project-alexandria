from __future__ import annotations
from enum import Enum

# --- scene tag vocabulary (shared enrichment enums; the user's to tune) --- #

# Rigid flavor tags for the goal: fetch scenes by emotional FLAVOR, then inject
# that flavor into the user's own prose. Under the restructure the tags are no
# longer scene-level HARD FILTERS: `tone` + `intensity` are picked PER MOMENT (a
# WORD each) by the enrichment LLM, then mapped to numeric coordinates by the
# tables at the bottom of this file (word -> number). The ordered per-moment
# coords become the scene's `vdi_curve` — the affective arc the soft re-rank
# matches. So a tone here names the QUALITY of a beat's feeling; the tables place
# that quality on the valence x dominance grid, and Intensity gives its strength.

class Tone(str, Enum):
    # CONTROLLED VOCABULARY — the single dominant feeling of a MOMENT. Mapped to a
    # (valence, dominance) coordinate by TONE_VD below (dominance is what separates
    # terror — low, victim — from menace — high, threat).
    #
    # Laid out on the empirically-derived 4-D affective space of Fontaine, Scherer,
    # Roesch & Ellsworth (2007), "The World of Emotions is not Two-Dimensional"
    # (Psychological Science). The four axes, in order of importance, are VALENCE
    # (pleasant<->unpleasant), POTENCY/CONTROL (weak<->dominant), AROUSAL
    # (calm<->activated) and NOVELTY (expected<->sudden). Named tones are drawn from
    # Scherer's Geneva Emotion Wheel (20 emotion families on a valence x control
    # wheel), pruned + extended for literary scene-flavor.
    #
    # INTENSITY is a SEPARATE per-moment word, so a tone names the QUALITY of a
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
    # PER-MOMENT strength of the feeling — presence / tension, kept SEPARATE from
    # the tone's inherent arousal so a scene can read low-intensity-focused then
    # rising while the emotion stays analytical. Mapped to the i-axis by INTENSITY_I.
    LOW = "low"            # a faint wash of the feeling, mostly beneath the surface
    MODERATE = "moderate"  # clearly present, colours the scene
    HIGH = "high"          # the feeling dominates every line


class Arc(str, Enum):
    # trajectory of the feeling ACROSS the scene. No longer LLM-authored or a hard
    # filter — DERIVED from the vdi_curve's intensity axis (derive.py) and kept only
    # as an optional display label.
    RISING = "rising"      # feeling intensifies toward the end
    STEADY = "steady"      # feeling holds at one level throughout
    FALLING = "falling"    # feeling releases / subsides toward the end
    TURN = "turn"          # feeling flips or pivots partway through


# --- hard-facet vocabularies (categorical payload filters; §3.2) --- #

class POV(str, Enum):
    # narrative point of view — a HARD filter (exclude, never softened). Coarse on
    # purpose so "third person" matches a single category cleanly.
    FIRST = "first"        # I / we
    SECOND = "second"      # you
    THIRD = "third"        # he / she / they
    MIXED = "mixed"        # shifts within the scene


class Tense(str, Enum):
    # primary narrative tense — a HARD filter. MIXED for scenes that shift.
    PAST = "past"
    PRESENT = "present"
    MIXED = "mixed"


class ProseRegister(str, Enum):
    # register of the prose itself — the LLM picks one WORD (`prose_word`); PROSE_REGISTER
    # maps it to a float 0..1 (`prose_register`, a SOFT facet). Clipped/telegraphic -> grand/metaphorical.
    TELEGRAPHIC = "telegraphic"  # clipped, minimal, staccato
    PLAIN = "plain"              # unadorned, direct
    MEASURED = "measured"        # balanced, moderately literary
    LYRICAL = "lyrical"          # rhythmic, image-rich
    GRAND = "grand"              # ornate, metaphorical, elevated


# --- word -> coordinate tables (store-the-word / derive-the-number; §3.4-3.5) --- #
# The LLM stores a WORD per moment (tone, intensity) or per scene (prose_word); derive.py
# and query.py map those words to numbers through the tables below. Re-tuning a coordinate
# is a payload REFRESH (re-run derive) — only changing the *vocabulary* costs a re-enrich.
# All coordinates are in [0, 1]; the values are DEFAULTS meant to be tuned. Keep every
# tone's (valence, dominance) distinct so a coordinate round-trips back to its word.

# tone WORD -> (valence, dominance). valence: unpleasant 0 .. pleasant 1; dominance: weak/victim 0 .. strong/threatening 1.
TONE_VD: dict[Tone, tuple[float, float]] = {
    # negative · high-arousal
    Tone.DREAD: (0.20, 0.30),    Tone.TERROR: (0.08, 0.15),   Tone.ANXIETY: (0.28, 0.32),
    Tone.MENACE: (0.18, 0.82),   Tone.RAGE: (0.15, 0.78),     Tone.DEFIANCE: (0.38, 0.72),
    Tone.DISGUST: (0.22, 0.52),  Tone.CONTEMPT: (0.26, 0.68),
    # negative · low-arousal
    Tone.GRIEF: (0.12, 0.22),    Tone.MELANCHOLY: (0.32, 0.38), Tone.DESPAIR: (0.06, 0.12),
    Tone.LONELINESS: (0.20, 0.26), Tone.SHAME: (0.16, 0.18),   Tone.GUILT: (0.24, 0.28),
    Tone.REGRET: (0.30, 0.34),   Tone.RESIGNATION: (0.36, 0.24),
    # positive · high-arousal
    Tone.JOY: (0.90, 0.62),      Tone.DELIGHT: (0.86, 0.58),  Tone.EXCITEMENT: (0.80, 0.66),
    Tone.TRIUMPH: (0.92, 0.90),  Tone.HOPE: (0.74, 0.54),     Tone.PASSION: (0.78, 0.72),
    Tone.AMUSEMENT: (0.82, 0.50), Tone.WONDER: (0.76, 0.44),
    # positive · low-arousal
    Tone.SERENITY: (0.84, 0.52), Tone.CONTENTMENT: (0.80, 0.48), Tone.TENDERNESS: (0.78, 0.46),
    Tone.AFFECTION: (0.86, 0.56), Tone.RELIEF: (0.70, 0.42),   Tone.GRATITUDE: (0.80, 0.44),
    Tone.COMPASSION: (0.68, 0.50), Tone.PRIDE: (0.76, 0.82),
    # novelty axis
    Tone.SURPRISE: (0.54, 0.46), Tone.SUSPENSE: (0.36, 0.40), Tone.CURIOSITY: (0.64, 0.56),
    Tone.AWE: (0.72, 0.34),
    # complex / blended
    Tone.BITTERSWEET: (0.50, 0.40), Tone.NOSTALGIA: (0.56, 0.42), Tone.LONGING: (0.40, 0.36),
    Tone.FOREBODING: (0.22, 0.38), Tone.IRONY: (0.46, 0.62),   Tone.SATIRE: (0.44, 0.66),
    Tone.WHIMSY: (0.74, 0.58),   Tone.SOLEMNITY: (0.42, 0.56),
}

# intensity WORD -> i (presence / tension), faint wash 0 .. dominates every line 1.
INTENSITY_I: dict[Intensity, float] = {
    Intensity.LOW: 0.20, Intensity.MODERATE: 0.55, Intensity.HIGH: 0.90,
}

# prose-register WORD -> float, clipped/telegraphic 0 .. grand/metaphorical 1.
PROSE_REGISTER: dict[ProseRegister, float] = {
    ProseRegister.TELEGRAPHIC: 0.0, ProseRegister.PLAIN: 0.25, ProseRegister.MEASURED: 0.5,
    ProseRegister.LYRICAL: 0.75, ProseRegister.GRAND: 1.0,
}

# value-keyed views so a lookup accepts either an enum or the LLM's raw lowercase string.
_TONE_BY_VALUE = {t.value: t for t in Tone}
_INTENSITY_BY_VALUE = {t.value: t for t in Intensity}
_PROSE_BY_VALUE = {t.value: t for t in ProseRegister}


# ** LOCKED **
# Normalize a word (enum or str) to its lowercase value.
def _word(w) -> str:
    return w.value if isinstance(w, Enum) else str(w).strip().lower()


# ---- forward lookups (word -> number); derive.py + query.py call these ----

# ** MAIN ** — derive.moment_curve + query normalizer map a tone word to (valence, dominance).
# A tone word -> its (valence, dominance) coordinate. Raises KeyError on an out-of-vocab word (callers pass in-vocab tones).
def tone_vd(word) -> tuple[float, float]:
    return TONE_VD[_TONE_BY_VALUE[_word(word)]]


# An intensity word -> its i coordinate (presence/tension). Raises KeyError on an out-of-vocab word.
def intensity_i(word) -> float:
    return INTENSITY_I[_INTENSITY_BY_VALUE[_word(word)]]


# ** MAIN ** — derive.py builds each vdi_curve sample from a moment's two words through here.
# A moment's (tone, intensity) words -> its (valence, dominance, intensity) sample = one vdi_curve point.
def moment_vdi(tone, intensity) -> tuple[float, float, float]:
    v, d = tone_vd(tone)
    return (v, d, intensity_i(intensity))


# A prose-register word -> its float coordinate. Raises KeyError on an out-of-vocab word.
def prose_coord(word) -> float:
    return PROSE_REGISTER[_PROSE_BY_VALUE[_word(word)]]


# ---- inverse lookup (coordinate -> nearest word); display + the round-trip check ----

# ** MAIN ** — webtest / query display labels a (valence, dominance) point back to a tone word.
# The tone whose (valence, dominance) coordinate is nearest the given point (Euclidean). Inverse of tone_vd.
def nearest_tone(v: float, d: float) -> Tone:
    return min(TONE_VD, key=lambda t: (TONE_VD[t][0] - v) ** 2 + (TONE_VD[t][1] - d) ** 2)
