from enum import Enum

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
