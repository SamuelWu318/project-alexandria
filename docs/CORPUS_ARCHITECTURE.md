# Corpus Architecture — Levels, Tag Objects, Scenes and Beats, Session

**Status:** design, updated 2026-10-03.
**Scope:** how one book is stored and tagged at the book level, paragraph level, and sentence level; the
scenes and beats built on top of those; and the session assembled from scenes and beats. The retrieval
index, user records, and the skill model are out of scope here (see the end of this doc).

---

## Terms

| term | meaning |
|---|---|
| raw html text | the Gutenberg HTML a book arrives as (`pg{id}-h.zip`); never modified |
| processed html json | `pg{id}.processed.json`: the book level, paragraph level, and sentence level data, built deterministically from the raw html text |
| metadata json | `pg{id}.metadata.json`: everything generated or computed: book level fields, tag objects, scenes and beats |
| book level, paragraph level, sentence level | the three levels a book is stored at |
| tag objects | the eight sentence level labels: Noise, Type, Pace, Mood, Shift, Cast, Setting, Drift |
| scene | a sentence range covering one continuous dramatic unit |
| beat | about 80–120 words of a scene, labelled with one theme |
| move | one dramatic function in play; a long move spans several beats |
| theme | the dramatic function a beat carries; 12 are defined |
| session | one practice drill: the beat order, the Franklin analysis, the bridge exercises, the continuation exercise, the skeleton exercise, the Franklin reconstruction exercise, and the self-check |

---

## Principles

- **The sentence is the atom.** Every sentence has a book-wide index, contiguous from 0. Paragraphs,
  scenes, beats, and exercise passages are all ranges `[start, end)` over that index.
- **Sentences store signals; ranges carry labels.** The tag objects carry local measurements for each
  sentence (mood, pace, shifts). Judgments about a stretch of text (scene borders, beat themes, POV,
  tense, a tension arc) live on ranges: scenes and beats.
- **Every field has one source.** *Deterministic* (the processed html json, built from the raw html
  text), *generated* (a tag object or scene generator, defined by meaning, recorded at the book level), or
  *computed* (from the other two, recomputed whenever its inputs change, never generated or hand-edited).
  Every field, from the book level down to beats, has exactly one of these sources.
- **The text is never edited.** Only artifacts that markup alone identifies are stripped from the raw html
  text (the Gutenberg header/footer, inline page numbers, footnote anchors). Everything else, including
  headings, breaks, footnotes, and front matter, stays in the processed html json and is classified by the
  **Noise** tag object. Rendering hides noise, except a blank line or section break inside a scene, which
  shows as a visible break. Because nothing is deleted after Pass 1, indices never shift.
- **Two jsons per book.** The processed html json (deterministic) and the metadata json (generated or
  computed). Either can be regenerated without the other.
- **Tags are defined by meaning; generators are swappable.** Each definition below is fixed. The model
  or method that produces it may change; the metadata json records which generator and version produced
  each tag object and the scenes.
- **Generation always covers whole books.** Every tag object, and the scenes, are generated for the whole
  book, so generator and version are recorded once, at the book level.
- **Judged in context.** Each sentence is tagged while seeing the sentences around it; each scene is
  generated with a summary of what happened right before it.
- **Noise sentences carry no other tags.** Their entries in every other tag object are empty. They still
  appear as context to neighbouring sentences, since a chapter heading is evidence of a shift.

---

## Files and storage layout

### The raw html text and the two jsons

```
raw html text         pg{id}-h.zip            the Gutenberg HTML; never modified
processed html json   pg{id}.processed.json   deterministic: raw html text + parser_version
                                              ->  book level, paragraph level, sentence level
metadata json         pg{id}.metadata.json    generated: processed html json + generators
                                              ->  book level fields, tag objects, scenes and beats
```

The metadata json records the hash of the processed html json it was built from (`processed_hash`). If
the hashes differ, the metadata json is stale.

### Label-major: labels own sentences

The processed html json owns the sentences: each sentence holds only its text and the deterministic facts
about it. No label lives on a sentence. In the metadata json, each **label owns the sentences it applies
to**, in one of two shapes chosen by how its values behave:

- **Columns**, for values every story sentence has: one array per tag, aligned to the sentence index,
  `null` on noise sentences. A range of sentences is a slice.
- **Index lists**, for things that happen only sometimes: each label lists just the sentences it applies
  to, in index order.

| tag object | shape | entry |
|---|---|---|
| Noise | one index list per `kind` | `[index, border]`. A sentence is noise if it appears under any kind. |
| Type | one column per tag | a value, or `null` on noise |
| Pace | one column | a value, or `null` on noise |
| Mood | one column per tag | a value, or `null` on noise |
| Shift | one index list per tag | `[index, strength]`. Strengths below 0.1 are not stored and read as 0. |
| Cast | one index list per character id, under `present`, `mentioned`, and `speaker` | `present`: runs `[start, end)`; `mentioned` and `speaker`: indices |
| Setting | one index list per setting id, under `current` and `mentioned` | `current`: runs `[start, end)`; `mentioned`: indices |
| Drift | one column | a value, or `null` on noise |

Values on 0..1 scales are stored to 2 decimals. Character and setting ids are strings in the file, because
they are JSON keys. Re-running one tag object replaces one block and touches nothing else. The tag objects
section further down defines every tag per sentence; this section only says how they are stored. Scenes
are one list of scene records, each holding its beats.

### Example

The end of Chapter 1 and the start of Chapter 2 of *Pride and Prejudice*. Indices are illustrative.

```jsonc
// pg1342.processed.json (excerpt: sentences 93–103; paragraph 40 begins at sentence 92, not shown)
{
  "book": {"id": 1342, "title": "Pride and Prejudice", "author": "Jane Austen",
           "subjects": ["..."], "language": "en", "parser_version": "1"},
  "paragraphs": [
    {"index": 40, "sentences": [92, 97],   "markup": "div.chapter > p",  "word_count": 75},
    {"index": 41, "sentences": [97, 98],   "markup": "div.chapter > h2", "word_count": 2},
    {"index": 42, "sentences": [98, 102],  "markup": "div.chapter > p",  "word_count": 70},
    {"index": 43, "sentences": [102, 103], "markup": "div.chapter > p",  "word_count": 8},
    {"index": 44, "sentences": [103, 104], "markup": "div.chapter > p",  "word_count": 22}
  ],
  "sentences": [
    {"index": 93,  "paragraph": 40, "text": "Her mind was less difficult to develop.", "word_count": 7, "quotes": []},
    {"index": 94,  "paragraph": 40, "text": "She was a woman of mean understanding, little information, and uncertain temper.", "word_count": 12, "quotes": []},
    {"index": 95,  "paragraph": 40, "text": "When she was discontented, she fancied herself nervous.", "word_count": 8, "quotes": []},
    {"index": 96,  "paragraph": 40, "text": "The business of her life was to get her daughters married; its solace was visiting and news.", "word_count": 17, "quotes": []},
    {"index": 97,  "paragraph": 41, "text": "Chapter 2", "word_count": 2, "quotes": []},
    {"index": 98,  "paragraph": 42, "text": "Mr. Bennet was among the earliest of those who waited on Mr. Bingley.", "word_count": 13, "quotes": []},
    {"index": 99,  "paragraph": 42, "text": "He had always intended to visit him, though to the last always assuring his wife that he should not go; and till the evening after the visit was paid she had no knowledge of it.", "word_count": 35, "quotes": []},
    {"index": 100, "paragraph": 42, "text": "It was then disclosed in the following manner.", "word_count": 8, "quotes": []},
    {"index": 101, "paragraph": 42, "text": "Observing his second daughter employed in trimming a hat, he suddenly addressed her with:", "word_count": 14, "quotes": []},
    {"index": 102, "paragraph": 43, "text": "“I hope Mr. Bingley will like it, Lizzy.”", "word_count": 8, "quotes": [[1, 40]]},
    {"index": 103, "paragraph": 44, "text": "“We are not in a way to know what Mr. Bingley likes,” said her mother resentfully, “since we are not to visit.”", "word_count": 22, "quotes": [[1, 52], [84, 110]]}
  ]
}
```

```jsonc
// pg1342.metadata.json (excerpt: lists cover the whole book; only entries touching 93–103 are shown)
{
  "processed_hash": "...",
  "generators": {"noise":  {"method": "...", "version": "..."},
                 "type":   {"method": "...", "version": "..."},
                 "scenes": {"method": "...", "version": "...", "themes": "1"}},
  "book": {"readability": 0.71, "word_count": 121000, "sentence_count": 6100, "paragraph_count": 2100,
           "characters": [{"id": 0, "name": "Elizabeth Bennet", "aliases": ["Lizzy", "Eliza"]},
                          {"id": 1, "name": "Mr. Bennet", "aliases": []},
                          {"id": 2, "name": "Mrs. Bennet", "aliases": []},
                          {"id": 3, "name": "Mr. Bingley", "aliases": ["Bingley"]}],
           "settings":   [{"id": 0, "name": "Longbourn", "aliases": ["the house"]},
                          {"id": 1, "name": "Netherfield Park", "aliases": ["Netherfield"]}]},
  "sentences": {

    // INDEX LISTS: each label lists only the sentences it applies to
    "noise": {
      "chapter_heading": [[0, 1.00], [97, 1.00]]              // [index, border]
    },
    "shift": {                                                // [index, strength]; below 0.1 not stored
      "character": [[98, 0.30], [101, 0.60]],
      "place":     [[98, 0.30], [101, 0.40]],
      "time":      [[98, 0.60]],
      "goal":      [[98, 0.50]],
      "pov":       [],
      "action":    [[93, 0.20], [98, 0.80], [101, 0.60], [103, 0.20]]
    },
    "cast": {
      "present":   {"0": [[101, 140]], "1": [[3, 97], [101, 140]], "2": [[3, 97], [101, 140]]},
      "mentioned": {"0": [101, 102, 103], "1": [98, 99, 101, 102], "2": [93, 94, 95, 96, 99, 103],
                    "3": [98, 99, 102, 103]},
      "speaker":   {"1": [102], "2": [103]}
    },
    "setting": {
      "current":   {"0": [[3, 97], [101, 140]]},
      "mentioned": {}                                         // no place is named in 93–103
    },

    // COLUMNS: one value per sentence (sliced here to 93–103); null = noise sentence
    //                93    94    95    96    97    98    99    100   101   102   103
    "type": {
      "dialogue":      [0.00, 0.00, 0.00, 0.00, null, 0.00, 0.00, 0.00, 0.10, 1.00, 0.90],
      "introspection": [0.10, 0.00, 0.20, 0.10, null, 0.00, 0.10, 0.00, 0.00, 0.00, 0.10],
      "description":   [0.30, 0.60, 0.40, 0.30, null, 0.00, 0.00, 0.00, 0.30, 0.00, 0.00],
      "action":        [0.00, 0.00, 0.00, 0.00, null, 0.40, 0.30, 0.10, 0.70, 0.20, 0.20],
      "exposition":    [0.90, 0.80, 0.80, 0.90, null, 0.80, 0.80, 0.60, 0.20, 0.00, 0.00]
    },
    "pace": {
      "pace":          [0.20, 0.20, 0.30, 0.30, null, 0.80, 0.80, 0.60, 0.50, 0.50, 0.50]
    },
    "mood": {
      "valence":       [0.35, 0.25, 0.30, 0.45, null, 0.55, 0.55, 0.55, 0.60, 0.65, 0.35],
      "arousal":       [0.20, 0.25, 0.30, 0.30, null, 0.30, 0.35, 0.35, 0.40, 0.40, 0.50],
      "dominance":     [0.50, 0.40, 0.40, 0.50, null, 0.60, 0.65, 0.60, 0.65, 0.70, 0.40],
      "tension":       [0.10, 0.10, 0.15, 0.15, null, 0.30, 0.40, 0.45, 0.45, 0.40, 0.45]
    },
    "drift": {
      "drift":         [0.40, 0.10, 0.10, 0.10, null, 0.80, 0.10, 0.10, 0.30, 0.10, 0.10]
    }
  },
  "scenes": [
    {"index": 2, "sentences": [98, 140], "...": "..."}
  ]
}
```

What the example shows: the chapter heading (97) is the only noise, and Chapter 2 opens at 98 with time,
goal, and action shifts. Sentences 98–100 are summary (exposition at high pace): Mr. Bingley is mentioned,
nobody is present, and there is no current setting. The scene proper starts at 101, where three characters
become present and Longbourn is the current setting again. Sentence 103 holds two quote spans around its
speech tag, and its speaker is Mrs. Bennet.

### Reading the layout

- **Load once per book.** Columns load straight into arrays, and index lists expand into in-memory
  arrays, in a few milliseconds per book. After that, every lookup is direct.
- **Lookup rules:**
  - Noise: a sentence is noise if it appears under any kind.
  - Shift: a sentence not listed under a tag has strength 0.
  - Presence: a character is present at a sentence if one of their runs contains it.
  - Speaker: the character whose list contains the sentence; `null` if none.
  - Current setting: a setting is current at a sentence if one of its runs contains it.
- **Common reads:**
  - Beat averages: slice the columns over the beat's range.
  - A beat's opening shift: look up its first story sentence in each shift list.
  - Scene border: the highest noise `border` in the gap before the scene.
  - Scene cast: the characters whose presence runs overlap the scene.
  - Scene settings: the settings whose `current` runs overlap the scene.
- **One sentence's full view** is assembled by a helper on demand, for debugging or for feeding a model.
  It is never stored.

```jsonc
// the helper's view of sentence 98
{
  "index": 98, "paragraph": 42,
  "text": "Mr. Bennet was among the earliest of those who waited on Mr. Bingley.",
  "word_count": 13, "quotes": [],
  "noise":   null,
  "type":    {"dialogue": 0.0, "introspection": 0.0, "description": 0.0, "action": 0.4, "exposition": 0.8},
  "pace":    0.8,
  "mood":    {"valence": 0.55, "arousal": 0.3, "dominance": 0.6, "tension": 0.3},
  "shift":   {"character": 0.3, "place": 0.3, "time": 0.6, "goal": 0.5, "pov": 0.0, "action": 0.8},
  "cast":    {"mentioned": [1, 3], "present": [], "speaker": null},
  "setting": {"mentioned": [], "current": []},
  "drift":   0.8
}
```

### Why this layout

Measured on a synthetic book of 6,000 sentences and about 1,000 beats, against storing every label on its
sentence:

| step | labels on each sentence | label-major |
|---|---|---|
| sentence level tag data on disk | 2.37 MB | 0.48 MB |
| load | 12.5 ms | 2.6 ms |
| expand index lists (once per book) | — | 1.2 ms |
| beat averages (about 1,000 beats) | 4.0 ms | 0.2 ms |
| per-sentence rows for a model | 0.4 ms | 0.1 ms |

Both are milliseconds against model calls that take seconds, so the layout does not change how long
Pass 5 takes. The win is size, about 5× smaller (roughly 5 GB instead of 24 GB of sentence level tag data
for 10,000 books), and simpler code, since beat averages and model inputs come straight from arrays.

---

## Book level

**Processed html json** (from the raw html text's own metadata):

| field | meaning |
|---|---|
| `id` | Project Gutenberg ebook number |
| `title` | book title |
| `author` | author name |
| `subjects` | Gutenberg subject headings |
| `language` | kept for future-proofing; currently always `en` |
| `parser_version` | version of the parse + sentence-split rules; changing it regenerates the whole book |

**Metadata json:**

| field | meaning |
|---|---|
| `processed_hash` | hash of the processed html json this metadata was built from |
| `generators` | per tag object, and for the scenes: the method/model and version that produced it. The scenes entry also records the theme-list version. |
| `readability` | 0..1, how easily a modern adult reads the book's story text. 1 = effortless, 0 = very demanding. Book level only. |
| `word_count`, `sentence_count`, `paragraph_count` | **story** counts: noise excluded (a paragraph counts if it holds at least one story sentence). Computed after the Noise pass. |
| `characters` | the book's character roster, `{id, name, aliases}`; the ids used by the Cast tag object and by scenes |
| `settings` | the book's setting roster, `{id, name, aliases}`: places, buildings, and rooms; the ids used by the Setting tag object and by scenes |

Rights are a gate (Pass 0), not a field: a book that fails is never written. Publication year and
year-added-to-Gutenberg are deliberately absent.

---

## Paragraph level (processed html json)

| field | meaning |
|---|---|
| `index` | book-wide, contiguous from 0 |
| `sentences` | `[start, end)` sentence range; sentence count = `end − start` |
| `markup` | the raw html text's block element with its classes, plus its nearest classed container, e.g. `p`, `h2`, `hr.tb`, `div.footnote > p`, `div.poem > p`. Deterministic evidence for Noise and for borders. |
| `word_count` | sum over the paragraph's sentences |

Every block element becomes a paragraph, including headings, captions, and breaks. A textless element
(`<hr>`, an image with no caption) becomes a paragraph holding **one empty sentence**, so the break stays
in the sentence stream. An empty paragraph (a deliberate blank line) does the same, and a run of them
counts once; in plain text, where one blank line separates every paragraph, only extra blank lines count.
Book id is inherited (one processed html json per book).

---

## Sentence level

**Processed html json:**

| field | meaning |
|---|---|
| `index` | book-wide, contiguous from 0, noise included |
| `paragraph` | index of the containing paragraph |
| `text` | the sentence exactly as in the raw html text (after the markup strip) |
| `word_count` | words in `text` |
| `quotes` | character ranges `[start, end)` of `text` that lie inside quotation marks. Quote state is tracked across sentence and paragraph boundaries. The deterministic dialogue signal, used for narration-only judgments such as scene POV and tense. |

**Metadata json:** the eight tag objects below (Noise, Type, Pace, Mood, Shift, Cast, Setting, Drift),
stored label-major as described in [Files and storage layout](#files-and-storage-layout).

---

## Tag objects

### Noise: is this sentence part of the story?

Classifies non-story text so it can be hidden at render time and reasoned about as a possible scene
border. Generated **first**: a noise sentence gets no entry in any other tag object. Letters, poems,
songs, and diary entries *inside* the story are story, not noise.

| tag | range | meaning |
|---|---|---|
| `kind` | enum | what the noise is (below): book furniture or editorial apparatus. A sentence listed under no kind is story. |
| `border` | 0..1 | how strongly this noise marks a border between scenes. 1 = certain border (chapter heading, section break); 0 = none (a footnote mid-scene). |

`kind` values:

| kind | covers |
|---|---|
| `chapter_heading` | chapter, part, and book titles or numbers |
| `section_break` | `* * *`, rules (`<hr>`), ornaments |
| `blank_line` | an empty paragraph: a deliberate blank line in the raw html text |
| `front_matter` | title page, contents, dedication, an editor's preface or introduction |
| `back_matter` | "THE END", index, advertisements, catalogues |
| `epigraph` | quotations heading a chapter or the book |
| `footnote` | note bodies |
| `illustration` | captions and image placeholders |
| `editorial` | transcriber's or editor's notes |
| `other` | non-story text that fits none of the above |

**Rendering:** noise is hidden, except a `blank_line` or `section_break` that falls inside a scene, which
shows as a visible break. Breaks between scenes belong to no scene, and a break at the very start or end
of a displayed passage is dropped.

### Type: what kind of prose is this?

Independent scores, **not normalized**: several can be high at once ("'Go,' he said, slamming the door"
is both dialogue and action). Each score is how much of the sentence does that work: 0 = none, 1 = all of it.

| tag | range | meaning |
|---|---|---|
| `dialogue` | 0..1 | characters' spoken words, rendered as speech, with their speech tags. Judged, so it catches speech the quote marks miss. |
| `introspection` | 0..1 | a character's inner life from inside: thoughts, feelings, memories, interior monologue |
| `description` | 0..1 | what people, places, and things are like: appearance, sound, atmosphere. The picture, not events. |
| `action` | 0..1 | events happening on the page: characters moving, doing, reacting |
| `exposition` | 0..1 | telling the reader: backstory, explanation, context, and summarized events |

### Pace: how fast is story time moving?

One value, after narratology's *duration*: how much story time passes per unit of text.

| tag | range | meaning |
|---|---|---|
| `pace` | 0..1 | 0 = story time stopped (description, reflection: a pause). 0.5 = real time (dialogue, moment-by-moment action; reading takes about as long as the events). 1 = compressed (summary or a skip: hours to years in a line). |

Pace is what separates summary from other exposition: summary = exposition at high pace.

### Mood: what does the sentence feel like?

All four describe **what the text conveys**. All are 0..1 and judged in context: the emotional reading of
the sentence where it stands, not an average of its words' dictionary scores.

| tag | range | meaning |
|---|---|---|
| `valence` | 0..1 | unpleasant (0) to pleasant (1) |
| `arousal` | 0..1 | calm, low energy (0) to agitated, high energy (1) |
| `dominance` | 0..1 | powerless, controlled (0) to in control, dominant (1) |
| `tension` | 0..1 | the anticipation or unease the text conveys about what comes next: settled (0) to on edge (1). It comes from context (stakes, an unanswered threat), so a calm sentence can be high tension. |

Arousal, pace, and tension are independent. A slow description of a door handle turning is low pace and
high tension; a joyful celebration is high arousal and low tension.

The Mood tag object converts valence/arousal/dominance to the nearest mood word, and a mood word back to
valence/arousal/dominance, using the NRC VAD Lexicon v2 norms mapped onto 0..1. Tension has no word mapping.

### Shift: does something change here?

Six independent strengths, 0..1, **not normalized** (several things can change at once). A shift is
tagged on the **first sentence of the new state**. 0 = no change; higher = a bigger change.

| tag | range | meaning |
|---|---|---|
| `character` | 0..1 | who is present changes: someone enters or leaves. One arrival (low) to the whole cast replaced (high). |
| `place` | 0..1 | where the action happens changes. Crossing a room (low) to a different location (high). |
| `time` | 0..1 | story time jumps, forward or back (a flashback counts). "A moment later" (low) to "Ten years passed" (high). |
| `goal` | 0..1 | what the focal character is trying to achieve changes. A minor redirect (low) to an entirely new objective (high). |
| `pov` | 0..1 | whose perspective the narration follows changes. A one-line slip into another character's head (low) to a full switch of viewpoint character or narrator (high). |
| `action` | 0..1 | the current move ends and a new one begins (talk turns into a fight). |

Character, place, time, goal, and pov are evidence for **scene** borders; action is evidence for
**seams**: where a new move begins inside a scene.

### Cast: who is here?

Uses the ids from the book's character roster (`book.characters`). Characters come from the entity
tagger: people, with their mentions (names, titles, pronouns) linked by coreference.

| tag | range | meaning |
|---|---|---|
| `mentioned` | ids | characters named or referred to in this sentence, pronouns resolved |
| `present` | ids | characters physically in the action at this point, whether or not this sentence mentions them |
| `speaker` | id / null | who is speaking, if the sentence carries dialogue |

`mentioned` is not `present`: in "She thought of her mother," the mother is mentioned but not present.

### Setting: where is this happening?

Uses the ids from the book's setting roster (`book.settings`). Settings come from the same entity tagger
as characters: places, buildings, and rooms (locations, facilities, and geo-political entities), with
their mentions ("Netherfield", "the house", "it") linked by coreference.

| tag | range | meaning |
|---|---|---|
| `mentioned` | ids | settings named or referred to in this sentence |
| `current` | ids | the setting the action is in at this point, whether or not this sentence mentions it. A room and the house it belongs to can both be current. Empty when no action is on the page (summary of an offstage event, a reflection). |

`mentioned` is not `current`: in "She longed to be back at Pemberley," Pemberley is mentioned, but the
action is elsewhere. A change in `current` is evidence for a `place` shift.

### Drift: has the subject changed?

| tag | range | meaning |
|---|---|---|
| `drift` | 0..1 | how different the text starting here is from the text just before it. 0 = the same subject continues; 1 = an entirely new subject. Tagged on the first sentence after the change. |

A cheap, text-only border hint (topic change), independent of every judged tag.

---

## Scenes

Scenes live in the metadata json. A scene is one continuous dramatic unit: a stretch of story in one place
and time that plays out one situation. Its fields come from the same three sources as every level:

- **Structure**, from the scene generator: where the scene sits.
- **Generated**, by the scene generator: labels defined by meaning, generator recorded at the book level.
- **Computed**, from the processed html json and the tag objects: recomputed when their inputs change.

| field | source | meaning |
|---|---|---|
| `index` | structure | book-wide scene index, contiguous from 0 |
| `sentences` | structure | `[start, end)` sentence range |
| `context` | generated | a very simple summary of what happened right before (1–2 sentences), carried forward from the previous scene's `summary`. The scene and its beats are generated with it, and the session shows it as the setup line. Empty for the first scene. |
| `summary` | generated | what happens, specifically: who does what, and how it ends (1–3 sentences). The main retrieval text, and the next scene's `context`. |
| `place` | generated | a short phrase for where the scene happens ("the parsonage drawing room at Hunsford") |
| `time` | generated | a short phrase for when ("an evening in spring") |
| `pov` | generated | point of view of the narration, judged on narration only (text outside `quotes`): `first`, `second`, `third_limited`, `third_omniscient` |
| `tense` | generated | tense of the narration, judged the same way: `past`, `present` |
| `beats` | generated + computed | the scene's beats, in order (see Beats) |
| `word_count`, `sentence_count` | computed | story counts (noise excluded) |
| `noise` | computed | indices of noise sentences inside the range, usually empty. Rendering hides them, except a `blank_line` or `section_break`, which shows as a break. |
| `cast` | computed | ids of every character present at some point in the scene (the union of `cast.present` over its sentences) |
| `settings` | computed | ids of every setting current at some point in the scene (the union of `setting.current` over its sentences) |
| `border` | computed | strength of the noise border just before the scene: the highest `noise.border` in the noise run preceding it, 0 if none. Together with `beats[0].shift`, it shows how the scene got split. |

**Read from the beats, not stored:**
- **Intensity arc:** `beats[i].mood.tension`, in order.
- **Shape:** the themes of the beats that start a move (`continues: false`), in order.
- **Actions:** `beats[i].summary`, in order.
- **Scene-wide averages:** word-weighted over the beats, computed when needed.

Retrieval embeddings (of `summary`, beat summaries and gists, `place`) live in the search index, not in
the record.

**Invariants**
- Scenes are in order, never overlap, and together cover every story sentence. Noise between scenes
  belongs to no scene.
- A scene starts and ends on a story sentence.
- Scenes are generated in book order: each scene's `context` is the previous scene's `summary`.

```jsonc
{
  "index": 87,
  "sentences": [2310, 2420],
  "context": "Elizabeth has just learned from Colonel Fitzwilliam that Darcy separated Bingley from Jane.",
  "summary": "Darcy unexpectedly proposes to Elizabeth at Hunsford. She refuses him, accusing him of ruining Jane's happiness and wronging Wickham; he leaves in anger and she breaks down.",
  "place": "the parsonage drawing room at Hunsford",
  "time": "an evening in spring",
  "pov": "third_limited",
  "tense": "past",
  "beats": [
    {"sentences": [2310, 2315], "theme": "establish", "continues": false, "peak": 2312,
     "summary": "Alone at the parsonage, Elizabeth rereads Jane's letters for signs of her sister's unhappiness.",
     "gist": "alone, a woman rereads letters from someone she loves, searching for signs of hurt",
     "word_count": 104, "sentence_count": 5,
     "type": {"dialogue": 0.0, "introspection": 0.7, "description": 0.2, "action": 0.1, "exposition": 0.3},
     "pace": 0.3,
     "mood": {"valence": 0.3, "arousal": 0.4, "dominance": 0.4, "tension": 0.4},
     "shift": {"character": 0.6, "place": 0.2, "time": 0.4, "goal": 0.3, "pov": 0.0, "action": 0.8}},
    {"sentences": [2315, 2319], "theme": "establish", "continues": true, "peak": 2317,
     "summary": "She dwells on Darcy's boast of separating Bingley from Jane, and her resentment grows.",
     "gist": "her resentment toward the man responsible hardens", "...": "..."},
    {"sentences": [2319, 2324], "theme": "pursue", "continues": false, "peak": 2322,
     "summary": "Darcy arrives unexpectedly, paces in silence, then declares his love.",
     "gist": "an unexpected visitor declares his love", "...": "..."}
    // ...about 20 more beats: the declaration's contempt, her refusal, the accusations, his exit, her tears
  ],
  "word_count": 2400,
  "sentence_count": 110,
  "noise": [],
  "cast": [0, 4],          // Elizabeth, Darcy
  "settings": [6],         // the Hunsford parsonage
  "border": 1.0
}
```

---

## Themes

A theme is the **dramatic function of a beat**: what the beat does in its scene. The list has 12 themes
and is fixed beforehand, so every book is segmented against the same vocabulary and the same robbery
always splits the same way. Every beat gets exactly one theme.

Themes name the move, not the subject. A rescue and a reversal of power are both `turn`; the beat's `gist`
carries the specifics.

### Where the themes come from

The list is derived, not collected. Every beat sits on two axes:

- **Arena:** outer (the world: bodies, speech, objects, places) or inner (a mind: feeling, knowledge,
  will).
- **Motion:** build (adds something without flipping anything) or turn (a value flips: the advantage, a
  relationship, what is known, what is chosen).

Each arena then splits by what the beat works on: the stage, the contest between wills, and the
relationship in the world; feeling, knowledge, and will in a mind. A cell splits further only while the
parts train different skills, and stops when one more split would overlap (a writer good at one part
would be equally good at the other).

```
                       BUILD (adds; nothing flips)        TURN (a value flips)
OUTER   stage          establish
(world) contest        pursue · oppose · escalate         act · turn
        relationship                                      bond · rupture
INNER   feeling        react
(mind)  knowledge                                         reveal
        will           reflect                            decide
```

- **Never split by rendering mode.** Pursuing by talking and pursuing by doing are the same move;
  dialogue vs action is the Type tag object's job.
- **Empty cells are empty on purpose.** The stage changes only through other moves. A relationship builds
  through `pursue`, `oppose`, and `bond`. A change of heart is a `reveal`, `decide`, `bond`, or `rupture`.
  Planting a question (foreshadowing) is not a theme: it rarely fills a beat, and whether something is a
  plant depends on a payoff outside the scene, which the scene generator cannot see. It belongs to
  seed/payoff (deferred).

### The 12 themes

| theme | the beat... | the skill it trains | e.g. |
|---|---|---|---|
| `establish` | grounds the reader: a place, person, situation, mood, or background, or the new state of things after events, wherever it appears in the scene | choosing the few details that ground a reader | an opening description; a newcomer introduced mid-scene; backstory; the aftermath |
| `pursue` | has a character go after what they want: ask, persuade, plan, prepare, approach, attempt, deceive | making a want visible through what a character does | a request; a seduction; readying a weapon; a scheme set in motion |
| `oppose` | meets the pursuit with resistance: refusal, argument, obstacle, threat, a competing will | making resistance believable and specific | a refusal; a challenge; a locked door |
| `escalate` | raises the pressure: an engaged conflict intensifies, or a threat draws nearer | raising pressure in steps | voices rise; footsteps approach; a demand becomes an ultimatum |
| `act` | delivers a decisive deed, physical or irrevocable | making a decisive deed clear and irreversible | the blow; the kiss; the theft; the shot; an escape |
| `turn` | reverses the situation: power or fortune flips | a reversal that surprises yet feels inevitable | a rescue; the hunter becomes the hunted; a sudden defeat |
| `bond` | draws characters closer: comfort, affection, alliance, reconciliation | earning closeness without sentimentality | a confidence shared; a reconciliation; a pact |
| `rupture` | breaks a bond: betrayal, abandonment, walking out, a falling-out | making a broken bond cost something | the guard was in on it; a friend sells someone out; storming out on a lover |
| `react` | has a character absorb what just happened | conveying a feeling without naming it | shock; grief; relief; regret; rage |
| `reveal` | surfaces information that changes what a character, or the reader, understands | landing information so it changes how what came before reads | a confession; a discovery; a secret exposed; a realization |
| `reflect` | has a character's mind work: remembering, interpreting, weighing, stating what they believe | keeping thought moving without stalling | a memory surfacing; puzzling over a remark; weighing two options; a statement of conviction |
| `decide` | has a character commit to a course | making a choice visible and costly | resolving to leave; choosing to forgive |

### Tie-breaks between neighbours

| pair | rule |
|---|---|
| `pursue` / `act` | trying vs the decisive, irrevocable deed |
| a failed attempt | an attempt and its immediate failure in the same beat is `oppose`; an attempt that succeeds is `act` |
| `act` / `turn` | the deed itself is `act`; when the point is that the advantage flips, `turn` |
| `oppose` / `escalate` | resistance first meeting the pursuit vs pressure rising on a conflict already engaged (or a threat drawing nearer) |
| `oppose` / `rupture` | resistance inside a contest (a refusal) is `oppose`; a bond breaking is `rupture` |
| `rupture` / `turn` | when a bond breaks, `rupture`, even if the situation also flips; any other reversal is `turn` |
| `rupture` / `pursue` | deceiving an opponent to get something is `pursue`; turning on someone who trusted you is `rupture` |
| `rupture` / `reveal` | the breaking act is `rupture`; discovering an earlier betrayal is `reveal` |
| `reveal` / `turn` | label what changes most: knowledge is `reveal`, circumstances are `turn` |
| `reveal` / `escalate` | label by effect: information that changes understanding is `reveal`; information that brings a threat nearer is `escalate` |
| `reveal` / `react` | a realization while reeling: `reveal` if the new knowledge is the point, `react` if the feeling is. Split them when they separate. |
| `establish` / `reveal` | background that grounds the reader is `establish`; background that changes what we have already seen is `reveal` |
| `reflect` / `establish` | a character remembering is `reflect`; the narrator supplying backstory is `establish` |
| `reflect` / `react` | thinking is `reflect`; feeling is `react` |
| `reflect` / `decide` | weighing is `reflect`; the moment of commitment is `decide` |
| `react` / `establish` | a character's response is `react`; the state of things after events is `establish` |
| exits | an exit is labelled by what it does: escaping danger is `act`, walking out on someone is `rupture`, leaving as things wind down is `establish` |

**Examples**
- A robbery: `establish` (a coach on a dark road), `escalate` (riders close in), `oppose` (the hold-up;
  the guard resists), `act` (the shot), `rupture` (the guard was in on it), `act` (the robbers ride off).
- A quarrel: `pursue` (a conversation about a relationship), `escalate` (it becomes a physical fight),
  `react` (regret), `rupture` (one of them storms out).

---

## Beats

Beats live inside their scene. A beat is a run of consecutive sentences, about **80–120 words**, labelled
with the one move it carries. Beats are sized so the session can stitch them predictably: one beat is
about 100 words, four beats about 400.

**Size.** Target 80–120 words. A beat may stretch to 60–150 words only to put its cut exactly on a move
change, or when the scene or a single long sentence leaves no other choice.

**Where beats are cut,** in order of preference: a move change, a paragraph break, any sentence end.
Never mid-sentence.

**Moves and beats**
- A long move spans several beats; every beat after its first has `continues: true`.
- A move too short for its own beat (under about 60 words) joins the neighbouring beat it belongs to
  most. That beat's theme names the move it spends most of its words on, and its gist covers both. The
  move change inside a merged beat is not a seam.
- Neighbouring beats can share a theme either way: the same move continuing (`continues: true`), or a new
  move with the same function (`continues: false`; two different requests in a row).

**Seam:** the boundary before any beat with `continues: false`, inside a scene. The continuation exercise
cuts at a seam.

| field | source | meaning |
|---|---|---|
| `sentences` | structure | `[start, end)` range inside the scene |
| `theme` | generated | the move the beat carries, from the fixed list |
| `continues` | generated | true when the beat carries on the previous beat's move; always false on a scene's first beat |
| `peak` | generated | index of the sentence that does the beat's work most decisively: the line the beat turns on. The bridge exercises cut their gap here. |
| `summary` | generated | one specific sentence: who does what ("Darcy declares his love, dwelling on her family's inferiority") |
| `gist` | generated | the same beat in general wording: no names, places, objects, or period details, so it could be rewritten in any setting ("a declaration of love undercut by contempt"). The skeleton exercise is built from gists. |
| `word_count`, `sentence_count` | computed | story counts |
| `type` | computed | each Type score averaged over the beat's story sentences, **weighted by words**: how the beat spends its words |
| `pace` | computed | pace averaged the same way, weighted by words |
| `mood` | computed | each Mood score averaged over the beat's story sentences, **one vote per sentence** (a short "He fired." is a full emotional moment; weighting by words would bury it) |
| `shift` | computed | the Shift tag object's values at the beat's first story sentence, **copied, not averaged**: what changed to open the beat (shifts land on the first sentence of the new state) |

**Computation rules**
- Noise sentences are excluded from every computed field.
- Computed fields are recomputed whenever the beat's range or the underlying tag object changes.
- A tag object not generated yet leaves its beat field `null`.
- A beat has no `context` field: beat k's context is beat k−1's `summary`, and the first beat's is the
  scene's `context`.

**Invariants**
- Beats are in order, never overlap, and together span their scene's range exactly. A scene has at least
  one beat.
- A beat starts and ends on a story sentence; its `peak` is a story sentence inside it.
- A scene's first beat has `continues: false`.

---

## Book pass pipeline

**Pass 0: Gate.** Rights (US public domain), language (English), prose (subjects are not poetry or drama
collections). A book that fails is skipped; nothing is written.

**Pass 1: Processed html json** (deterministic, no models), built from the raw html text.
1. Strip markup-identified artifacts only: the Gutenberg header/footer, inline page-number markers,
   footnote anchors.
2. Walk block elements in document order. Each becomes a paragraph with its `markup`; a textless break
   or a blank line becomes one empty sentence.
3. Split into sentences with the one frozen splitter; assign contiguous book-wide indices.
4. Track quotation marks across sentences and paragraphs to fill `quotes`.
5. Count words.

The processed html json is never edited after this pass.

**Pass 2: Noise**, writes the Noise tag object and the book level story counts. Reads text, `markup`, and
context. It runs first because it decides which sentences get every other tag object.

**Pass 3: Tag objects**, writes Type, Pace, Mood, Shift, Cast, Setting, and Drift. Each tag object is
independent, covers the whole book, and is judged in a context window. Noise sentences get no entries but
stay visible as context. Cast and Setting first build the book's character and setting rosters
(`book.characters`, `book.settings`) from the entity tagger, then tag sentences. Several tag objects may
share one model call per window for cost; each is still stored and versioned separately.

**Pass 4: Book level tags**, writes `book.readability`. Reads the story text several paragraphs at a time
and writes one score for the book.

**Pass 5: Scenes and beats**, writes `scenes`. Sequential within a book (books run in parallel), because
each scene's context is the previous scene's summary.
1. The scene generator reads the story sentences in order, with their tag objects, the current context,
   and the theme list.
2. It returns the next scene: where it ends, its beats (80–120 words each, with range, theme,
   `continues`, `peak`, summary, and gist), and its summary, place, time, pov, and tense.
3. The scene's summary becomes the next scene's context.
4. Computed fields are filled from the processed html json and the tag objects.

Pass 5 runs with whatever tag objects exist. Beat fields for tag objects not generated yet stay `null` and
fill in when their tag object is generated.

Every pass records its generator and version in `generators`, alongside `processed_hash`.

### Re-run rules

| change | re-run |
|---|---|
| parse or split rules (`parser_version`) | Pass 1, then the whole metadata json (indices may move) |
| Noise generator | Passes 2–5 (Noise decides which sentences carry tags and what counts as story) |
| any other tag object generator | that tag object, whole book, then recompute the beat and scene fields computed from it. Scene boundaries stay; re-segmenting is an explicit Pass 5 re-run. |
| readability generator | Pass 4 only |
| scene generator or theme list | Pass 5 |
| `processed_hash` differs from the processed html json's hash | the whole metadata json |

---

## Session

A session (about 42 minutes) drills one **beat order**, part to whole: the key line of a beat, the join
between two beats, the whole beat order with the writer's own content, then the whole beat order from
memory. It contains seven parts: the beat order, the Franklin analysis, the bridge exercises, the
continuation exercise, the skeleton exercise, the Franklin reconstruction exercise, and the self-check.

### Beat order

- **Franklin passage:** 4 consecutive beats in one scene, each starting a new move (`continues: false` on
  all four), ending on a complete move (the next beat starts a new move, or the scene ends). About
  320–480 words.
- **Beat order:** the Franklin passage's 4 themes, in order, e.g. `establish`, `act`, `turn`, `react`. It
  has 3 **seams**: between beats 1 and 2, 2 and 3, and 3 and 4.
- **The exercises in between come from other scenes.** The bridge exercises, the continuation exercise,
  and the skeleton exercise draw their beats from scenes other than the Franklin passage's scene, matched
  to the beat order by theme, and from other books where possible. The writer does not see the Franklin
  passage again until the Franklin reconstruction exercise.

### The parts, in order

1. **Beat order** (30 sec). The writer sees the beat order: "Today: establish, act, turn, react."
2. **Franklin analysis** (6 min).
   - Read the Franklin passage (about 3 min).
   - Fill 4 note slots, one per beat, labelled with its theme: what happens, plus one thing the author
     does to make the beat work. About 15 words per slot.
   - The passage and the notes are then locked away.
3. **Bridge exercises** (3 of them, about 2 min each). Beats from other scenes.
   - Each is a ~100-word passage (80–130 words) with a gap at a beat's **peak**: the beat's `peak`
     sentence, 12–30 words. A peak shorter than 12 words takes the next sentence with it. The gap is never
     the passage's first or last sentence.
   - The 3 bridge exercises cover beats 2–4 of the beat order (an `establish` beat rarely has a peak),
     each from a beat with that theme in another scene.
   - The writer fills the gap; the reveal is the original line.
4. **Continuation exercise** (exactly one per session, 7 min). Beats from another scene.
   - From a scene whose beats run the theme of beat 2, then the theme of beat 3, across a seam.
   - Lead-in: 1–2 beats ending on beat 2's theme (100–200 words).
   - The writer is told the next theme ("next: turn") and writes for 5 minutes.
   - Reveal: the author's next beat.
5. **Skeleton exercise** (9 min). Beats from another scene.
   - Source: 4 consecutive beats from another scene, chosen by the same rules as the Franklin passage,
     whose themes match the beat order.
   - The writer sees the 4 gists as a skeleton and writes with their own characters and setting for
     8 minutes.
   - Reveal: the source's beats.
6. **Franklin reconstruction exercise** (12 min).
   - The notes come back. The writer writes for 10 minutes.
   - Reveal: the Franklin passage side by side with the reconstruction, with the writer's notes beside
     each beat.
7. **Self-check** (1 min). One self-check question per beat of the beat order (below), asked of the
   writer's reconstruction.

**Writing is time-boxed.** Beats fix the size of every source passage. What the writer produces is
whatever fits the box; there are no word targets on the writer.

**Short version (about 35 min):** 2 bridge exercises (beats 2 and 4), a 6-minute skeleton exercise box,
an 8-minute Franklin reconstruction box.

### Matching rules

- Themes match exactly. The skeleton exercise may substitute one theme when no exact match exists.
- Seams are real: the beat after a seam has `continues: false`.
- A beat qualifies for a bridge exercise only if its peak gap fits 12–30 words.
- The bridge, continuation, and skeleton exercises never use the Franklin passage's scene, no two of them
  share a scene, and they come from other books where possible.
- No passage crosses a scene border.
- **Setup line:** any passage that starts mid-scene shows the scene's `context` plus the summaries of the
  beats before it in the scene.

### Self-check questions (draft wording)

| theme | question |
|---|---|
| `establish` | Could a reader picture where they are, who is there, and what the situation is? |
| `pursue` | Is it clear what the character wants and how they go after it? |
| `oppose` | Does the resistance push back hard enough to matter? |
| `escalate` | Is the pressure clearly higher at the end of the beat than at the start? |
| `act` | Is the decisive deed unmistakable, and does it feel irreversible? |
| `turn` | Does the advantage visibly flip? |
| `bond` | Are the characters closer at the end than at the start? |
| `rupture` | Do we feel what is lost when the bond breaks? |
| `react` | Do we feel the character absorb what just happened? |
| `reveal` | Does the new information change how what came before reads? |
| `reflect` | Does the thinking go somewhere instead of circling? |
| `decide` | Can we see the options and the moment of choosing? |

---

## Out of scope here

- **Retrieval index:** embeddings of scene summaries, beat summaries and gists, and scene places; filters
  on pov, tense, cast, settings, and themes; the tension arc for shape matching.
- **User records, progress, and the skill model:** deferred for the MVP. A session is assembled fresh each
  time.

## Deferred tag candidates

- **Time expressions:** dates and times of day per sentence. For now, each scene's `time` covers when it
  happens.
- **Stakes:** a second beat label for the value in play (power, trust, affection, safety, knowledge,
  standing). `turn` + power = a reversal of power; `turn` + safety = a rescue.
- **Seed/payoff:** links between a setup (a planted question, a foreshadowing detail) and its later
  resolution. A relation between distant spans, not a sentence tag or a theme; foreshadowing lives here.
