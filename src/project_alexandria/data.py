from __future__ import annotations
import re, csv, json, zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup, Comment, Tag

from utils import log, SrcPaths, write_json, read_json

# ---- Stage 1: parse + recall (raw Project Gutenberg files -> in-memory objects) ----
# MetadataParser pulls one book's catalog row; SceneParser strips a book's HTML and cuts the body
# into a Book -> Chunks -> Paragraphs tree. build_library / ensure_book tie them behind the recall
# cache (parse each zip once, reload from JSON forever after). See CLAUDE.md for the invariants
# (global paragraph index, lossless extraction, chunk-packing budgets) this stage must not break.

# ---- scene parser constants ----

BOILERPLATE_SELECTORS = ["#pg-header", "#pg-footer",
                         "#project-gutenberg-license"]  # boilerplate to strip out
CHAPTER_SELECTOR = "div.chapter"                        # primary chunk boundary
HEADING_TAGS = ["h1", "h2", "h3", "h4"]                 # mark the start of a segment
MIN_SEGMENT_CHARS = 200                                 # below this = not prose, drop
TARGET_CHARS = 25000                                    # ~6k-token chunk budget
MAX_PARAGRAPHS = 100                                    # per-chunk paragraph cap; bounds dialogue-heavy chapters whose many short paragraphs stay under TARGET_CHARS yet overload the segmenter
OVERLAP_PARAGRAPHS = 3                                  # lookback window for scene context
KEEP_TAGS = {"i", "b", "em", "strong", "sub",           # inline tags kept for rendering
             "u", "small", "br"}


# ---- metadata parser: one book's pg_catalog.csv row -> metadata dict ----

class MetadataParser:

    # Load the catalog into a Text# -> row map, seeding an empty metadata dict.
    def __init__(self, catalog_path: str = SrcPaths.CATALOG_PATH):
        self.metadata = {
            "ID": None,
            "Link": None,
            "Author": None,
            "Translator": None,
            "Title": None,
            "Subjects": set(),
            "Date": None,
            "Language": None,
        }
        self._catalog = self._load_catalog(Path(catalog_path))

    # ** LOCKED **
    # Load pg_catalog.csv into a Text# -> row dict.
    @staticmethod
    def _load_catalog(path: Path) -> dict:
        catalog = {}
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f): catalog[row["Text#"]] = row
        return catalog

    # ** LOCKED **
    # Flip a "Last, First" catalog name into "First Last" (drops [notes]/dates).
    @staticmethod
    def _parse_name(entry: str) -> str | None:
        entry = re.sub(r"\[.*?\]", "", entry).strip()
        if not entry: return None

        parts = [p.strip() for p in entry.split(",")]
        if parts and re.match(r"^\d", parts[-1]):
            parts = parts[:-1]

        if len(parts) >= 2: return f"{parts[1]} {parts[0]}"
        if parts: return parts[0]
        return None

    # Build the metadata dict for one book from its catalog row (author/translator/date parsed out).
    def feed(self, file_code: str) -> dict:
        row = self._catalog.get(file_code)
        if row is None:
            raise KeyError(f"Text# {file_code} not found in {SrcPaths.CATALOG_PATH}")

        metadata = {
            "ID": file_code,
            "Link": "https://www.gutenberg.org/ebooks/" + file_code,
            "Author": None,
            "Translator": None,
            "Title": row["Title"],
            "Subjects": set((s.strip() for s in row["Subjects"].split(";") if s.strip())),
            "Date": None,
            "Language": row["Language"],
        }

        authors = [a.strip() for a in row["Authors"].split(";") if a.strip()]
        if authors: metadata["Author"] = self._parse_name(authors[0])   # "Last, First" -> "First Last"

        for entry in authors:
            if "[translator]" in entry.lower():
                metadata["Translator"] = self._parse_name(entry)
                break

        if row["Issued"]:
            dt = datetime.strptime(row["Issued"].strip(), "%Y-%m-%d")
            metadata["Date"] = f"{dt.strftime('%B')} {dt.day}, {dt.strftime('%Y')}"

        self.metadata = metadata
        return metadata

    # ** LOCKED **  ** MAIN ** — process.scenes_to_records serializes metadata for the sink through here
    # Serialize metadata for JSON/Qdrant: the Subjects set becomes a sorted list.
    @staticmethod
    def to_dict(md: dict) -> dict:
        d = dict(md)
        subj = d.get("Subjects")
        if isinstance(subj, set): d["Subjects"] = sorted(subj)
        return d

    # ** LOCKED **
    # Rehydrate metadata from JSON: Subjects list becomes a set again.
    @staticmethod
    def from_dict(d: dict) -> dict:
        md = dict(d)
        md["Subjects"] = set(md.get("Subjects") or [])
        return md


# ---- Book / Chunk / Paragraph tree ----

@dataclass
class Book:
    file_code: str
    title: str | None
    chunks: list[Chunk] = field(default_factory=list)

    # ** MAIN ** — tests.payload_dump_test dumps every book through here
    # Dump this book's chunk payloads (lossy view) to folder/pg{code}-p.json.
    def to_json(self, folder) -> None:
        write_json(f"{folder}/pg{self.file_code}-p.json", [c.payload() for c in self.chunks])

    # ** LOCKED **
    # Lossless dict for the recall cache (round-trips via from_dict).
    def to_dict(self) -> dict:
        return {
            "file_code": self.file_code,
            "title": self.title,
            "chunks": [c.to_dict() for c in self.chunks],
        }

    # ** LOCKED **  ** MAIN ** — data.ensure_book rebuilds a cached book through here
    # Rebuild a Book from its recall-cache dict.
    @classmethod
    def from_dict(cls, d: dict) -> Book:
        return cls(
            file_code=d["file_code"],
            title=d["title"],
            chunks=[Chunk.from_dict(c) for c in d["chunks"]],
        )


@dataclass
class Chunk:
    chunk_index: int                # chunk number within the book
    chapter_index: int              # chapter the chunk comes from
    chapter_heading: str | None
    part: int                       # 1-based sub-chunk within the chapter
    part_count: int                 # total sub-chunks for this chapter

    context: list[Paragraph] = field(default_factory=list)     # read-only lookback paragraphs
    paragraphs: list[Paragraph] = field(default_factory=list)  # the chunk's own paragraphs

    # Full inspection view of the chunk (used by payload dumps).
    def payload(self) -> dict:
        return {
            "chunk_index": self.chunk_index,
            "chapter_index": self.chapter_index,
            "chapter_heading": self.chapter_heading,
            "part": self.part,
            "part_total": self.part_count,
            "context_paragraphs_total": len(self.context),
            "context_paragraphs": [asdict(p) for p in self.context],
            "paragraphs_total": len(self.paragraphs),
            "paragraphs": [asdict(p) for p in self.paragraphs],
        }

    # ** MAIN ** — process.segment_book feeds this compact JSON to the segmenter LLM
    # Compact JSON string for one section the segmenter must split (context + indexed paragraphs).
    def scene_payload(self) -> dict:
        return json.dumps({
            "chapter_title": self.chapter_heading,
            "section_within_chunk": f"{self.part}/{self.part_count}",
            "read_only_context_paragraphs": [asdict(p) for p in self.context],
            "number_of_indexed_paragraphs": len(self.paragraphs),
            "indexed_paragraphs": [asdict(p) for p in self.paragraphs],
        }, ensure_ascii=False)

    # ** LOCKED **
    # Pretty-print any payload dict as indented JSON.
    def json(self, payload) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=4)

    # ** LOCKED **
    # Lossless dict for the recall cache (round-trips via from_dict).
    def to_dict(self) -> dict:
        return {
            "chunk_index": self.chunk_index,
            "chapter_index": self.chapter_index,
            "chapter_heading": self.chapter_heading,
            "part": self.part,
            "part_count": self.part_count,
            "context": [p.to_dict() for p in self.context],
            "paragraphs": [p.to_dict() for p in self.paragraphs],
        }

    # ** LOCKED **
    # Rebuild a Chunk from its recall-cache dict.
    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        return cls(
            chunk_index=d["chunk_index"],
            chapter_index=d["chapter_index"],
            chapter_heading=d["chapter_heading"],
            part=d["part"],
            part_count=d["part_count"],
            context=[Paragraph.from_dict(p) for p in d["context"]],
            paragraphs=[Paragraph.from_dict(p) for p in d["paragraphs"]],
        )


@dataclass
class Paragraph:
    index: int  # global, book-wide paragraph index (never resets per chapter — see parse_book)
    text: str

    # ** LOCKED **
    # Serialize to a plain {index, text} dict.
    def to_dict(self) -> dict:
        return {"index": self.index, "text": self.text}

    # ** LOCKED **
    # Rebuild a Paragraph from its dict.
    @classmethod
    def from_dict(cls, d: dict) -> Paragraph:
        return cls(index=d["index"], text=d["text"])


# ---- scene parser: book HTML -> Book(Chunks(Paragraphs)) ----

class SceneParser:

    # Open a book's -h.zip, sniff its encoding, decode the HTML, return the stripped body.
    def parse_file(self, file_code: str, folder: str) -> Tag:
        with zipfile.ZipFile(folder + '/pg' + file_code + "-h.zip", 'r') as z:
            file_name = ""
            for name in z.namelist():
                if name.endswith(('.htm', '.html')):
                    file_name = name
                    break

            with z.open(file_name, 'r') as _:
                method = "utf-8"
                head = _.read(200).decode('ascii', errors="ignore").lower()
                if "utf-8" in head: method = "utf-8"
                if "iso-8859-1" in head: method = "iso-8859-1"
                if "ascii" in head: method = "ascii"

            with z.open(file_name, 'r') as h:
                raw_html = h.read().decode(method)

        return self.parse_html(raw_html)      # strip comments/boilerplate, return body

    # Parse HTML and remove comments, headers, footers, and the license; return the body.
    def parse_html(self, raw_html: str) -> Tag:
        soup = BeautifulSoup(raw_html, "html.parser")

        for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
            comment.extract()

        for selector in BOILERPLATE_SELECTORS:
            tag = soup.select_one(selector)
            if tag: tag.decompose()

        return soup.body or soup

    # Return the stripped body as newline-joined plain text.
    def get_text(self, raw_html: str) -> str:
        body = self.parse_html(raw_html)
        return body.get_text(separator="\n", strip=True)

    # ** LOCKED **
    # Keep inline formatting (KEEP_TAGS), unwrap everything else, collapse whitespace.
    @staticmethod
    def _clean_html(tag: Tag) -> str:
        s = BeautifulSoup(tag.decode_contents(), "html.parser")
        for e in s.find_all(True):
            if e.name in KEEP_TAGS: e.attrs = {}
            else: e.unwrap()
        return re.sub(r"\s+", " ", s.decode()).strip()

    # ** LOCKED **
    # First heading text inside `tag`, whitespace-normalized, or None.
    @staticmethod
    def _heading(tag: Tag) -> str | None:
        heading = tag.find(HEADING_TAGS)
        return " ".join(heading.get_text(separator=" ").split()) if heading else None

    # Chapterless fallback: split into (heading, paragraphs) at each heading.
    def _get_segments_fallback(self, body: Tag) -> list[tuple[str | None, list[str]]]:
        segments: list[tuple[str | None, list[str]]] = []
        heading: str | None = None
        texts: list[str] = []

        for e in body.find_all(HEADING_TAGS + ["p"]):
            if e.name in HEADING_TAGS:
                if texts: segments.append((heading, texts))
                heading, texts = " ".join(e.get_text(separator=" ").split()), []
            else:
                t = self._clean_html(e)
                if t: texts.append(t)

        if texts: segments.append((heading, texts))
        return segments

    # Cut the body into ordered (heading, paragraphs) segments (chapters > headings > whole); loose <p> swept up losslessly.
    def get_segments(self, body: Tag) -> list[tuple[str | None, list[str]]]:
        chapters = body.select(CHAPTER_SELECTOR)
        if chapters:
            segments: list[tuple[str | None, list[str]]] = []
            # sweep <p> outside every div.chapter into a leading segment so nothing is dropped
            chapter_ps = {id(p) for ch in chapters for p in ch.find_all("p")}
            loose = [t for t in (self._clean_html(p) for p in body.find_all("p")
                                 if id(p) not in chapter_ps) if t]
            if loose:
                segments.append(("Front Matter", loose))
            for ch in chapters:
                texts = [t for t in (self._clean_html(p) for p in ch.find_all("p")) if t]
                if texts:
                    segments.append((self._heading(ch), texts))
            return segments

        if body.find_all("h2"):
            return self._get_segments_fallback(body)

        # chapterless single document (e.g. the Declaration)
        texts = [t for t in (self._clean_html(p) for p in body.find_all("p")) if t]
        return [(None, texts)] if texts else []

    # Greedily pack paragraphs into parts, flushing at whichever trips first: TARGET_CHARS or MAX_PARAGRAPHS (a lone over-budget paragraph is kept whole).
    def _pack(self, paragraphs: list[Paragraph]) -> list[list[Paragraph]]:
        parts: list[list[Paragraph]] = []
        current: list[Paragraph] = []
        size = 0

        for p in paragraphs:
            over_chars = size + len(p.text) > TARGET_CHARS
            over_count = len(current) >= MAX_PARAGRAPHS
            if current and (over_chars or over_count):
                parts.append(current)
                current, size = [], 0
            current.append(p)
            size += len(p.text)

        if current:
            parts.append(current)
        return parts

    # Segment -> drop tiny segments -> assign GLOBAL paragraph indices -> pack into Chunks (each with OVERLAP lookback context).
    def parse_book(self, body: Tag, file_code: str = "", title: str | None = None) -> Book:
        segments = [(h, ts) for h, ts in self.get_segments(body)
                    if sum(len(t) for t in ts) >= MIN_SEGMENT_CHARS]   # drop sub-prose segments

        all_paras: list[Paragraph] = []   # every kept paragraph, global order
        chunks: list[Chunk] = []
        gi = 0                            # running global paragraph index

        for chapter_index, (heading, texts) in enumerate(segments):
            chapter_paras = [Paragraph(index=gi + i, text=t) for i, t in enumerate(texts)]
            gi += len(chapter_paras)
            all_paras.extend(chapter_paras)

            parts = self._pack(chapter_paras)   # char/count budget split
            for part_no, owned in enumerate(parts):
                # lookback = OVERLAP paragraphs preceding this chunk's first (all_paras[k].index == k)
                start = owned[0].index
                context = all_paras[max(0, start - OVERLAP_PARAGRAPHS):start]
                chunks.append(Chunk(
                    chunk_index=len(chunks),
                    chapter_index=chapter_index,
                    chapter_heading=heading,
                    part=part_no + 1,
                    part_count=len(parts),
                    context=context,
                    paragraphs=owned,
                ))

        return Book(file_code=file_code, title=title, chunks=chunks)

    # ** MAIN ** — data.ensure_book parses a book through here (zip -> chunked Book)
    # Convenience entry: zip -> stripped body -> chunked Book.
    def parse(self, file_code: str, folder: str, title: str | None = None) -> Book:
        return self.parse_book(self.parse_file(file_code, folder), file_code, title)   # read zip, then chunk


# ---- recall cache: parse each book once, reload from JSON after ----

_RIGHTS_RE = re.compile(r'name="dc\.rights"\s+content="([^"]*)"', re.I)


# ** MAIN ** — process.presegmentation_gate reads the US public-domain gate through here
# Search a book's HTML head for the dc.rights <meta> and return its content (opens the -h.zip directly), or None.
def parse_rights(file_code: str, folder: str) -> str | None:
    try:
        with zipfile.ZipFile(f"{folder}/pg{file_code}-h.zip", "r") as z:
            name = next((n for n in z.namelist() if n.endswith((".htm", ".html"))), None)
            if name is None:
                return None
            with z.open(name, "r") as h:
                raw = h.read().decode("utf-8", errors="ignore")
    except (OSError, KeyError, zipfile.BadZipFile):
        return None
    m = _RIGHTS_RE.search(raw)
    return m.group(1).strip() if m else None


# ** LOCKED **
# Per-book recall shard path: recall/books/pg{code}.json (one parsed Book per file — sharded, not monolithic).
def book_file(recall_path: str | Path, file_code: str) -> Path:
    return Path(recall_path) / "books" / f"pg{file_code}.json"


# ** MAIN ** — tests + process load a book's parsed tree through here (lazy, per-book)
# Load a book's parsed tree from its recall shard, PARSING + writing the shard if absent (idempotent).
def ensure_book(file_code: str, data_path: str = SrcPaths.DATA_DIR,
                recall_path: str = SrcPaths.RECALL_DIR, title: str | None = None) -> Book:
    bf = book_file(recall_path, file_code)                       # recall/books/pg{code}.json
    if bf.is_file():
        return Book.from_dict(read_json(bf, {}))                 # reuse the cached tree
    book = SceneParser().parse(file_code, str(data_path), title)  # parse the zip once
    bf.parent.mkdir(parents=True, exist_ok=True)
    write_json(bf, book.to_dict())                              # cache it as its own shard
    return book


# ** MAIN ** — tests.step_two_processing builds the metadata cache + lazy books cache through here
# Load every book's metadata (full cache) and return a LAZY, empty books cache filled on demand by ensure_book.
def build_library(data_path: str = SrcPaths.DATA_DIR,
                  recall_path: str = SrcPaths.RECALL_DIR) -> tuple[dict, dict]:
    path = Path(data_path)
    recall = Path(recall_path)
    md_file = recall / "metadata.json"

    md_cache = read_json(md_file, {})        # code -> metadata dict (Subjects as list)

    metadata: dict = {}
    mdp = MetadataParser()
    md_dirty = False

    for file in sorted(path.iterdir()):
        if not file.is_file():
            log.warn(f"build_library: {file} is not a file — skipping")
            continue
        if not zipfile.is_zipfile(file):
            log.warn(f"build_library: {file} is not a zip — skipping")
            continue

        match = re.search(r"pg(\d+)-h.zip", file.name)
        if not match: continue
        file_code = match.group(1)

        if file_code in md_cache:
            md = MetadataParser.from_dict(md_cache[file_code])   # rehydrate cached row (Subjects -> set)
        else:
            md = mdp.feed(file_code)                             # parse the catalog row
            md_cache[file_code] = MetadataParser.to_dict(md)     # cache it (Subjects -> list)
            md_dirty = True
        metadata[file_code] = md

    if md_dirty:
        write_json(md_file, md_cache)        # persist newly-parsed metadata

    return metadata, {}          # books: lazy per-book shards, filled by ensure_book on demand
