# FOR CLAUDE — Stage 1: parse + recall (the read-from-Gutenberg layer).
# -----------------------------------------------------------------------------
# Turns raw Project Gutenberg files into in-memory objects the rest of the
# pipeline consumes. Two halves:
#   * MetadataParser — pulls one book's metadata row out of pg_catalog.csv.
#   * SceneParser    — unzips a book's HTML, strips boilerplate, and cuts the body
#                      into a Book -> Chunks -> Paragraphs tree.
# build_library() ties them together behind the RECALL cache (master/recall):
# parse each .zip once, then reload from JSON forever after.
#
# Key invariants (do not break):
#   * Paragraph.index is GLOBAL and book-wide — it never resets per chapter, and
#     all_paras[k].index == k. process.py's stitching and context lookback depend
#     on this contiguity.
#   * Chunks pack paragraphs up to TARGET_CHARS on paragraph boundaries; a lone
#     over-budget paragraph is kept whole (never split mid-paragraph).
#   * Extraction is LOSSLESS: <p> outside every div.chapter is swept up as a
#     leading "Front Matter" segment rather than silently dropped.
#   * to_dict/from_dict on Book/Chunk/Paragraph are the lossless recall round-trip;
#     payload()/scene_payload() are lossy views for dumps / the LLM.
# All path constants + JSON IO come from storage.py.
# -----------------------------------------------------------------------------
from __future__ import annotations
import re, csv, json, zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup, Comment, Tag

from storage import CATALOG_PATH, DATA_PATH, RECALL_PATH, read_json, write_json
import log

# --- scene parser constants --- #

BOILERPLATE_SELECTORS = ["#pg-header", "#pg-footer",
                         "#project-gutenberg-license"]  # boilerplate to strip out
CHAPTER_SELECTOR = "div.chapter"                        # primary chunk boundary
HEADING_TAGS = ["h1", "h2", "h3", "h4"]                 # mark the start of a segment
MIN_SEGMENT_CHARS = 200                                 # below this = not prose, drop
TARGET_CHARS = 30000                                    # ~6k-token chunk budget
OVERLAP_PARAGRAPHS = 3                                  # lookback window for scene context
KEEP_TAGS = {"i", "b", "em", "strong", "sub",           # inline tags kept for rendering
             "u", "small", "br"}


# --- METADATA PARSER --- #

class MetadataParser:
    def __init__(self, catalog_path: str = CATALOG_PATH):
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

    @staticmethod
    def _load_catalog(path: Path) -> dict:
        """Load pg_catalog.csv into a Text# -> row dict."""
        catalog = {}
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f): catalog[row["Text#"]] = row
        return catalog

    @staticmethod
    def _parse_name(entry: str) -> str | None:
        """Flip a "Last, First" catalog name into "First Last" (drops [notes]/dates)."""
        entry = re.sub(r"\[.*?\]", "", entry).strip()
        if not entry: return None

        parts = [p.strip() for p in entry.split(",")]
        if parts and re.match(r"^\d", parts[-1]):
            parts = parts[:-1]

        if len(parts) >= 2: return f"{parts[1]} {parts[0]}"
        if parts: return parts[0]
        return None

    def feed(self, file_code: str) -> dict:
        """Build the metadata dict for one book from its catalog row."""
        row = self._catalog.get(file_code)
        if row is None:
            raise KeyError(f"Text# {file_code} not found in {CATALOG_PATH}")

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
        if authors: metadata["Author"] = self._parse_name(authors[0])

        for entry in authors:
            if "[translator]" in entry.lower():
                metadata["Translator"] = self._parse_name(entry)
                break

        if row["Issued"]:
            dt = datetime.strptime(row["Issued"].strip(), "%Y-%m-%d")
            metadata["Date"] = f"{dt.strftime('%B')} {dt.day}, {dt.strftime('%Y')}"

        self.metadata = metadata
        return metadata

    @staticmethod
    def to_dict(md: dict) -> dict:
        """Serialize metadata for JSON/Qdrant: the Subjects set becomes a sorted list."""
        d = dict(md)
        subj = d.get("Subjects")
        if isinstance(subj, set): d["Subjects"] = sorted(subj)
        return d

    @staticmethod
    def from_dict(d: dict) -> dict:
        """Rehydrate metadata from JSON: Subjects list becomes a set again."""
        md = dict(d)
        md["Subjects"] = set(md.get("Subjects") or [])
        return md


# --- BOOK / CHUNK / PARAGRAPH TREE --- #

@dataclass
class Book:
    file_code: str
    title: str | None
    chunks: list[Chunk] = field(default_factory=list)

    def to_json(self, folder) -> None:
        """Dump this book's chunk payloads (lossy view) to folder/pg{code}-p.json."""
        write_json(f"{folder}/pg{self.file_code}-p.json", [c.payload() for c in self.chunks])

    def to_dict(self) -> dict:
        """Lossless dict for the recall cache (round-trips via from_dict)."""
        return {
            "file_code": self.file_code,
            "title": self.title,
            "chunks": [c.to_dict() for c in self.chunks],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Book:
        """Rebuild a Book from its recall-cache dict."""
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

    def payload(self) -> dict:
        """Full inspection view of the chunk (used by payload dumps)."""
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

    def scene_payload(self) -> dict:
        """Compact JSON string fed to the segmenter LLM (one section to split)."""
        return json.dumps({
            "chapter_title": self.chapter_heading,
            "section_within_chunk": f"{self.part}/{self.part_count}",
            "read_only_context_paragraphs": [asdict(p) for p in self.context],
            "number_of_indexed_paragraphs": len(self.paragraphs),
            "indexed_paragraphs": [asdict(p) for p in self.paragraphs],
        }, ensure_ascii=False)

    def json(self, payload) -> str:
        """Pretty-print any payload dict as indented JSON."""
        return json.dumps(payload, ensure_ascii=False, indent=4)

    def to_dict(self) -> dict:
        """Lossless dict for the recall cache (round-trips via from_dict)."""
        return {
            "chunk_index": self.chunk_index,
            "chapter_index": self.chapter_index,
            "chapter_heading": self.chapter_heading,
            "part": self.part,
            "part_count": self.part_count,
            "context": [p.to_dict() for p in self.context],
            "paragraphs": [p.to_dict() for p in self.paragraphs],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        """Rebuild a Chunk from its recall-cache dict."""
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
    index: int  # global, book-wide paragraph index (never resets per chapter)
    text: str

    def to_dict(self) -> dict:
        """Serialize to a plain {index, text} dict."""
        return {"index": self.index, "text": self.text}

    @classmethod
    def from_dict(cls, d: dict) -> Paragraph:
        """Rebuild a Paragraph from its dict."""
        return cls(index=d["index"], text=d["text"])


# --- SCENE PARSER --- #

class SceneParser:
    def parse_file(self, file_code: str, folder: str) -> Tag:
        """Open a book's -h.zip, decode its HTML, and return the stripped body."""
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

        return self.parse_html(raw_html)

    def parse_html(self, raw_html: str) -> Tag:
        """Parse HTML and remove comments, headers, footers, and the license."""
        soup = BeautifulSoup(raw_html, "html.parser")

        for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
            comment.extract()

        for selector in BOILERPLATE_SELECTORS:
            tag = soup.select_one(selector)
            if tag: tag.decompose()

        return soup.body or soup

    def get_text(self, raw_html: str) -> str:
        """Return the stripped body as newline-joined plain text."""
        body = self.parse_html(raw_html)
        return body.get_text(separator="\n", strip=True)

    @staticmethod
    def _clean_html(tag: Tag) -> str:
        """Keep inline formatting (KEEP_TAGS), unwrap everything else, collapse whitespace."""
        s = BeautifulSoup(tag.decode_contents(), "html.parser")
        for e in s.find_all(True):
            if e.name in KEEP_TAGS: e.attrs = {}
            else: e.unwrap()
        return re.sub(r"\s+", " ", s.decode()).strip()

    @staticmethod
    def _heading(tag: Tag) -> str | None:
        """First heading text inside `tag`, whitespace-normalized, or None."""
        heading = tag.find(HEADING_TAGS)
        return " ".join(heading.get_text(separator=" ").split()) if heading else None

    def _get_segments_fallback(self, body: Tag) -> list[tuple[str | None, list[str]]]:
        """Chapterless fallback: split into (heading, paragraphs) at each heading."""
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

    def get_segments(self, body: Tag) -> list[tuple[str | None, list[str]]]:
        """Cut the body into ordered (heading, paragraphs) segments (chapters > headings > whole)."""
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

    def _pack(self, paragraphs: list[Paragraph]) -> list[list[Paragraph]]:
        """Greedily group paragraphs into parts <= TARGET_CHARS, never splitting one."""
        parts: list[list[Paragraph]] = []
        current: list[Paragraph] = []
        size = 0

        for p in paragraphs:
            if current and size + len(p.text) > TARGET_CHARS:
                parts.append(current)
                current, size = [], 0
            current.append(p)
            size += len(p.text)

        if current:
            parts.append(current)
        return parts

    def parse_book(self, body: Tag, file_code: str = "", title: str | None = None) -> Book:
        """Segment -> drop tiny segments -> assign global indices -> pack into Chunks.

        Each chunk's context reaches back OVERLAP_PARAGRAPHS in global order (across
        the chapter boundary for a chapter's first part) so cross-chunk scene
        continuations can be caught during segmentation.
        """
        segments = [(h, ts) for h, ts in self.get_segments(body)
                    if sum(len(t) for t in ts) >= MIN_SEGMENT_CHARS]

        all_paras: list[Paragraph] = []   # every kept paragraph, global order
        chunks: list[Chunk] = []
        gi = 0                            # running global paragraph index

        for chapter_index, (heading, texts) in enumerate(segments):
            chapter_paras = [Paragraph(index=gi + i, text=t) for i, t in enumerate(texts)]
            gi += len(chapter_paras)
            all_paras.extend(chapter_paras)

            parts = self._pack(chapter_paras)
            for part_no, owned in enumerate(parts):
                # lookback = OVERLAP paragraphs preceding this chunk's first paragraph (all_paras[k].index == k)
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

    def parse(self, file_code: str, folder: str, title: str | None = None) -> Book:
        """Convenience entry point: zip -> stripped body -> chunked Book."""
        return self.parse_book(self.parse_file(file_code, folder), file_code, title)


# --- RECALL CACHE --- #

_RIGHTS_RE = re.compile(r'name="dc\.rights"\s+content="([^"]*)"', re.I)


def parse_rights(file_code: str, folder: str) -> str | None:
    """Search a book's HTML head for the dc.rights <meta> and return its content, or None.

    Gutenberg stamps every book with <meta name="dc.rights" content="..."> — the US
    public-domain gate reads this. Opens the -h.zip directly, independent of the
    body-stripping parse, so the value is available even for books we never segment.
    """
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


def build_library(data_path: str = DATA_PATH,
                  recall_path: str = RECALL_PATH) -> tuple[dict, dict]:
    """Load every book's metadata + parsed Book, backed by the recall cache.

    metadata.json and books.json under recall_path are code -> data maps: reuse a
    cached entry when present, else parse the .zip and add it. Returns live dicts
    keyed by file_code — metadata[code] -> metadata dict, books[code] -> Book.
    """
    path = Path(data_path)
    recall = Path(recall_path)
    md_file = recall / "metadata.json"
    books_file = recall / "books.json"

    md_cache = read_json(md_file, {})        # code -> metadata dict (Subjects as list)
    books_cache = read_json(books_file, {})  # code -> Book dict

    metadata: dict = {}
    books: dict = {}
    mdp = MetadataParser()
    sp = SceneParser()
    md_dirty = books_dirty = False

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
            md = MetadataParser.from_dict(md_cache[file_code])
        else:
            md = mdp.feed(file_code)
            md_cache[file_code] = MetadataParser.to_dict(md)
            md_dirty = True
        metadata[file_code] = md

        if file_code in books_cache:
            book = Book.from_dict(books_cache[file_code])
        else:
            book = sp.parse(file_code, str(path), md.get("Title"))
            books_cache[file_code] = book.to_dict()
            books_dirty = True
        books[book.file_code] = book

    if md_dirty:
        write_json(md_file, md_cache)
    if books_dirty:
        write_json(books_file, books_cache)

    return metadata, books
