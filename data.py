from __future__ import annotations
import re, csv, json, zipfile, unicodedata
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup, Comment, Tag

# --- metadata parser constants --- #

CATALOG_PATH = "pg_catalog.csv"
DATA_PATH = "data"

# --- scene parser constants --- #

BOILERPLATE_SELECTORS = ["#pg-header", "#pg-footer", 
                         "#project-gutenberg-license"]  # selectors for boilerplate removal
CHAPTER_SELECTOR = "div.chapter"                        # chapter selectors to break into chunks
HEADING_TAGS = ["h1", "h2", "h3", "h4"]                 # identify the start of segments/chapters 
MIN_SEGMENT_CHARS = 200                                 # guaranteed to not be prose; at least 200 chars
TARGET_CHARS = 24000                                    # about a maximum of 6k tokens for each chunk
OVERLAP_PARAGRAPHS = 3                                  # number of paragraphs to look back to
KEEP_TAGS = {"i", "b", "em", "strong", "sub",           # keep for text rendering with html 
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
            "Subjects": [],
            "Date": None,
            "Language": None,
        }

        self._catalog = self._load_catalog(Path(catalog_path))

    # loads the catalog of metadata
    @staticmethod
    def _load_catalog(path: Path) -> dict:
        catalog = {}
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f): catalog[row["Text#"]] = row
        return catalog

    # flips names to be first name, last name
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

    # bring all metadata found in catalog into metadata dict
    def feed(self, file_code: str) -> dict:
        row = self._catalog.get(file_code)
        if row is None:
            raise KeyError(f"Text# {file_code} not found in {CATALOG_PATH}")

        metadata = {
            "ID": file_code,
            "Link": "https://www.gutenberg.org/ebooks/" + file_code,
            "Author": None,
            "Translator": None,
            "Title": row["Title"],
            "Subjects": [s.strip() for s in row["Subjects"].split(";") if s.strip()],
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


# --- BOOK CLASS --- #


# each book will be wrapped in a book class
@dataclass
class Book:
    file_code: str
    title: str | None
    chunks: list[Chunk] = field(default_factory=list)

    def to_json(self, folder) -> str:
        path = Path(f"{folder}/pg{self.file_code}-p.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        out = []

        for c in self.chunks:
            out.append(c.payload())

        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

# each book will contain chunks
@dataclass
class Chunk:
    # chunk: one chapter, or a chunk of size TARGET_CHARS
    chunk_index: int                # chunk number within the book
    chapter_index: int              # chapter the chunk comes from
    chapter_heading: str | None
    part: int                       # 1-based sub-chunk within the chapter
    part_count: int                 # total sub-chunks for this chapter

    # read-only: used to identify potential scenes. Will allow 
    # previous paragraph to recognize the scene.
    context: list[Paragraph] = field(default_factory=list)
    paragraphs: list[Paragraph] = field(default_factory=list)

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

    def scene_payload(self) -> dict:
        return json.dumps({
            "chapter_title": self.chapter_heading,
            "section_within_chunk": f"{self.part}/{self.part_count}",
            "read_only_context_paragraphs": [asdict(p) for p in self.context],
            "number_of_indexed_paragraphs": len(self.paragraphs),
            "indexed_paragraphs": [asdict(p) for p in self.paragraphs],
        }, ensure_ascii=False)

    def json(self, payload) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=4)

# each chunk will contain paragraphs
@dataclass
class Paragraph:
    index: int  # global paragraph index, book-wide (never resets per chapter)
    text: str   # text of each paragraph


# --- SCENE PARSER --- #


class SceneParser:
    # open up the file and sent it to parse_html
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

        return self.parse_html(raw_html)
    
    # elimination of headers, footers, and licenses
    def parse_html(self, raw_html: str) -> Tag:
        soup = BeautifulSoup(raw_html, "html.parser")

        for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
            comment.extract()

        for selector in BOILERPLATE_SELECTORS:
            tag = soup.select_one(selector)
            if tag: tag.decompose()

        return soup.body or soup

    def get_text(self, raw_html: str) -> str:
        body = self.parse_html(raw_html)
        return body.get_text(separator="\n", strip=True)

    @staticmethod
    def _clean_html(tag: Tag) -> str:
        # preserve inline formatting. HTML reparsed into an isolated tree
        # then the whitespace is collapsed, returning a string safe for use
        s = BeautifulSoup(tag.decode_contents(), "html.parser")
        for e in s.find_all(True):
            # remove unnecessary attrs if keep, unwrap if remove
            if e.name in KEEP_TAGS: e.attrs = {}
            else: e.unwrap()
        return re.sub(r"\s+", " ", s.decode()).strip()

    @staticmethod
    def _heading(tag: Tag) -> str | None:
        heading = tag.find(HEADING_TAGS)
        return " ".join(heading.get_text(separator=" ").split()) if heading else None

    def _get_segments_fallback(self, body: Tag) -> list[tuple[str | None, list[str]]]:
        # only necessary if there are no chapters. Divides at headings instead.
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
        # stage 2: cut the body into ordered (heading, paragraphs) segments.
        chapters = body.select(CHAPTER_SELECTOR)
        if chapters:
            segments: list[tuple[str | None, list[str]]] = []
            # <p> that live outside every div.chapter (a title page, a stray
            # note) would be silently dropped by a chapter-only walk; sweep them
            # up as a leading front-matter segment so extraction stays lossless.
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

        # for chapterless single document (e.g. the Declaration)
        texts = [t for t in (self._clean_html(p) for p in body.find_all("p")) if t]
        return [(None, texts)] if texts else []

    def _pack(self, paragraphs: list[Paragraph]) -> list[list[Paragraph]]:
        # stage 3: greedily pack paragraphs into parts, each <= TARGET_CHARS,
        # on paragraph boundaries. A lone paragraph over budget gets its own
        # part (kept whole - never split mid-paragraph). Indices are assigned by
        # the caller (global, book-wide) and preserved here untouched.
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
        # segment, drop structural front/back matter, assign one book-wide
        # global paragraph index, then pack each content chapter into
        # budget-bounded chunks. Context reaches back OVERLAP_PARAGRAPHS in
        # global order - across the chapter boundary for a chapter's first
        # part - so scene continuations spanning chapters can be caught later.
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
                # lookback = the OVERLAP paragraphs immediately preceding this
                # chunk's first owned paragraph in global order (empty only for
                # the book's very first chunk). all_paras[k].index == k.
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
        # convenience entry point: zip -> stripped body -> chunked Book.
        return self.parse_book(self.parse_file(file_code, folder), file_code, title)


def main():
    path = Path(DATA_PATH)

    metadata = {}
    mdp = MetadataParser()

    # search all files in path and create scenes
    books = []
    sp = SceneParser()
    mdp = MetadataParser()

    for file in sorted(path.iterdir()):
        if not file.is_file(): continue

        match = re.search(r"pg(\d+)-h.zip", file.name)
        if not match: continue

        file_code = match.group(1)
        metadata = mdp.feed(match.group(1))
        book = sp.parse(file_code, str(path), metadata.get("Title"))

        books.append(book)

    for book in books:
        print(book.file_code)
        book.to_json("test")

if __name__ == "__main__":
    main()
