from dataclasses import dataclass
from utils.core import ProjectPaths as proj

@dataclass(frozen=True)
class BookGates:
    is_public_domain: bool
    is_prose: bool

@dataclass(frozen=True)
class Subjects:
    root: str
    branch: list[Subjects]

@dataclass(frozen=True)
class BookMetadata:
    book_id: int
    book_link: str
    author: str
    translator: str
    et_al: set[str]
    title: str
    subjects: Subjects
    language: str

@dataclass(frozen=True)
class TextParagraph:
    index: int
    sentence_range: list[int]
    markup: str
    word_count: int

@dataclass(frozen=True)
class TextSentence:
    index: int
    paragraph_index: int
    text: str
    quote_range: list[int]
    word_count: int

@dataclass(frozen=True)
class BookText:
    parser_version: str
    paragraphs: list[TextParagraph]
    sentences: list[TextSentence]

@dataclass(frozen=True)
class BookProcessed:
    metadata: BookMetadata
    text: BookText

