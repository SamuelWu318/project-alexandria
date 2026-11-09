from dataclasses import dataclass

@dataclass(frozen=True)
class TextChapter:
    index: int
    paragraph_range: list[int]
    markup: str

@dataclass(frozen=True)
class TextParagraph:
    sentences: tuple[TextSentence]
    markup: str

@dataclass(frozen=True)
class TextSentence:
    text: str
    quote_range: list[tuple[int, int]]

@dataclass(frozen=True)
class BookText:
    parser_version: str
    chapters: list[TextChapter]
    paragraphs: list[TextParagraph]
    sentences: list[TextSentence]