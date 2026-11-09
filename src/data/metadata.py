from dataclasses import dataclass
from utils.core import log_return
from utils import log
import data

@dataclass(frozen=True)
class BookMetadata:
    book_id: int
    book_link: str
    contributors: dict[str, list[str]]
    title: str
    subjects: list[str]
    language: str


def build(id: int):
    log.info("Book ID: %d | building metadata", id)
    book_id = id
    book_link = "https://www.gutenberg.org/ebooks/" + str(id)
    contributors = data.get.authors(id)
    title = data.get.title(id)
    subjects = data.get.subjects(id)
    language = data.get.language(id)

    instance = BookMetadata(book_id, book_link, contributors, title, subjects, language)
    return log_return(book_id, "metadata", instance)