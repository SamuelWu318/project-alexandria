from dataclasses import dataclass
import data

@dataclass(frozen=True)
class BookProcessed:
    metadata: data.metadata.BookMetadata
    text: data.text.BookText