"""Extract, index, and retrieve passages from a text-based research paper."""

from dataclasses import dataclass
from hashlib import blake2b
from io import BytesIO
import re
from uuid import uuid4

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError

MAX_PDF_MB = 20
MAX_PAGES = 80
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class PaperError(ValueError):
    """The uploaded file cannot be indexed."""


@dataclass(frozen=True)
class Passage:
    id: str
    page: int
    text: str


@dataclass(frozen=True)
class PaperIndex:
    filename: str
    page_count: int
    collection: Collection
    client: ClientAPI
    lexical: bool = False

    @property
    def passage_count(self) -> int:
        return self.collection.count()

    def search(self, query: str, limit: int = 4) -> list[Passage]:
        query_args = {"query_embeddings": [word_vector(query)]} if self.lexical else {"query_texts": [query]}
        results = self.collection.query(
            **query_args,
            n_results=min(limit, self.passage_count),
            include=["documents", "metadatas"],
        )
        return [
            Passage(passage_id, int(metadata["page"]), document)
            for passage_id, document, metadata in zip(
                results["ids"][0], results["documents"][0], results["metadatas"][0]
            )
        ]

    def delete(self) -> None:
        self.client.delete_collection(self.collection.name)


def read_passages(pdf_bytes: bytes) -> tuple[int, list[Passage]]:
    if len(pdf_bytes) > MAX_PDF_MB * 1024 * 1024:
        raise PaperError(f"Choose a PDF smaller than {MAX_PDF_MB} MB.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=140)
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        if reader.is_encrypted:
            raise PaperError("Choose a PDF that opens without a password.")
        page_count = len(reader.pages)
        if page_count > MAX_PAGES:
            raise PaperError(f"Choose a paper with at most {MAX_PAGES} pages.")

        passages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").replace("\x00", " ").strip()
            for chunk_number, chunk in enumerate(splitter.split_text(text), start=1):
                passages.append(Passage(f"p{page_number}-{chunk_number}", page_number, chunk))
    except (PdfReadError, OSError) as exc:
        raise PaperError("This PDF could not be read. Try another copy of the paper.") from exc

    if not passages:
        raise PaperError("This PDF has no selectable text. Choose a text-based research paper.")
    return page_count, passages


def word_vector(text: str) -> list[float]:
    vector = [0.0] * 1024
    for word in re.findall(r"\b[\w-]{3,}\b", text.casefold()):
        slot = int.from_bytes(blake2b(word.encode(), digest_size=2).digest(), "big") % len(vector)
        vector[slot] += 1
    return vector


def index_paper(filename: str, pdf_bytes: bytes, *, lexical: bool = False) -> PaperIndex:
    return index_passages(filename, *read_passages(pdf_bytes), lexical=lexical)


def index_passages(filename: str, page_count: int, passages: list[Passage], *, lexical: bool) -> PaperIndex:
    client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
    collection = client.create_collection(
        name=f"paper-{uuid4().hex}",
        configuration={"hnsw": {"space": "cosine"}},
    )
    try:
        embedding_args = {"embeddings": [word_vector(p.text) for p in passages]} if lexical else {}
        collection.add(
            **embedding_args,
            ids=[passage.id for passage in passages],
            documents=[passage.text for passage in passages],
            metadatas=[{"page": passage.page} for passage in passages],
        )
    except Exception:
        client.delete_collection(collection.name)
        raise
    return PaperIndex(filename, page_count, collection, client, lexical)
