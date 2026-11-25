"""Document-processing workflow: extract -> clean -> chunk -> embed -> store.

Run `python ingest.py` to build the FAISS index from the files in data/.
"""
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config


class DocumentError(Exception):
    """A document could not be read. The message is shown to the student."""


@dataclass
class IngestReport:
    processed: List[Tuple[str, int]] = field(default_factory=list)  # (file, chunks)
    failed: List[Tuple[str, str]] = field(default_factory=list)     # (file, reason)
    total_chunks: int = 0


# ------------------------------------------------------------ finding files
def find_documents(data_dir: Path = config.DATA_DIR) -> List[Path]:
    """Return all supported files (PDF, TXT) inside data_dir, sorted by name."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    return sorted(
        p for p in data_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in config.SUPPORTED_EXTENSIONS
        and not p.name.startswith(".")
    )


# ------------------------------------------------------------ extraction
def _extract_pdf_pages(path: Path) -> List[Tuple[Optional[int], str]]:
    try:
        with open(path, "rb") as handle:
            head = handle.read(1024)
    except OSError as exc:
        raise DocumentError(f"cannot open file ({exc})") from exc
    if b"%PDF" not in head:
        raise DocumentError("not a valid or readable PDF (invalid PDF header)")

    pages: List[Tuple[Optional[int], str]] = []
    try:
        import fitz  # PyMuPDF (primary reader)
    except ImportError:
        fitz = None

    try:
        if fitz is not None:
            doc = fitz.open(str(path))
            if doc.needs_pass:
                raise DocumentError("the PDF is password protected")
            for number, page in enumerate(doc, start=1):
                pages.append((number, page.get_text("text") or ""))
            doc.close()
        else:
            from pypdf import PdfReader  # fallback reader
            reader = PdfReader(str(path))
            if reader.is_encrypted:
                raise DocumentError("the PDF is password protected")
            for number, page in enumerate(reader.pages, start=1):
                pages.append((number, page.extract_text() or ""))
    except DocumentError:
        raise
    except Exception as exc:  # corrupted file, unsupported structure, etc.
        raise DocumentError(f"not a valid or readable PDF ({exc})") from exc

    if not any(text.strip() for _, text in pages):
        raise DocumentError("no readable text found (a scanned PDF needs OCR)")
    return pages


def extract_pages(path: Path) -> List[Tuple[Optional[int], str]]:
    """Return a list of (page_number, raw_text). TXT files use page None."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_pages(path)
    if suffix == ".txt":
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise DocumentError(f"cannot open file ({exc})") from exc
        if not text.strip():
            raise DocumentError("the text file is empty")
        return [(None, text)]
    raise DocumentError(f"unsupported file type '{suffix}'")


# ------------------------------------------------------------ cleaning
def clean_text(text: str) -> str:
    """Remove extra spaces, odd characters and broken line endings."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", " ").replace("\u00a0", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)            # join words split at line end
    text = re.sub(r"[ \t]+", " ", text)                  # many spaces -> one space
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)         # trim spaces around newlines
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)         # single newline -> space
    text = re.sub(r"\n{3,}", "\n\n", text)               # many blank lines -> one
    return text.strip()


# ------------------------------------------------------------ chunking
def _is_heading(paragraph: str) -> bool:
    return (
        len(paragraph) <= 80
        and "\n" not in paragraph
        and not paragraph.endswith((".", "?", "!", ",", ";"))
    )


def _attach_headings(paragraphs: List[str]) -> List[str]:
    """Join a short heading line to the paragraph that follows it."""
    result: List[str] = []
    pending: List[str] = []
    for paragraph in paragraphs:
        if _is_heading(paragraph):
            pending.append(paragraph)
            continue
        if pending:
            paragraph = "\n".join(pending) + "\n" + paragraph
            pending = []
        result.append(paragraph)
    if pending:
        result.append("\n".join(pending))
    return result


def _split_long_paragraph(paragraph: str, chunk_size: int) -> List[str]:
    """Split a paragraph longer than chunk_size at sentence borders."""
    if len(paragraph) <= chunk_size:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces: List[str] = []
    current = ""
    for sentence in sentences:
        while len(sentence) > chunk_size:           # a single huge sentence
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:chunk_size])
            sentence = sentence[chunk_size:]
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= chunk_size:
            current += " " + sentence
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


def _overlap_tail(chunk: str, overlap: int) -> str:
    """Last `overlap` characters of a chunk, starting at a word border."""
    if overlap <= 0:
        return ""
    tail = chunk[-overlap:]
    if len(chunk) > overlap and not chunk[-overlap - 1].isspace():
        cut = tail.find(" ")
        if 0 <= cut < len(tail) - 1:
            tail = tail[cut + 1:]
    return tail.strip()


def split_into_chunks(
    text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP
) -> List[str]:
    """Split text into overlapping chunks of about `chunk_size` characters.

    Paragraph borders are respected where possible. The last `overlap`
    characters of a chunk are repeated at the start of the next chunk, so a
    sentence that falls on a border is not lost.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be 0 or more and smaller than chunk_size")
    text = (text or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    paragraphs = _attach_headings(paragraphs)

    pieces: List[str] = []
    for paragraph in paragraphs:
        pieces.extend(_split_long_paragraph(paragraph, chunk_size))

    chunks: List[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + 2 + len(piece) <= chunk_size:
            current += "\n\n" + piece
        else:
            chunks.append(current)
            tail = _overlap_tail(current, overlap)
            current = f"{tail} {piece}".strip() if tail else piece
    if current:
        chunks.append(current)
    return chunks


# ------------------------------------------------------------ loading
def load_document(path: Path) -> List[Dict]:
    """Read one file and return its chunk records (text, source, page)."""
    path = Path(path)
    chunks: List[Dict] = []
    for page_number, raw in extract_pages(path):
        for piece in split_into_chunks(clean_text(raw)):
            chunks.append({"text": piece, "source": path.name, "page": page_number})
    if not chunks:
        raise DocumentError("no readable text found")
    return chunks


# ------------------------------------------------------------ index building
def build_index(
    data_dir: Path = config.DATA_DIR,
    store_dir: Path = config.VECTOR_STORE_DIR,
    embedder=None,
) -> IngestReport:
    """Read every document, create embeddings and save the FAISS index.

    A file that cannot be read is skipped and listed in report.failed, so one
    bad PDF never stops the others from being indexed.
    """
    from rag import SentenceTransformerEmbedder, VectorStore  # local import: heavy libs

    report = IngestReport()
    all_chunks: List[Dict] = []

    for path in find_documents(data_dir):
        try:
            file_chunks = load_document(path)
        except DocumentError as exc:
            report.failed.append((path.name, str(exc)))
            continue
        report.processed.append((path.name, len(file_chunks)))
        all_chunks.extend(file_chunks)

    if not all_chunks:
        return report  # nothing usable: the old index (if any) is kept

    for number, chunk in enumerate(all_chunks):
        chunk["id"] = number

    embedder = embedder or SentenceTransformerEmbedder()
    vectors = embedder.encode([c["text"] for c in all_chunks])

    store = VectorStore()
    store.build(vectors, all_chunks, getattr(embedder, "model_name", "unknown"))
    store.save(store_dir)

    report.total_chunks = len(all_chunks)
    return report


def main() -> int:
    print(f"Reading documents from: {config.DATA_DIR}")
    try:
        report = build_index()
    except Exception as exc:  # e.g. packages not installed, model download failed
        print(f"Could not build the index: {exc}")
        return 1
    for name, count in report.processed:
        print(f"  OK       {name}: {count} chunks")
    for name, reason in report.failed:
        print(f"  SKIPPED  {name}: {reason}")
    if report.total_chunks:
        print(f"Index saved to {config.VECTOR_STORE_DIR} ({report.total_chunks} chunks).")
        return 0
    print("No readable text was found. The existing index (if any) was not changed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
