"""Central settings for the AI-Powered University Student Assistant.

Every setting lives here so that no other file contains hard-coded values.
The Gemini API key is read from the .env file (never written in the code).
"""
import os
from pathlib import Path

try:  # python-dotenv is optional: without it, normal environment variables are used
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:  # pragma: no cover
    pass

# ---------------------------------------------------------------- folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# ---------------------------------------------------------------- chunking
CHUNK_SIZE = 600      # characters per chunk (about)
CHUNK_OVERLAP = 120   # characters repeated at the start of the next chunk

# ---------------------------------------------------------------- embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-dimension vectors, runs on CPU

# ---------------------------------------------------------------- retrieval
TOP_K = 4              # chunks sent to the LLM
MIN_SIMILARITY = 0.30  # below this score a chunk is treated as "not relevant"
                       # (starting value - tune it on your real documents)

# ---------------------------------------------------------------- LLM (Gemini)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
LLM_TEMPERATURE = 0.2  # low randomness -> factual, consistent answers

_PLACEHOLDER_KEYS = {"", "your_api_key_here", "your_key_here", "paste_your_key_here", "none"}


def _clean_key(raw: str) -> str:
    key = (raw or "").strip().strip('"').strip("'")
    if key.lower() in _PLACEHOLDER_KEYS or key.lower().startswith("your_"):
        return ""      # placeholder from .env.example -> treated as "no key"
    return key


GEMINI_API_KEY = _clean_key(os.getenv("GEMINI_API_KEY", ""))

# ---------------------------------------------------------------- messages
FALLBACK_MESSAGE = (
    "I could not find this information in the university documents. "
    "Please contact the university office for help."
)
EMPTY_MESSAGE = "Please type a question."
NO_INDEX_MESSAGE = (
    "No documents have been processed yet. Please upload university PDF or TXT "
    "files in the sidebar and press 'Process documents'."
)
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
