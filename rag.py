"""Retrieval-Augmented Generation: embeddings, vector store, Gemini and the answer logic."""
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)

EMPTY_MESSAGE = config.EMPTY_MESSAGE
NO_INDEX_MESSAGE = config.NO_INDEX_MESSAGE


class LLMError(Exception):
    """Any problem while calling the LLM (key, internet, quota, empty answer)."""


class VectorStoreError(Exception):
    """The saved vector store is missing, damaged or made with another model."""


class EmbeddingError(Exception):
    """The embedding model could not be loaded."""


# ------------------------------------------------------------------ data classes
@dataclass
class Source:
    text: str
    source: str
    page: Optional[int]
    score: float


@dataclass
class AnswerResult:
    text: str
    status: str   # empty | no_index | error | fallback | ok | ok_extractive | llm_error
    sources: List[Source] = field(default_factory=list)
    used_llm: bool = False


# ------------------------------------------------------------------ embeddings
class SentenceTransformerEmbedder:
    """Turns texts into normalised 384-dimension vectors (loaded on first use)."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "The 'sentence-transformers' package is not installed. "
                    "Run: pip install -r requirements.txt"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: List[str]) -> np.ndarray:
        model = self._load()
        vectors = model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype="float32")


# ------------------------------------------------------------------ vector store
class VectorStore:
    """FAISS IndexFlatIP over normalised vectors (inner product = cosine similarity).

    If FAISS is not installed an equivalent NumPy search is used.
    """

    def __init__(self):
        self.chunks: List[Dict] = []
        self.meta: Dict = {}
        self._vectors: Optional[np.ndarray] = None
        self._index = None

    @property
    def size(self) -> int:
        return len(self.chunks)

    # -------------------------------------------------------------- build / save / load
    def _make_index(self):
        try:
            import faiss
        except ImportError:
            self._index = None
            return
        index = faiss.IndexFlatIP(self._vectors.shape[1])
        index.add(self._vectors)
        self._index = index

    def build(self, vectors: np.ndarray, chunks: List[Dict], model_name: str = "unknown") -> None:
        vectors = np.ascontiguousarray(np.asarray(vectors, dtype="float32"))
        if vectors.ndim != 2 or len(vectors) != len(chunks):
            raise VectorStoreError("The number of vectors must equal the number of chunks.")
        self._vectors = vectors
        self.chunks = list(chunks)
        self.meta = {
            "embedding_model": model_name,
            "dimension": int(vectors.shape[1]) if len(vectors) else 0,
            "num_chunks": len(chunks),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._make_index()

    def save(self, store_dir: Path = config.VECTOR_STORE_DIR) -> None:
        store_dir = Path(store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)
        np.save(store_dir / "vectors.npy", self._vectors)
        with open(store_dir / "chunks.json", "w", encoding="utf-8") as handle:
            json.dump({"meta": self.meta, "chunks": self.chunks}, handle, ensure_ascii=False, indent=2)
        if self._index is not None:
            import faiss
            faiss.write_index(self._index, str(store_dir / "faiss.index"))

    @classmethod
    def load(cls, store_dir: Path = config.VECTOR_STORE_DIR) -> "VectorStore":
        store_dir = Path(store_dir)
        chunks_file = store_dir / "chunks.json"
        vectors_file = store_dir / "vectors.npy"
        if not chunks_file.exists() or not vectors_file.exists():
            raise VectorStoreError("No saved vector store found.")
        try:
            with open(chunks_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            vectors = np.load(vectors_file)
        except Exception as exc:
            raise VectorStoreError(f"The saved vector store is damaged: {exc}") from exc
        chunks = data.get("chunks", [])
        if len(chunks) != len(vectors):
            raise VectorStoreError(
                "The saved vector store is inconsistent (vectors and chunks differ). "
                "Please rebuild the index."
            )
        store = cls()
        store.chunks = chunks
        store.meta = data.get("meta", {})
        store._vectors = np.ascontiguousarray(vectors.astype("float32"))
        store._make_index()   # rebuilt from vectors.npy: always consistent
        return store

    # -------------------------------------------------------------- search
    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[Dict, float]]:
        """Return up to k (chunk, similarity_score) pairs, best first."""
        if self.size == 0 or self._vectors is None:
            return []
        query = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        if query.shape[1] != self._vectors.shape[1]:
            raise VectorStoreError(
                "The saved index was created with a different embedding model. "
                "Please rebuild the index (run ingest.py or press the Rebuild button)."
            )
        k = max(1, min(int(k), self.size))
        if self._index is not None:
            scores, ids = self._index.search(query, k)
            pairs = list(zip(ids[0], scores[0]))
        else:  # NumPy fallback (same maths, no FAISS)
            sims = self._vectors @ query[0]
            order = np.argsort(-sims)[:k]
            pairs = [(i, sims[i]) for i in order]
        return [(self.chunks[int(i)], float(s)) for i, s in pairs if int(i) >= 0]


# ------------------------------------------------------------------ LLM
class GeminiLLM:
    """Thin wrapper around the Gemini API (google-genai package)."""

    def __init__(self, api_key: str, model: str = config.GEMINI_MODEL):
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> str:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMError(
                "The 'google-genai' package is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc
        try:
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=config.LLM_TEMPERATURE),
            )
            text = (response.text or "").strip()
        except Exception as exc:  # invalid key, no internet, quota, etc.
            raise LLMError(str(exc)) from exc
        if not text:
            raise LLMError("The model returned an empty response.")
        return text


# ------------------------------------------------------------------ prompt
PROMPT_TEMPLATE = """You are the AI Student Assistant of a university.

RULES
1. Answer ONLY from the CONTEXT below. Do not use outside knowledge.
2. Never invent rules, dates, fees, names, numbers or policies.
3. If the CONTEXT does not contain the answer, reply with exactly this sentence and nothing else:
{fallback}
4. Use simple, clear English. Be short and direct.
5. Mention the source number(s) you used, for example (Source 1).

CONTEXT
{context}

{history}STUDENT QUESTION
{question}

ANSWER"""


def build_prompt(question: str, sources: List[Source], history: Optional[List[Dict]] = None) -> str:
    """Create the text that is sent to the LLM."""
    blocks = []
    for i, s in enumerate(sources, start=1):
        page = f", page {s.page}" if s.page else ""
        blocks.append(f"[Source {i}: {s.source}{page}]\n{s.text}")
    context = "\n\n".join(blocks)

    history_text = ""
    if history:
        lines = []
        for m in history[-4:]:  # only the last few messages
            who = "Student" if m.get("role") == "user" else "Assistant"
            lines.append(f"{who}: {str(m.get('content', ''))[:500]}")
        history_text = (
            "RECENT CONVERSATION (for understanding follow-up questions only):\n"
            + "\n".join(lines) + "\n\n"
        )

    return PROMPT_TEMPLATE.format(
        fallback=config.FALLBACK_MESSAGE,
        context=context,
        history=history_text,
        question=question,
    )


# ------------------------------------------------------------------ engine
class RAGEngine:
    """Connects validation, retrieval, threshold check, prompt, LLM and error handling."""

    def __init__(
        self,
        embedder=None,
        llm=None,
        store: Optional[VectorStore] = None,
        store_dir: Path = config.VECTOR_STORE_DIR,
        min_similarity: float = config.MIN_SIMILARITY,
        top_k: int = config.TOP_K,
    ):
        self.embedder = embedder
        self.llm = llm
        self.store_dir = Path(store_dir)
        self.min_similarity = min_similarity
        self.top_k = top_k
        self.store = store
        self.load_error: Optional[str] = None
        if self.store is None:
            self.reload()

    # -------------------------------------------------------------- index handling
    def reload(self) -> None:
        """(Re)load the saved vector store from disk."""
        try:
            self.store = VectorStore.load(self.store_dir)
            self.load_error = None
        except VectorStoreError as exc:
            self.store = None
            self.load_error = str(exc)

    def has_index(self) -> bool:
        return self.store is not None and self.store.size > 0

    # -------------------------------------------------------------- retrieval
    def retrieve(self, question: str, k: Optional[int] = None) -> List[Source]:
        if self.embedder is None:
            self.embedder = SentenceTransformerEmbedder()
        vector = self.embedder.encode([question])[0]
        hits = self.store.search(vector, k or self.top_k)
        return [
            Source(text=c["text"], source=c.get("source", "unknown"), page=c.get("page"), score=s)
            for c, s in hits
        ]

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _extractive_answer(sources: List[Source], prefix: str = "", limit: int = 2) -> str:
        parts = []
        for s in sources[:limit]:
            page = f", page {s.page}" if s.page else ""
            parts.append(f"{s.text}\n(Source: {s.source}{page})")
        return prefix + "\n\n".join(parts)

    # -------------------------------------------------------------- answer
    def answer(self, question: str, history: Optional[List[Dict]] = None,
               k: Optional[int] = None) -> AnswerResult:
        question = (question or "").strip()
        if not question:
            return AnswerResult(EMPTY_MESSAGE, "empty")
        if not self.has_index():
            return AnswerResult(NO_INDEX_MESSAGE, "no_index")

        try:
            retrieved = self.retrieve(question, k)
        except Exception as exc:
            logger.exception("Retrieval failed")
            return AnswerResult(f"Sorry, searching the documents failed: {exc}", "error")

        # Keep only chunks that are similar enough to the question.
        relevant = [s for s in retrieved if s.score >= self.min_similarity]
        if not relevant:
            return AnswerResult(config.FALLBACK_MESSAGE, "fallback")

        # No API key -> show the best matching passages without the LLM.
        if self.llm is None:
            return AnswerResult(self._extractive_answer(relevant, prefix=(
                "(Document-search mode: no Gemini API key is configured.) "
                "Here is the most relevant information from the university documents:\n\n"
            )), "ok_extractive", relevant, used_llm=False)

        try:
            text = self.llm.generate(build_prompt(question, relevant, history))
        except LLMError as exc:
            logger.warning("LLM error: %s", exc)
            message = (
                "The AI service could not be reached, so I cannot write a full answer "
                f"right now (reason: {exc}). Here is the most relevant information "
                "found in the university documents:\n\n"
            )
            return AnswerResult(self._extractive_answer(relevant, prefix=message),
                                "llm_error", relevant, used_llm=False)
        except Exception as exc:  # any unexpected problem
            logger.exception("Unexpected LLM failure")
            return AnswerResult(
                f"Sorry, something went wrong while generating the answer: {exc}",
                "error", relevant)

        # If the model itself said the information is unavailable, show no sources.
        if text.strip() == config.FALLBACK_MESSAGE:
            return AnswerResult(text.strip(), "fallback", [], used_llm=True)
        return AnswerResult(text, "ok", relevant, used_llm=True)
