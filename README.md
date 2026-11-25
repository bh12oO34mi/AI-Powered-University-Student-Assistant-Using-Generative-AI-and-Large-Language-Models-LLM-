# AI-Powered University Student Assistant (Generative AI + LLM + RAG)

A Streamlit chatbot that answers university questions **only from your own PDF/TXT documents**.
Documents are split into chunks, converted to vectors (Sentence Transformers `all-MiniLM-L6-v2`), stored in a **FAISS** index, and the best passages are sent to **Gemini** to write the answer. Every answer shows its source file and page. If the documents do not contain the answer, a fallback message is shown and the LLM is not asked to guess.

## Setup on Windows (VS Code)

Open this folder in VS Code, then open the terminal (Ctrl + `):

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and replace `your_api_key_here` with your key from Google AI Studio (https://aistudio.google.com). Linux/macOS: use `source venv/bin/activate` and `cp .env.example .env`.

Python 3.10 - 3.12 is recommended. The first run downloads the embedding model (about 90 MB) and needs internet.

## Run

```
python ingest.py          (optional - builds the index from the data/ folder)
streamlit run app.py      (opens http://localhost:8501)
```

You can also upload PDFs in the sidebar and press **Process documents**. Without an API key the app still works in *document-search mode* (it shows the best matching passages).

## Run the tests

```
python -m unittest discover -s tests -v
```

The automated tests use a stand-in embedder and a fake LLM, so they run in seconds without internet. One test uses the real embedding model and is skipped until `sentence-transformers` is installed. Manual test cases are in `tests/manual_test_log.md`.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface (chat, sources, upload, history) |
| `ingest.py` | Read PDF/TXT, clean, chunk, build the FAISS index |
| `rag.py` | Embeddings, vector store, Gemini call, answer logic and fallbacks |
| `config.py` | All settings and messages (chunk size, top-k, threshold, model name) |
| `make_sample_pdf.py` | (optional) regenerates the sample PDF, needs `reportlab` |
| `data/` | Sample (fictional) documents; uploads go to `data/uploads/` |
| `vector_store/` | Saved index (created automatically) |
| `tests/` | Automated tests and the manual test log |
| `documentation/` | Synopsis, report, reference-paper list, demo guide |

## Tuning

If good questions get the fallback message, lower `MIN_SIMILARITY` in `config.py` (default 0.30). If answers miss details, try a larger `TOP_K` or `CHUNK_SIZE`. If Google retires the default model, set `GEMINI_MODEL` in `.env`.

## Notes

- Scanned (image-only) PDFs need OCR and are skipped with a warning.
- Never share your `.env` file; it is listed in `.gitignore`.
- The sample data is fictional. Replace it with your real university documents.
