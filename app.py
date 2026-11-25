"""Streamlit web interface of the AI-Powered University Student Assistant.

Start with:  streamlit run app.py
"""
from pathlib import Path

import streamlit as st

import config
import ingest
from rag import GeminiLLM, RAGEngine, SentenceTransformerEmbedder

st.set_page_config(page_title="University Student Assistant", page_icon="🎓", layout="wide")

# ---------------------------------------------------------------- styling
st.markdown(
    """
    <style>
    .app-header {
        background: linear-gradient(90deg, #1F3A8A 0%, #6D28D9 100%);
        padding: 1.2rem 1.5rem; border-radius: 14px; color: white; margin-bottom: 1rem;
    }
    .app-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .app-header p  { color: #E0E7FF; margin: 0.3rem 0 0 0; }
    @media (max-width: 640px) { .app-header h1 { font-size: 1.3rem; } }
    </style>
    <div class="app-header">
        <h1>🎓 AI-Powered University Student Assistant</h1>
        <p>Ask about attendance, exams, library, admission, scholarships and more.
           Answers come only from the university documents.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

SAMPLE_QUESTIONS = [
    "How many books can I borrow from the library?",
    "What is the minimum attendance required to write exams?",
    "What is the fine for late return of a library book?",
    "Which documents are needed for admission?",
    "How can I apply for a scholarship?",
]


# ---------------------------------------------------------------- engine
@st.cache_resource(show_spinner="Loading the embedding model (first time only)...")
def get_engine() -> RAGEngine:
    llm = GeminiLLM(config.GEMINI_API_KEY) if config.GEMINI_API_KEY else None
    return RAGEngine(embedder=SentenceTransformerEmbedder(), llm=llm)


engine = get_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_report(report: "ingest.IngestReport") -> None:
    for name, count in report.processed:
        st.success(f"{name}: {count} chunks indexed")
    for name, reason in report.failed:
        st.warning(f"{name} was skipped: {reason}")
    if not report.total_chunks:
        st.error("No readable text was found. The existing index was not changed.")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📂 Documents")
    uploads = st.file_uploader(
        "Upload university PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )
    if st.button("Process documents", use_container_width=True, disabled=not uploads):
        config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        for file in uploads:
            (config.UPLOAD_DIR / Path(file.name).name).write_bytes(file.getbuffer())
        with st.spinner("Reading documents and building the index..."):
            report = ingest.build_index(embedder=engine.embedder)
        engine.reload()
        show_report(report)

    if st.button("Rebuild index", use_container_width=True):
        with st.spinner("Rebuilding the index from the data folder..."):
            report = ingest.build_index(embedder=engine.embedder)
        engine.reload()
        show_report(report)

    st.divider()
    st.subheader("Status")
    if engine.has_index():
        st.write(f"✅ Index ready: **{engine.store.size}** chunks")
    else:
        st.write("⚠️ No index yet. Upload documents and press **Process documents**.")
    if engine.llm is None:
        st.write("🔎 Mode: **document search** (no Gemini API key)")
    else:
        st.write(f"🤖 Mode: **Gemini** ({config.GEMINI_MODEL})")

    top_k = st.slider("Passages to retrieve", min_value=1, max_value=8, value=config.TOP_K)

    st.divider()
    st.subheader("Try a sample question")
    for i, sample in enumerate(SAMPLE_QUESTIONS):
        if st.button(sample, key=f"sample_{i}", use_container_width=True):
            st.session_state.pending_question = sample

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------- chat display
def render_sources(sources) -> None:
    if not sources:
        return
    with st.expander("Sources"):
        for s in sources:
            page = f", page {s['page']}" if s.get("page") else ""
            st.markdown(f"**{s['source']}{page}** (match score {s['score']:.2f})")
            st.caption(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))


if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.write("Hello! I am your university assistant. Ask me a question, or pick a "
                 "sample question from the sidebar.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources"))

# ---------------------------------------------------------------- new question
question = st.chat_input("Type your question here...") or st.session_state.pop("pending_question", None)
if question is not None:
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the university documents..."):
            result = engine.answer(question, history=history, k=top_k)
        st.write(result.text)
        sources = [
            {"source": s.source, "page": s.page, "score": s.score, "text": s.text}
            for s in result.sources
        ]
        render_sources(sources)
    st.session_state.messages.append(
        {"role": "assistant", "content": result.text, "sources": sources}
    )
