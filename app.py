"""
SarkarIntel — app.py
Streamlit front-end for the SI chatbot.
"""

import os
import streamlit as st
from sentence_transformers import SentenceTransformer

import rag

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SarkarIntel — SI",
    page_icon="🇮🇳",
    layout="centered",
)

# ── Minimal CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Tricolor accent strip at top */
    .block-container { padding-top: 1.5rem; }
    .tricolor-bar {
        display: flex; height: 5px; width: 100%;
        margin-bottom: 0.5rem;
    }
    .tricolor-bar div:nth-child(1) { flex: 1; background: #FF9933; }
    .tricolor-bar div:nth-child(2) { flex: 1; background: #FFFFFF; border-top: 1px solid #ccc; border-bottom: 1px solid #ccc; }
    .tricolor-bar div:nth-child(3) { flex: 1; background: #138808; }

    .brand { font-size: 1rem; font-weight: 700; color: #1a1a1a; }
    .hero-title {
        font-size: 2.4rem; font-weight: 800; letter-spacing: -0.5px;
        color: #1a1a1a; margin-bottom: 0.2rem;
    }
    .hero-sub {
        font-size: 1.1rem; color: #555; margin-bottom: 1.5rem;
    }
    .source-box {
        background: #f7f8fa; border-left: 3px solid #138808;
        padding: 0.5rem 0.8rem; border-radius: 4px;
        font-size: 0.85rem; color: #444; margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Tricolor bar ──────────────────────────────────────────────────────────────
st.markdown(
    '<div class="tricolor-bar"><div></div><div></div><div></div></div>',
    unsafe_allow_html=True,
)

# ── Top-left brand ────────────────────────────────────────────────────────────
st.markdown('<span class="brand">🇮🇳 SI</span>', unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">SARKARINTEL</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">What Intel do you want from SARKAR?</div>',
    unsafe_allow_html=True,
)

st.divider()


# ── Cached model & index ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_model() -> SentenceTransformer:
    return SentenceTransformer(rag.EMBED_MODEL)


@st.cache_resource(show_spinner="Indexing government documents…")
def load_govt_index(_model: SentenceTransformer):
    pdfs = rag.discover_pdfs()
    if not pdfs:
        return None, []
    index, metadata = rag.build_index(pdfs, _model)
    return index, metadata


model = load_model()
govt_index, govt_meta = load_govt_index(model)

if govt_index is None:
    st.warning(
        "⚠️ No PDF documents found in `Data/documents/`. "
        "Please add government PDFs to that folder and restart."
    )

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "upload_index" not in st.session_state:
    st.session_state.upload_index = None
    st.session_state.upload_meta = []
    st.session_state.upload_filename = ""


# ── PDF Upload ────────────────────────────────────────────────────────────────
with st.expander("📎 Upload a PDF to query", expanded=False):
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Uploaded documents are queried independently and never added to the permanent dataset.",
    )
    if uploaded_file is not None:
        fname = uploaded_file.name
        if fname != st.session_state.upload_filename:
            with st.spinner(f"Indexing {fname}…"):
                try:
                    u_index, u_meta = rag.build_upload_index(
                        uploaded_file.read(), fname, model
                    )
                    st.session_state.upload_index = u_index
                    st.session_state.upload_meta = u_meta
                    st.session_state.upload_filename = fname
                    st.success(f"✅ '{fname}' indexed. Ask questions about it below.")
                except Exception as e:
                    st.error(f"Could not process uploaded PDF: {e}")

    if st.session_state.upload_filename:
        st.info(
            f"📄 Active upload: **{st.session_state.upload_filename}** — "
            "questions prefixed with `upload:` will query this document only."
        )
        if st.button("Clear uploaded document"):
            st.session_state.upload_index = None
            st.session_state.upload_meta = []
            st.session_state.upload_filename = ""
            st.rerun()

st.markdown("")


# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown(
                "<div class='source-box'><b>Sources:</b><br>"
                + "<br>".join(msg["sources"])
                + "</div>",
                unsafe_allow_html=True,
            )


# ── Helper: deduplicated source lines ────────────────────────────────────────
def format_sources(chunks: list) -> list:
    seen = set()
    lines = []
    for c in chunks:
        key = (c["filename"], c["page"])
        if key not in seen:
            seen.add(key)
            lines.append(f"{c['filename']} — Page {c['page']}")
    return lines


# ── Chat input ────────────────────────────────────────────────────────────────
upload_hint = " (prefix `upload:` to query uploaded doc)" if st.session_state.upload_filename else ""
placeholder = f"Ask a question about government schemes{upload_hint}…"

if prompt := st.chat_input(placeholder):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Decide: query uploaded doc or govt dataset?
    is_upload_query = (
        st.session_state.upload_index is not None
        and prompt.lower().startswith("upload:")
    )

    answer = ""
    sources = []

    with st.chat_message("assistant"):
        with st.spinner("SI is retrieving…"):
            try:
                if is_upload_query:
                    clean_query = prompt[len("upload:"):].strip()
                    chunks = rag.retrieve_from_upload(
                        clean_query,
                        st.session_state.upload_index,
                        st.session_state.upload_meta,
                        model,
                    )
                    rag_prompt = rag.build_prompt(clean_query, chunks, is_upload=True)
                    raw_answer = rag.call_llm(rag_prompt)
                    # Prefix answer with uploaded-document label
                    if not raw_answer.lower().startswith("i couldn"):
                        answer = "**According to the uploaded document:**\n\n" + raw_answer
                    else:
                        answer = "I couldn't find this information in the uploaded document."
                    sources = format_sources(chunks)
                elif govt_index is not None:
                    chunks = rag.retrieve(prompt, govt_index, govt_meta, model)
                    rag_prompt = rag.build_prompt(prompt, chunks, is_upload=False)
                    answer = rag.call_llm(rag_prompt)
                    sources = format_sources(chunks)
                else:
                    answer = (
                        "No government documents are indexed. "
                        "Please add PDFs to `Data/documents/` and restart."
                    )
            except EnvironmentError as e:
                answer = f"⚠️ Configuration error: {e}"
            except Exception as e:
                answer = f"⚠️ An error occurred: {e}"

        st.markdown(answer)
        if sources:
            st.markdown(
                "<div class='source-box'><b>Sources:</b><br>"
                + "<br>".join(sources)
                + "</div>",
                unsafe_allow_html=True,
            )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
