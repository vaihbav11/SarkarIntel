"""
SarkarIntel — app.py
Streamlit front-end for the SI chatbot.
Powered by local Ollama LLM — no API key required.
"""

import os
import streamlit as st
from sentence_transformers import SentenceTransformer

import rag

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SarkarIntel — SI",
    page_icon="🇮🇳",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Layout ── */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 860px; }

/* ── Tricolor accent bar ── */
.tricolor-bar { display:flex; height:4px; width:100%; margin-bottom:1.2rem; border-radius:2px; overflow:hidden; }
.tricolor-bar div:nth-child(1){ flex:1; background:#FF9933; }
.tricolor-bar div:nth-child(2){ flex:1; background:#ffffff; }
.tricolor-bar div:nth-child(3){ flex:1; background:#138808; }

/* ── Brand ── */
.si-brand { font-size:0.95rem; font-weight:700; color:#138808; letter-spacing:0.04em; }
.hero-title {
    font-size:2.6rem; font-weight:900; letter-spacing:-1px;
    background: linear-gradient(90deg,#FF9933 0%,#1a1a1a 40%,#138808 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; margin-bottom:0.15rem;
}
.hero-sub { font-size:1.15rem; color:#444; margin-bottom:0.6rem; font-style:italic; }
.hero-desc { font-size:0.92rem; color:#666; margin-bottom:1.4rem; line-height:1.6; }

/* ── Chat messages ── */
.stChatMessage [data-testid="stChatMessageContent"] p { font-size:0.97rem; line-height:1.65; }

/* ── Source box ── */
.source-box {
    background:#f7f9f7; border-left:3px solid #138808;
    padding:0.55rem 0.9rem; border-radius:5px;
    font-size:0.82rem; color:#444; margin-top:0.6rem;
    line-height:1.7;
}
.source-box b { color:#138808; font-size:0.84rem; }
.source-rule { border:none; border-top:1px solid #d0d7d0; margin:0.3rem 0; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background:#fafafa; }
.sidebar-brand { font-size:1.1rem; font-weight:800; color:#1a1a1a; margin-bottom:0.1rem; }
.sidebar-sub { font-size:0.78rem; color:#888; margin-bottom:1rem; }
.kb-badge {
    display:inline-block; background:#e8f5e9; color:#2e7d32;
    border:1px solid #a5d6a7; border-radius:12px;
    font-size:0.78rem; padding:0.15rem 0.6rem; margin-bottom:1rem;
}
.upload-note { font-size:0.78rem; color:#888; margin-top:0.4rem; line-height:1.5; }
</style>
""", unsafe_allow_html=True)


# ── Cached model & index ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_model() -> SentenceTransformer:
    return SentenceTransformer(rag.EMBED_MODEL)


@st.cache_resource(show_spinner="Indexing government documents…")
def load_govt_index(_model: SentenceTransformer):
    pdfs = rag.discover_pdfs()
    if not pdfs:
        return None, [], 0
    index, metadata = rag.build_index(pdfs, _model)
    return index, metadata, len(pdfs)


model = load_model()
govt_index, govt_meta, pdf_count = load_govt_index(model)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "upload_index" not in st.session_state:
    st.session_state.upload_index = None
    st.session_state.upload_meta = []
    st.session_state.upload_filename = ""


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="tricolor-bar"><div></div><div></div><div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-brand">🇮🇳 SI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">SarkarIntel</div>', unsafe_allow_html=True)

    st.markdown("**Knowledge Base**")
    doc_label = f"{pdf_count} government document{'s' if pdf_count != 1 else ''}" if pdf_count else "No documents found"
    st.markdown(f'<span class="kb-badge">📚 {doc_label}</span>', unsafe_allow_html=True)

    st.divider()

    st.markdown("**Upload a Document**")
    st.markdown(
        '<p class="upload-note">Upload a PDF to query it independently. '
        "Uploaded documents are <em>not</em> added to SarkarIntel's permanent knowledge base.</p>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        fname = uploaded_file.name
        if fname != st.session_state.upload_filename:
            with st.spinner(f"Indexing {fname}…"):
                try:
                    u_index, u_meta = rag.build_upload_index(uploaded_file.read(), fname, model)
                    st.session_state.upload_index = u_index
                    st.session_state.upload_meta = u_meta
                    st.session_state.upload_filename = fname
                    st.success(f"✅ '{fname}' ready.")
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Could not process PDF: {e}")

    if st.session_state.upload_filename:
        st.info(f"📄 **{st.session_state.upload_filename}**\nPrefix questions with `upload:` to query this file.")
        if st.button("Remove uploaded document"):
            st.session_state.upload_index = None
            st.session_state.upload_meta = []
            st.session_state.upload_filename = ""
            st.rerun()

    st.divider()

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="tricolor-bar"><div></div><div></div><div></div></div>',
    unsafe_allow_html=True,
)
st.markdown('<span class="si-brand">🇮🇳 SI &nbsp;·&nbsp; SarkarIntel</span>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">SARKARINTEL</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">What Intel do you want from SARKAR?</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-desc">Ask questions about government schemes, scholarships and policies '
    'using verified government documents. Powered by a local AI model — no cloud API required.</div>',
    unsafe_allow_html=True,
)

if govt_index is None:
    st.warning(
        "⚠️ No PDF documents found in `Data/documents/`. "
        "Please add government PDFs to that folder and restart the app."
    )


# ── Helper: deduplicated source lines ────────────────────────────────────────
def format_sources(chunks: list) -> list:
    seen = set()
    lines = []
    for c in chunks:
        key = (c["filename"], c["page"])
        if key not in seen:
            seen.add(key)
            lines.append(f"📄 **{c['filename']}** — Page {c['page']}")
    return lines


def render_sources(sources: list):
    if not sources:
        return
    source_html = (
        "<div class='source-box'>"
        "<b>Sources</b><hr class='source-rule'>"
        + "<br>".join(sources)
        + "</div>"
    )
    st.markdown(source_html, unsafe_allow_html=True)


# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])


# ── Chat input ────────────────────────────────────────────────────────────────
upload_hint = " (or prefix `upload:` to query uploaded doc)" if st.session_state.upload_filename else ""
placeholder = f"Ask SI about a government scheme…{upload_hint}"

if prompt := st.chat_input(placeholder):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    is_upload_query = (
        st.session_state.upload_index is not None
        and prompt.lower().startswith("upload:")
    )

    answer = ""
    sources = []

    with st.chat_message("assistant"):
        with st.spinner("SI is thinking…"):
            try:
                if is_upload_query:
                    clean_query = prompt[len("upload:"):].strip()
                    chunks = rag.retrieve_from_upload(
                        clean_query,
                        st.session_state.upload_index,
                        st.session_state.upload_meta,
                        model,
                    )
                    raw_answer = rag.call_ollama(clean_query, chunks, is_upload=True)
                    if not raw_answer.lower().startswith("i couldn"):
                        answer = "**According to the uploaded document:**\n\n" + raw_answer
                    else:
                        answer = "I couldn't find this information in the uploaded document."
                    sources = format_sources(chunks)

                elif govt_index is not None:
                    chunks = rag.retrieve(prompt, govt_index, govt_meta, model)
                    answer = rag.call_ollama(prompt, chunks, is_upload=False)
                    sources = format_sources(chunks)

                else:
                    answer = (
                        "⚠️ No government documents are indexed. "
                        "Please add PDFs to `Data/documents/` and restart the app."
                    )

            except rag.OllamaUnavailableError as e:
                answer = (
                    f"⚠️ **Local AI model is not running.**\n\n{e}\n\n"
                    "**Quick fix:**\n"
                    "1. Install Ollama from https://ollama.com\n"
                    f"2. Run: `ollama pull {rag.OLLAMA_MODEL}`\n"
                    "3. Ollama starts automatically on most systems, or run `ollama serve`\n"
                    "4. Refresh this page and try again."
                )
            except ValueError as e:
                answer = f"⚠️ {e}"
            except Exception as e:
                answer = f"⚠️ An unexpected error occurred. Please try again.\n\n*Details: {e}*"

        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
