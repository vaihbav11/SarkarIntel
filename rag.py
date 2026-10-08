"""
SarkarIntel RAG pipeline.
Handles PDF discovery, text extraction, chunking, embedding, and FAISS retrieval.
"""

import os
import glob
import textwrap
from typing import List, Tuple, Dict

import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Constants ────────────────────────────────────────────────────────────────
DOCS_DIR = os.path.join(os.path.dirname(__file__), "Data", "documents")
CHUNK_SIZE = 400          # characters per chunk (approx)
CHUNK_OVERLAP = 80        # character overlap between consecutive chunks
TOP_K = 5                 # number of chunks to retrieve
EMBED_MODEL = "all-MiniLM-L6-v2"


# ── Text helpers ─────────────────────────────────────────────────────────────

def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


# ── PDF helpers ───────────────────────────────────────────────────────────────

def discover_pdfs(folder: str = DOCS_DIR) -> List[str]:
    """Return a sorted list of all PDF paths inside *folder*."""
    pattern = os.path.join(folder, "*.pdf")
    return sorted(glob.glob(pattern))


def extract_chunks_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract text from every page of a PDF and return a list of chunk dicts.

    Each dict has:
        text      – chunk content
        filename  – basename of the PDF
        page      – 1-based page number
    """
    filename = os.path.basename(pdf_path)
    chunks = []
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text()
            if not page_text.strip():
                continue
            for chunk in _chunk_text(page_text):
                chunks.append({
                    "text": chunk,
                    "filename": filename,
                    "page": page_num,
                })
        doc.close()
    except Exception as exc:
        print(f"[rag] Could not process {filename}: {exc}")
    return chunks


# ── Index builder ─────────────────────────────────────────────────────────────

def build_index(pdf_paths: List[str], model: SentenceTransformer) -> Tuple[faiss.Index, List[Dict]]:
    """
    Build a FAISS flat-L2 index from all chunks extracted from *pdf_paths*.

    Returns (index, metadata_list) where metadata_list[i] corresponds to the
    i-th vector in the index.
    """
    all_chunks: List[Dict] = []
    for path in pdf_paths:
        all_chunks.extend(extract_chunks_from_pdf(path))

    if not all_chunks:
        raise ValueError("No text could be extracted from any PDF.")

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, all_chunks


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    index: faiss.Index,
    metadata: List[Dict],
    model: SentenceTransformer,
    top_k: int = TOP_K,
) -> List[Dict]:
    """Return the top-k most relevant chunk dicts for *query*."""
    q_vec = model.encode([query], show_progress_bar=False)
    q_vec = np.array(q_vec, dtype="float32")
    distances, indices = index.search(q_vec, top_k)
    results = []
    for idx in indices[0]:
        if idx < len(metadata):
            results.append(metadata[idx])
    return results


# ── Upload-document helpers ───────────────────────────────────────────────────

def build_upload_index(pdf_bytes: bytes, filename: str, model: SentenceTransformer) -> Tuple[faiss.Index, List[Dict]]:
    """
    Build an in-memory FAISS index from an uploaded PDF (provided as raw bytes).
    Completely separate from the permanent government dataset.
    """
    chunks: List[Dict] = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text()
            if not page_text.strip():
                continue
            for chunk in _chunk_text(page_text):
                chunks.append({
                    "text": chunk,
                    "filename": filename,
                    "page": page_num,
                })
        doc.close()
    except Exception as exc:
        raise RuntimeError(f"Could not process uploaded PDF: {exc}") from exc

    if not chunks:
        raise ValueError("No text could be extracted from the uploaded PDF.")

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, chunks


def retrieve_from_upload(
    query: str,
    index: faiss.Index,
    metadata: List[Dict],
    model: SentenceTransformer,
    top_k: int = TOP_K,
) -> List[Dict]:
    """Retrieve top-k chunks from an uploaded-document index."""
    return retrieve(query, index, metadata, model, top_k)


# ── LLM call ──────────────────────────────────────────────────────────────────

def build_prompt(query: str, context_chunks: List[Dict], is_upload: bool = False) -> str:
    context_text = "\n\n".join(
        f"[{c['filename']} — Page {c['page']}]\n{c['text']}"
        for c in context_chunks
    )
    source_label = "the uploaded document" if is_upload else "the retrieved government documents"
    return f"""You are SI, an AI assistant for SarkarIntel.
You MUST answer ONLY using the context provided below.
Do NOT use any general knowledge or information outside this context.
Do NOT invent facts, schemes, eligibility criteria, financial amounts, or dates.
If the answer is not present in the context, reply exactly:
"I couldn't find this information in the {'uploaded document' if is_upload else 'available government documents'}."

Context (from {source_label}):
{context_text}

Question: {query}

Answer:"""


def call_llm(prompt: str) -> str:
    """
    Call the configured LLM.  Supports:
      - OpenAI-compatible API  (OPENAI_API_KEY)
      - Google Gemini          (GEMINI_API_KEY)
    Set the relevant environment variable before running.
    """
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    if openai_key:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    if gemini_key:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel(os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"))
        response = model.generate_content(prompt)
        return response.text.strip()

    raise EnvironmentError(
        "No LLM API key found. Set OPENAI_API_KEY or GEMINI_API_KEY as an environment variable."
    )
