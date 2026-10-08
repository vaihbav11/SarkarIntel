"""
SarkarIntel RAG pipeline.
Handles PDF discovery, text extraction, chunking, embedding, and FAISS retrieval.
Uses Ollama for local LLM inference — no API keys required.
"""

import os
import glob
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


# ── LLM call (Ollama — local, no API key required) ───────────────────────────

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_HOST  = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

SYSTEM_PROMPT = (
    "You are SI, the AI assistant for SarkarIntel. "
    "Answer the user's question ONLY using the supplied document context. "
    "Do not use outside knowledge. "
    "Do not invent government schemes, eligibility requirements, financial amounts, "
    "dates, deadlines, authorities, or procedures. "
    "If the supplied context does not contain enough information to answer the question, say: "
    "'I couldn't find this information in the available government documents.' "
    "Always cite the source document and page number for every piece of information you use."
)


def build_context(context_chunks: List[Dict]) -> str:
    return "\n\n".join(
        f"[{c['filename']} — Page {c['page']}]\n{c['text']}"
        for c in context_chunks
    )


def call_ollama(query: str, context_chunks: List[Dict], is_upload: bool = False) -> str:
    """
    Call the local Ollama service.
    Returns the model response, or a user-friendly error string.
    No API key required.
    """
    import urllib.request
    import urllib.error
    import json

    not_found_msg = (
        "I couldn't find this information in the uploaded document."
        if is_upload else
        "I couldn't find this information in the available government documents."
    )

    context_text = build_context(context_chunks)
    source_label = "the uploaded document" if is_upload else "the government documents"

    user_message = (
        f"Context (from {source_label}):\n{context_text}\n\n"
        f"Question: {query}"
    )

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            return data["message"]["content"].strip()
    except urllib.error.URLError:
        raise OllamaUnavailableError(
            "SI's local AI model is unavailable. "
            "Please install Ollama (https://ollama.com) and run: "
            f"ollama pull {OLLAMA_MODEL}"
        )
    except (KeyError, json.JSONDecodeError) as exc:
        raise OllamaUnavailableError(
            f"Unexpected response from Ollama: {exc}. "
            f"Make sure the model '{OLLAMA_MODEL}' is downloaded: "
            f"ollama pull {OLLAMA_MODEL}"
        )


class OllamaUnavailableError(Exception):
    """Raised when Ollama is not running or the model is missing."""
