"""
tests/test_retrieval.py
Basic retrieval tests for SarkarIntel.
"""

import os
import sys
import types

import pytest

# ── Make the project root importable ─────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import rag


# ── 1. PDF discovery ──────────────────────────────────────────────────────────

def test_pdfs_are_discovered():
    """At least one PDF must be found in Data/documents/."""
    pdfs = rag.discover_pdfs()
    assert len(pdfs) > 0, (
        f"No PDFs found in '{rag.DOCS_DIR}'. "
        "Make sure the folder exists and contains .pdf files."
    )


def test_discovered_files_are_pdfs():
    """Every discovered file must have a .pdf extension."""
    pdfs = rag.discover_pdfs()
    for path in pdfs:
        assert path.lower().endswith(".pdf"), f"Non-PDF returned: {path}"


# ── 2. Text extraction ────────────────────────────────────────────────────────

def test_text_extraction_returns_chunks():
    """At least the first PDF should yield extractable text chunks."""
    pdfs = rag.discover_pdfs()
    assert pdfs, "No PDFs to test."
    chunks = rag.extract_chunks_from_pdf(pdfs[0])
    assert len(chunks) > 0, f"No chunks extracted from {pdfs[0]}"


def test_chunks_have_text():
    """Every chunk must contain non-empty text."""
    pdfs = rag.discover_pdfs()
    assert pdfs
    chunks = rag.extract_chunks_from_pdf(pdfs[0])
    for c in chunks:
        assert isinstance(c["text"], str) and len(c["text"].strip()) > 0


# ── 3. Page metadata preservation ────────────────────────────────────────────

def test_page_metadata_present():
    """Every chunk must carry filename and page number."""
    pdfs = rag.discover_pdfs()
    assert pdfs
    chunks = rag.extract_chunks_from_pdf(pdfs[0])
    for c in chunks:
        assert "filename" in c, "Missing 'filename' in chunk metadata"
        assert "page" in c, "Missing 'page' in chunk metadata"
        assert isinstance(c["page"], int) and c["page"] >= 1


def test_filename_matches_source():
    """Chunk filename must match the basename of the source PDF."""
    pdfs = rag.discover_pdfs()
    assert pdfs
    path = pdfs[0]
    chunks = rag.extract_chunks_from_pdf(path)
    expected = os.path.basename(path)
    for c in chunks:
        assert c["filename"] == expected, (
            f"Chunk filename '{c['filename']}' != expected '{expected}'"
        )


# ── 4. FAISS retrieval ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def index_and_meta():
    """Build index once for all retrieval tests."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(rag.EMBED_MODEL)
    pdfs = rag.discover_pdfs()
    index, metadata = rag.build_index(pdfs, model)
    return index, metadata, model


def test_retrieval_returns_results(index_and_meta):
    """Retrieval must return at least one chunk."""
    index, metadata, model = index_and_meta
    results = rag.retrieve("scholarship eligibility", index, metadata, model, top_k=3)
    assert len(results) > 0


def test_retrieval_results_have_metadata(index_and_meta):
    """Every retrieved chunk must have text, filename, and page."""
    index, metadata, model = index_and_meta
    results = rag.retrieve("pension scheme benefits", index, metadata, model, top_k=3)
    for r in results:
        assert "text" in r
        assert "filename" in r
        assert "page" in r


def test_retrieval_top_k_respected(index_and_meta):
    """Retrieval must not return more than top_k results."""
    index, metadata, model = index_and_meta
    results = rag.retrieve("loan eligibility", index, metadata, model, top_k=4)
    assert len(results) <= 4
