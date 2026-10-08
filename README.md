# SarkarIntel — SI

**SarkarIntel** (SI) is a local RAG (Retrieval-Augmented Generation) chatbot that lets you query Indian government scheme documents using a fully local AI model.

> **No OpenAI API key required. No Gemini API key required. No paid cloud API of any kind.**

---

## What is SarkarIntel?

SI answers questions about government schemes, scholarships, and policies by:

1. Extracting text from government PDFs (PyMuPDF)
2. Chunking and embedding text (Sentence Transformers — `all-MiniLM-L6-v2`)
3. Storing embeddings in a FAISS vector index
4. Retrieving the most relevant chunks for each query
5. Sending those chunks to a **locally running Ollama LLM** for a grounded answer
6. Returning the answer with source document and page citations

SI never fabricates information. If the answer isn't in the documents, it says so.

---

## Architecture

```
Government PDFs (data/documents/)
        ↓
PyMuPDF — page-aware text extraction
        ↓
Text chunking (400-char chunks, 80-char overlap)
        ↓
Sentence Transformer embeddings (all-MiniLM-L6-v2)
        ↓
FAISS flat index
        ↓
Top-5 relevant chunks retrieved
        ↓
Local Ollama LLM (llama3.2:3b)
        ↓
Grounded answer + source citations (filename + page)
```

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd SarkarIntel
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama

Download and install from [https://ollama.com](https://ollama.com).

On most systems Ollama starts automatically as a background service.
If it doesn't, start it manually:

```bash
ollama serve
```

### 4. Download the model

```bash
ollama pull llama3.2:3b
```

This downloads the default model (~2 GB). It only needs to be done once.

### 5. Add government PDFs

Place PDF files in:

```
Data/documents/
```

The 15 government scheme PDFs are already included in this folder.
Do **not** rename, move, or delete them.

---

## Running the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Configuration

The model name and Ollama host are configurable via environment variables:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2:3b` | Ollama model to use |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API base URL |

Example (use a different model):

```bash
OLLAMA_MODEL=llama3.1:8b streamlit run app.py
```

No API keys are ever needed.

---

## Features

### Government knowledge base
- Automatically discovers all PDFs in `Data/documents/`
- Indexes them on first run (cached — not re-indexed on every question)
- Answers cite the source filename and page number

### PDF upload
- Upload any PDF from the sidebar to query it independently
- Prefix your question with `upload:` to query the uploaded document
- Uploaded documents are **never** added to the permanent knowledge base
- Uploaded-document answers always start with **"According to the uploaded document:"**

### Grounded answers
- SI only answers from retrieved document context
- If the answer isn't present: *"I couldn't find this information in the available government documents."*
- No hallucination, no outside knowledge

### Error handling
- Ollama not running → clear install/start instructions shown in the UI
- Model not downloaded → instructions to run `ollama pull`
- No PDFs found → warning with folder path
- Uploaded PDF with no readable text → clear error, no crash

---

## Project structure

```
SarkarIntel/
├── app.py                  # Streamlit UI
├── rag.py                  # RAG pipeline (extraction, embeddings, FAISS, Ollama)
├── requirements.txt
├── README.md
├── tests/
│   └── test_retrieval.py   # Retrieval tests
└── Data/
    └── documents/
        └── *.pdf           # Government scheme PDFs (do not modify)
```

---

## Running tests

```bash
pytest tests/
```

Tests verify:
- PDFs are discovered in `Data/documents/`
- Text is extracted from PDFs
- Page metadata (filename + page number) is preserved in every chunk
- FAISS retrieval returns relevant results

> **Note:** The full test suite builds the FAISS index, which may take a minute on first run.

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `pymupdf` | PDF text extraction |
| `sentence-transformers` | Text embeddings |
| `faiss-cpu` | Vector similarity search |
| `pytest` | Testing |

No OpenAI SDK. No Gemini SDK. No paid API.

---

## Included documents

| File | Scheme |
|---|---|
| APY.pdf | Atal Pension Yojana |
| CSSS_GUIDLINES_.pdf | Central Sector Scholarship Scheme |
| DEPDGuidelines.pdf | DEPD Guidelines |
| Guidlines_3099.pdf | Scheme Guidelines |
| NMMSSGuidelines.pdf | National Means-cum-Merit Scholarship |
| PMKVY-4.0-Guidelines.pdf | Pradhan Mantri Kaushal Vikas Yojana 4.0 |
| pm_sva_nidhi_loan_operational_guidelines.pdf | PM SVANidhi Loan |
| pm_sva_nidhi_scheme_guidelines.pdf | PM SVANidhi Scheme |
| SBM(G)_Guidelines.pdf | Swachh Bharat Mission (Grameen) |
| SukanyaSamriddhiAccountScheme2019English.pdf | Sukanya Samriddhi Account |
| + 5 additional government documents | Various schemes |

---

*SarkarIntel — Powered by local AI. No cloud. No API key.*
