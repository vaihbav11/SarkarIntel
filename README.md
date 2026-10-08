# SarkarIntel 🇮🇳

**SarkarIntel** is an AI-powered government information retrieval and RAG chatbot.  
Ask questions about Indian government schemes, scholarships, and policies — SI answers using only the actual government documents you provide.

---

## What is SI?

**SI** (SarkarIntel Intelligence) is the chatbot powering SarkarIntel.  
SI retrieves relevant passages from government PDFs and answers your questions based solely on those passages.  
SI never invents facts, schemes, eligibility criteria, financial amounts, or dates.

---

## RAG Architecture

```
Government PDFs (data/documents/)
        │
        ▼
PyMuPDF text extraction  (page-by-page)
        │
        ▼
Text chunking  (~400 chars, 80-char overlap)
        │
        ▼
Sentence Transformer embeddings  (all-MiniLM-L6-v2)
        │
        ▼
FAISS flat-L2 vector index
        │
        ▼
Top-5 relevant chunks retrieved
        │
        ▼
LLM (OpenAI / Gemini)
        │
        ▼
Grounded answer  +  Source: filename — Page N
```

---

## Features

- **Automatic PDF discovery** — detects all `.pdf` files in `Data/documents/` without hardcoding filenames.
- **Page-level metadata** — every answer cites the exact source document and page number.
- **Grounded answers only** — SI refuses to answer from general knowledge; if the answer isn't in the documents, it says so.
- **PDF upload** — upload any PDF and query it independently; uploaded documents are never mixed with the permanent government dataset.
- **Streamlit caching** — documents are indexed once per session; re-asking questions is fast.
- **English-only MVP** — no language selection or translation in this version.

---

## Project Structure

```
SarkarIntel/
├── app.py                  # Streamlit UI
├── rag.py                  # RAG pipeline
├── requirements.txt
├── README.md
├── tests/
│   └── test_retrieval.py
└── Data/
    └── documents/
        └── *.pdf           # Government PDFs (do not rename or move)
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/SarkarIntel.git
cd SarkarIntel

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## API Key Configuration

SarkarIntel requires **one** LLM API key.  
Set it as an environment variable — **never hardcode keys**.

### Option A — OpenAI

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."

# macOS / Linux
export OPENAI_API_KEY="sk-..."
```

To use a specific model (default: `gpt-4o-mini`):

```bash
export OPENAI_MODEL="gpt-4o"
```

### Option B — Google Gemini

```bash
export GEMINI_API_KEY="AIza..."
# Optional: override model (default: gemini-1.5-flash)
export GEMINI_MODEL="gemini-1.5-pro"
```

### Streamlit Community Cloud

Add secrets in **App Settings → Secrets**:

```toml
OPENAI_API_KEY = "sk-..."
# or
GEMINI_API_KEY = "AIza..."
```

---

## Running the Application

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Example Questions

```
What is the income limit for the PM-USP scholarship?
What are the eligibility criteria for the NMMS scholarship?
How much loan can be taken under PM SVANidhi?
What are the objectives of Sukanya Samriddhi Account Scheme?
Who is eligible for DEPD benefits?
What is the lock-in period for APY?
```

---

## PDF Upload Behavior

1. Click **"📎 Upload a PDF to query"** in the sidebar expander.
2. Upload any PDF document.
3. Prefix your question with `upload:` to query the uploaded document only.

**Example:**

```
upload: What are the eligibility requirements?
```

**SI responds:**

```
According to the uploaded document:

<answer from the uploaded PDF>

Source:
my_document.pdf — Page 3
```

**Important rules:**
- Uploaded PDFs are **never** added to `Data/documents/`.
- The permanent government dataset is **not** searched when you use the `upload:` prefix.
- If the answer isn't in the uploaded PDF, SI responds: *"I couldn't find this information in the uploaded document."*

---

## Source Citation Behavior

Every answer includes a **Sources** block:

```
Sources:
NMMSSGuidelines.pdf — Page 4
CSSS_GUIDLINES_.pdf — Page 2
```

- Sources come directly from retrieved chunk metadata.
- Duplicate source references are automatically deduplicated.
- No source is fabricated.

---

## Running Tests

```bash
pytest tests/ -v
```

Tests verify:
1. PDFs are discovered from `Data/documents/`.
2. Text is extractable from PDFs.
3. Page metadata is preserved in every chunk.
4. FAISS retrieval returns relevant results.

---

## Streamlit Community Cloud Deployment

1. Push the repository to GitHub (include `Data/documents/` with your PDFs).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch `main`, main file `app.py`.
4. Under **Advanced settings → Secrets**, add your API key:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
5. Click **Deploy**.

> **Note:** Streamlit Community Cloud has a free-tier memory limit (~1 GB). If your PDFs are large, consider reducing `CHUNK_SIZE` in `rag.py` or using a smaller embedding model.

---

## License

MIT
