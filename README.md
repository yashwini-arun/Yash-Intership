# 🔬 BugIQ — Intelligent Bug Analyzer

---
## 📌 What is BugIQ?

BugIQ is an AI-powered debugging tool that:

1. Takes any Python error or stack trace as input
2. Searches a knowledge base using **two retrievers** — BM25 (keyword) + FAISS (semantic)
3. Fuses results using **Reciprocal Rank Fusion (RRF)** to get the most relevant context
4. Sends the fused context to **Llama 3.3 70B** on Groq for diagnosis
5. Returns a structured report with root cause, 3 fixes with confidence levels, and fully corrected code

---

## 🖥️ Demo

```
Input:
  Traceback (most recent call last):
    File "users.py", line 34, in get_profile
      print(data['age'])
  KeyError: 'age'

Output:
  🔴 ROOT CAUSE: The key 'age' does not exist in the dictionary.
  📁 SUSPICIOUS LOCATION: users.py, line 34, in get_profile
  ✅ FIX #1 [Confidence: HIGH]: Use dict.get() instead of direct key access...
  ✅ FIX #2 [Confidence: MEDIUM]: Validate keys before accessing...
  ✅ FIX #3 [Confidence: LOW]: Use a defaultdict...
  ⚠️  RELATED ISSUES TO WATCH: Check all dict accesses in this file
  💻 CORRECTED CODE: <full function rewritten>
```

---

## ⚙️ How It Works

```
Your Error Input
      │
      ▼
┌─────────────────────────────────────┐
│           retriever.py              │
│                                     │
│  BM25 ──────────────► top 10 docs  │
│  (keyword match)                    │
│                                     │
│  FAISS ─────────────► top 10 docs  │
│  (semantic match)                   │
│                                     │
│  EnsembleRetriever                  │
│  (RRF Fusion) ──────► top 5 docs   │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│             app.py                  │
│                                     │
│  Prompt = error + top 5 chunks      │
│  → Llama 3.3 70B via Groq           │
│  → Structured Diagnosis Report      │
└─────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- A free Groq API key → [console.groq.com](https://console.groq.com)

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/bug-analyzer.git
cd bug-analyzer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_actual_key_here
```

### 4. Populate the knowledge base

```bash
python download_data.py
```

This auto-generates:
- `data/stackoverflow.txt` — 20 Python errors with causes, fixes, and examples
- `data/github_issues.csv` — 180 real closed issues from `psf/requests`, `pallets/flask`, `scrapy/scrapy`

### 5. Build the indexes

```bash
python ingest.py
```

This creates:
- `faiss_index/` — vector index for semantic search
- `chunks.pkl` — text chunks for BM25 keyword search

### 6. Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `langchain` | Core framework — loaders, splitters, chains |
| `langchain-community` | FAISS, BM25, HuggingFace integrations |
| `langchain-groq` | Groq API wrapper for LangChain |
| `faiss-cpu` | Vector database for semantic search |
| `rank-bm25` | BM25 keyword search algorithm |
| `sentence-transformers` | Local embedding model (no API key needed) |
| `streamlit` | Web UI framework |
| `groq` | Official Groq Python client |
| `python-dotenv` | Load API key from `.env` file |

---

## 🧠 Tech Stack Explained

### Why RRF instead of a single retriever?

| Retriever | Strength | Weakness |
|---|---|---|
| BM25 (keyword) | Exact error names, function names | Misses related concepts |
| FAISS (semantic) | Related concepts, similar meaning | Misses exact keywords |
| **RRF (fusion)** | **Best of both — keyword + semantic** | — |

RRF formula:
```
score(doc) = Σ  1 / (k + rank_i)
```
Where `k=60` prevents top-ranked docs from dominating, and `rank_i` is the document's position in each retriever's result list.

### Embedding Model

Uses `sentence-transformers/all-MiniLM-L6-v2` — runs **locally on CPU**, completely free, no API key needed. Converts text to 384-dimensional vectors.

### LLM

Uses **Llama 3.3 70B** via Groq's inference API — free tier available, extremely fast (~1-2 seconds for full diagnosis).

---

## 📈 Improving Results

### Add more errors to the knowledge base

Open `data/stackoverflow.txt` and add entries in this format:

```
ERROR: Your error type here
CAUSE: Why it happens
FIX: How to fix it
EXAMPLE:
  # Bad
  <broken code>
  # Good
  <fixed code>
SOURCE: stackoverflow.com/questions/xxxxx

---
```

Then re-run `python ingest.py`.

### Add your own codebase

```bash
# Windows (PowerShell)
Get-ChildItem -Recurse -Filter *.py | Get-Content | Out-File data\my_code.txt

# Mac / Linux
find . -name "*.py" -not -path "./__pycache__/*" | xargs cat > data/my_code.txt

# Then re-index
python ingest.py
```

This lets BugIQ find the exact suspicious location in your own code.

### Confidence level guide

| Confidence | Meaning |
|---|---|
| HIGH | Error exactly matches your `stackoverflow.txt` — BM25 found exact keywords |
| MEDIUM | Error is conceptually similar to known bugs — FAISS found semantic match |
| LOW | Error not in data at all — LLM using its own training knowledge |

---




