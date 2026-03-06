

## 📌 BUG IQ

**BugIQ** is an AI-powered debugging tool that combines hybrid retrieval-augmented generation (RAG) with Cross-Encoder reranking to diagnose Python errors with high accuracy.

Unlike simple LLM chatbots, BugIQ retrieves the most relevant bug context from a curated knowledge base before generating a diagnosis — resulting in precise, evidence-backed fixes rather than generic suggestions.

```
Input  →  BM25 + FAISS Retrieval  →  Cross-Encoder Reranking  →  Llama 3.3  →  Diagnosis + Fixed Code
```

---

## ✨ Features

- 🔍 **Hybrid Retrieval** — BM25 keyword search + FAISS semantic search run in parallel
- 🎯 **Cross-Encoder Reranking** — scores each (query, chunk) pair together for precise relevance
- 🧠 **Structured Diagnosis** — root cause, suspicious location, 3 ranked fixes with confidence levels
- 💻 **Corrected Code** — entire function rewritten with fixes applied and comments on every changed line
- 📊 **Live Reranking Visualization** — see exactly which chunks were retrieved and how they were reranked
- ⚡ **Fast Inference** — Llama 3.3 70B via Groq (~1–2 seconds)
- 🆓 **Fully Free** — no paid APIs needed (Groq free tier + local Cross-Encoder model)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        data/ folder                             │
│   stackoverflow.txt (20 errors)  +  github_issues.csv (180)    │
└────────────────────┬────────────────────────────────────────────┘
                     │ python ingest.py (run once)
          ┌──────────┴──────────┐
          ▼                     ▼
    faiss_index/           chunks.pkl
  (vector index)        (BM25 keyword index)
          │                     │
          └──────────┬──────────┘
                     │
             Your Error Input
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   FAISS Retriever        BM25 Retriever
   (semantic top-10)    (keyword top-10)
          │                     │
          └──────────┬──────────┘
                     ▼
           Merge (~20 unique docs)
                     │
                     ▼
         Cross-Encoder Reranking
      (scores each query-doc pair)
                     │
                     ▼
              Top 5 Chunks
                     │
                     ▼
         Llama 3.3 70B via Groq
                     │
                     ▼
    ┌────────────────────────────┐
    │  🔴 Root Cause             │
    │  📁 Suspicious Location    │
    │  ✅ Fix #1 [HIGH]          │
    │  ✅ Fix #2 [MEDIUM]        │
    │  ✅ Fix #3 [LOW]           │
    │  ⚠️  Related Issues        │
    │  💻 Corrected Code         │
    └────────────────────────────┘
```

---


## 🚀 Getting Started

### Prerequisites

| Requirement | Details |
|---|---|
| Python | 3.9 or higher |
| Groq API Key | Free at [console.groq.com](https://console.groq.com) |
| Disk Space | ~200MB (Cross-Encoder model download, one-time) |

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/bug-analyzer.git
cd bug-analyzer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_actual_key_here
```

> Get your free key at [console.groq.com](https://console.groq.com) — no credit card required.

### 4. Populate the knowledge base

```bash
python download_data.py
```

Creates:
- `data/stackoverflow.txt` — 20 Python errors with structured cause/fix/example entries
- `data/github_issues.csv` — 180 real closed issues from `psf/requests`, `pallets/flask`, `scrapy/scrapy`

### 5. Build the indexes

```bash
python ingest.py
```

Creates:
- `faiss_index/` — FAISS vector index for semantic search
- `chunks.pkl` — text chunks for BM25 keyword search

> ⚠️ Re-run this every time you add new files to `data/`

### 6. Run the app

```bash
streamlit run app.py
```

Visit [http://localhost:8501](http://localhost:8501)

> **Note:** On first run, the Cross-Encoder model (~90MB) will download automatically. This is a one-time download.

---

## 🧪 Sample Errors to Test

```python
# Test 1 — KeyError
Traceback (most recent call last):
  File "users.py", line 34, in get_age
    print(data['age'])
KeyError: 'age'

# Test 2 — ZeroDivisionError
Traceback (most recent call last):
  File "math_ops.py", line 17, in compute
    result = 100 / x
ZeroDivisionError: division by zero

# Test 3 — RecursionError
Traceback (most recent call last):
  File "tree.py", line 18, in traverse
    return traverse(node.left) + traverse(node.right)
RecursionError: maximum recursion depth exceeded

# Test 4 — RuntimeError
Traceback (most recent call last):
  File "cleaner.py", line 34, in remove_nulls
    for key in my_dict:
        del my_dict[key]
RuntimeError: dictionary changed size during iteration
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `langchain` | Core framework — loaders, splitters, prompt templates, chains |
| `langchain-community` | FAISS retriever, BM25 retriever, HuggingFace embeddings |
| `langchain-groq` | Groq API integration for LangChain |
| `faiss-cpu` | Facebook AI Similarity Search — vector database |
| `rank-bm25` | BM25 keyword search algorithm |
| `sentence-transformers` | Embedding model + Cross-Encoder reranker (local, free) |
| `streamlit` | Web UI framework |
| `groq` | Official Groq Python client |
| `python-dotenv` | Load API key from `.env` file |

---

## 🔬 How Cross-Encoder Reranking Works

### The problem with single retrievers

| Method | Strength | Weakness |
|---|---|---|
| BM25 | Exact keyword match | Misses semantic meaning |
| FAISS | Semantic similarity | Misses exact keywords |

### The solution — two-stage retrieval

**Stage 1 — Candidate retrieval:**
```
BM25  → top 10 docs  (keyword match)
FAISS → top 10 docs  (semantic match)
Merge → ~20 unique candidates
```

**Stage 2 — Cross-Encoder reranking:**
```python
# Cross-Encoder reads the query AND each chunk together
ce_inputs = [(query, chunk) for chunk in candidates]
ce_scores = cross_encoder.predict(ce_inputs)
# Returns a relevance score per pair — not just positional rank
```

Unlike BM25/FAISS which score documents independently, the Cross-Encoder reads **both the query and document together** — understanding the relationship between your specific error and each potential fix.

---

## 📈 Improving Results

### Add more errors to the knowledge base

Append to `data/stackoverflow.txt` in this format:

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

Then rebuild indexes:
```bash
python ingest.py
```

### Add your own codebase

```bash
# Windows
Get-ChildItem -Recurse -Filter *.py | Get-Content | Out-File data\my_code.txt

# Mac / Linux
find . -name "*.py" -not -path "./__pycache__/*" | xargs cat > data/my_code.txt

python ingest.py
```


<br/>

⭐ **Star this repo if it helped you!**

</div>
