# 🎯 JobFind — Smart Job Portal Search Engine

> A Python-based job search system demonstrating **three distinct search techniques** — Keyword (BM25), Semantic (FAISS), and Hybrid — through a polished, dark-themed Streamlit interface.

---

## 🧭 Overview

**JobFind** is an intermediate-level NLP project that showcases how modern job portals implement intelligent search. It combines classical information retrieval (BM25) with deep learning-based semantic search (Sentence Transformers + FAISS) to deliver relevant results regardless of how a user phrases their query.

Built as a self-contained prototype with:
- **30 hardcoded job listings** across diverse tech roles
- **3 search modes** — each visually distinct and independently functional
- **Zero external API calls** — runs fully offline after initial model download
- **Interactive UI** with real-time alpha tuning for hybrid search

---

## ✨ Features

- **⚡ Keyword Search (BM25)** — Exact token matching using the BM25Okapi algorithm. Fast and precise for role names and technology keywords.
- **🧠 Semantic Search (FAISS)** — Meaning-based search using `all-MiniLM-L6-v2` sentence embeddings and cosine similarity via FAISS vector index.
- **🔀 Hybrid Search** — Weighted combination of BM25 and semantic scores with a live-tunable alpha (α) slider.
- **Color-coded UI** — Each search mode has a completely distinct visual identity (amber / green / purple) across banners, cards, score bars, and rank labels.
- **Per-result score breakdown** — Every result card shows the scoring method, score value, and (for hybrid) the individual BM25 and semantic sub-scores.
- **Contextual placeholder text** — Search bar hint changes per mode to guide the user.
- **Sidebar quick-launch queries** — Pre-built example queries for each mode.
- **Adjustable result count** — Choose 5, 10, or 15 results per search.

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.8+ | Core language |
| **UI Framework** | Streamlit | Web interface |
| **Keyword Search** | rank-bm25 (BM25Okapi) | Exact token scoring |
| **Embedding Model** | sentence-transformers (`all-MiniLM-L6-v2`) | Text → vector conversion |
| **Vector Search** | FAISS (faiss-cpu) | Similarity search over embeddings |
| **Numerical Computing** | NumPy | Score normalization and sorting |
| **Data** | Python dictionaries (hardcoded) | 30 job listings |

---

## 🔍 How Each Search Works

### ⚡ Keyword Search — BM25

BM25 (Best Match 25) is a ranking algorithm that scores documents based on how often query terms appear in them, adjusted for document length and term rarity across the corpus.

```
Query: "Python Developer"
  ↓
Tokenize → ["python", "developer"]
  ↓
BM25 scores all 30 jobs using:
  - Term Frequency (TF)  — how often the word appears in the job
  - Inverse Doc Frequency (IDF) — how rare the word is across all jobs
  ↓
Return top-K jobs sorted by BM25 score
```

**Best for:** Specific job titles, exact technology names, skill keywords

---

### 🧠 Semantic Search — FAISS + Sentence Transformers

Converts text to high-dimensional vectors (embeddings) that capture meaning, then finds the most similar job vectors to the query vector.

```
Startup:
  All 30 job texts → SentenceTransformer → 30 vectors (384 dims each)
  Normalize vectors → store in FAISS index

Query: "remote job involving AI"
  ↓
SentenceTransformer → query vector (384 dims)
  ↓
Normalize → FAISS cosine similarity search
  ↓
Return top-K jobs by similarity score (0.0 to 1.0)
```

**Best for:** Natural language queries, intent-based searches, synonym matching

---

### 🔀 Hybrid Search — Weighted Combination

Combines both approaches using a tunable alpha (α) weight:

```
hybrid_score = α × semantic_score + (1 − α) × bm25_normalized_score
```

| Alpha (α) | Effect |
|-----------|--------|
| `1.0` | Pure semantic search |
| `0.6` | Default — 60% semantic, 40% keyword |
| `0.0` | Pure keyword search |

BM25 scores are normalized to `[0, 1]` before combining so both are on the same scale.

**Best for:** Mixed queries with both specific terms and natural language intent

---

## ⚙️ Installation & Setup

### Prerequisites

- Python **3.8 or higher**
- pip
- ~500MB disk space (for the sentence-transformer model)
- Internet connection (first run only, to download the model)

### Step 1 — Clone or Download

```bash
# If using git
git clone https://github.com/yourusername/job_portal_search.git
cd job_portal_search

# Or just place all 4 files in a folder and cd into it
cd job_portal_search
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Mac / Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏳ First install takes **2–5 minutes** — it downloads PyTorch, Transformers, and FAISS.

---

## ▶️ Running the App

```bash
streamlit run app.py
```

The app opens automatically at **`http://localhost:8501`**

> ⚠️ **First launch** takes ~30 seconds while the sentence-transformer model (`all-MiniLM-L6-v2`, ~90MB) downloads and caches locally. Subsequent launches are instant.

### Stopping the App

```bash
Ctrl + C
```

---


**⚡ Keyword &nbsp;·&nbsp; 🧠 Semantic &nbsp;·&nbsp; 🔀 Hybrid**

</div>
