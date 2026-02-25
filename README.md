<div align="center">

<br/>

# RAG Explorer v2

**An interactive platform to explore, compare, and understand Retrieval-Augmented Generation**



> Upload any document. See how it gets split. Ask a question. Compare four completely different AI answers — with confidence scores, chat memory, and strategy ratings.

<br/>

</div>

---

## What This Project Does

RAG Explorer v2 makes the RAG pipeline **visual and interactive**. Instead of a black-box Q&A tool, it exposes every step of the process — how documents are split, how questions are framed, how answers are retrieved — so you can see exactly what affects the quality of AI responses.

---

## Features

### 📦 Four Chunking Strategies
See how the same document gets split four different ways and understand how each strategy affects what the AI can find and answer.

| Strategy | How it works | Best for |
|---|---|---|
| **Fixed-Size** | Splits every N tokens regardless of content | Raw data, logs, unstructured text |
| **Sentence-Based** | Groups complete sentences together | Articles, reports, general prose |
| **Semantic** | Detects topic shifts using embeddings and splits there | Mixed-topic documents, research papers |
| **Document Structure** | Splits at headings and paragraphs | Manuals, structured PDFs, textbooks |

Each strategy shows chunk count, average token size, pros, cons, and a chunk viewer with a dropdown to read any individual chunk.

<br/>

### 💬 Four Prompting Strategies
The same question, asked four structurally different ways, produces four noticeably different answers — different format, depth, and style.

| Strategy | Instruction given to AI | Output style |
|---|---|---|
| **⚡ Zero-Shot** | Answer directly, no examples | 2–3 sentence plain prose |
| **🎯 Few-Shot** | Here are 2 examples, now answer | Structured bullet-point list |
| **🧠 Chain-of-Thought** | Think step-by-step before answering | Step 1 → Step 2 → Step 3 → Final Answer |
| **🎭 Role-Based** | You are a senior expert, answer accordingly | Expert Overview → Key Details → Recommendation |

<br/>

### 🎯 Confidence Scoring
Every answer includes a reliability score calculated from how closely the retrieved document chunks matched the question — no extra AI call needed.

| Score | Label | Meaning |
|---|---|---|
| 75%+ | 🟢 High | Document strongly covers this topic |
| 50–74% | 🟡 Medium | Partial coverage |
| 30–49% | 🔴 Low | Limited match |
| Below 30% | ⚫ Very Low | Question may be outside the document's scope |

<br/>

### 🗨️ Chat Mode with Memory
A full conversational interface that remembers the last three turns. Ask follow-up questions using pronouns and natural references — "What causes it?", "Tell me more", "Give an example" — and the AI resolves them from previous context. Each reply is tagged with the strategy used and shows confidence inline.

<br/>

### 📂 Multi-Document Search
Upload multiple documents and query across all of them simultaneously. The system retrieves the most relevant chunks from each document, merges and re-ranks them globally by relevance score, and colour-codes each chunk by its source document.

<br/>

### 📈 Performance Tracker
Rate every answer 👍 or 👎 from both the Prompting tab and the Chat tab. Ratings accumulate into a ranked leaderboard showing which strategy performs best for your document type — with a live mini-tracker in the sidebar.

---

## Tech Stack

### Backend

| Technology | Version | Purpose |
|---|---|---|
| **FastAPI** | 0.115.0 | REST API framework — async, auto-generates docs at `/docs` |
| **ChromaDB** | 0.5.15 | Local vector database — stores and retrieves chunk embeddings |
| **Sentence-Transformers** | 3.1.1 | `all-MiniLM-L6-v2` — converts text into 384-dimensional semantic vectors |
| **Groq + Llama 3** | — | Free, fast LLM inference for answer generation |
| **PyPDF2** | — | PDF text extraction |
| **python-docx** | — | Word document extraction |
| **python-pptx** | — | PowerPoint extraction |
| **openpyxl** | — | Excel extraction |
| **python-dotenv** | 1.0.1 | Loads API keys from `.env` securely |
| **Uvicorn** | 0.30.6 | ASGI server that runs the FastAPI application |

### Frontend

| Technology | Purpose |
|---|---|
| **Streamlit** | Pure-Python web UI — no JavaScript required |
| **requests** | HTTP calls from the frontend to the FastAPI backend |

---

## Getting Started

### Prerequisites
- Python 3.11 or higher
- A free Groq API key from [console.groq.com](https://console.groq.com) — takes two minutes to create

<br/>

### Step 1 — Clone the Repository

```bash
git clone https://github.com/yourusername/rag-explorer-v2.git
cd rag-explorer-v2
```

### Step 2 — Backend Setup

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
pip install PyPDF2 python-docx python-pptx openpyxl
```

### Step 3 — Configure Your API Key

Create a file named `.env` inside the `backend/` folder and add:

```
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama3-8b-8192
CHROMA_PERSIST_DIR=./chroma_db
```

### Step 4 — Start the Backend

```bash
python main.py
```

API runs at `http://localhost:8000` · Interactive docs at `http://localhost:8000/docs`

### Step 5 — Start the Frontend

Open a second terminal window:

```bash
cd frontend_streamlit
pip install streamlit requests
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chunking/analyze` | Upload a document — returns all four chunking results and indexes into ChromaDB |
| `POST` | `/api/rag/query` | Ask a question — retrieves relevant chunks and returns answers from one or all four strategies |
| `POST` | `/api/prompting/generate` | Generate prompt structures for a given context and question |
| `GET` | `/api/rag/status` | Check if backend is running and Groq is configured |

---

## Supported File Formats

| Format | Extraction method |
|---|---|
| `.pdf` | Text extracted page by page via PyPDF2 |
| `.docx` | Paragraphs and tables via python-docx |
| `.pptx` | Slide text and speaker notes via python-pptx |
| `.xlsx` | Cell values row by row via openpyxl |
| `.txt` | Direct read with encoding detection |
| `.md` | Markdown treated as plain text |
| `.csv` | Comma-separated values as plain text |

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | — | Your Groq API key from console.groq.com |
| `GROQ_MODEL` | No | `llama3-8b-8192` | Model to use — also supports `llama3-70b-8192` |
| `CHROMA_PERSIST_DIR` | No | `./chroma_db` | Folder where ChromaDB stores data on disk |

> ⚠️ The `.env` file is listed in `.gitignore` and must never be committed to version control.

---

<div align="center">

<br/>

Built with FastAPI · ChromaDB · Sentence-Transformers · Groq · Streamlit


</div>
