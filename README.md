# ⬡ DocMind AI  
### *Document Intelligence Bot powered by RAG + LangChain + Zephyr-7B*

---

## 🌌 Overview
DocMind AI is a **next-generation document intelligence system**.  
Upload any PDF, ask natural language questions, and explore answers through **three retrieval approaches** — all wrapped in a sleek, modern Streamlit interface.

---

## ✨ Key Features

- 📄 **PDF Upload & Processing**  
  Extracts text, chunks intelligently, embeds with `sentence-transformers/all-MiniLM-L6-v2`.

- 🔍 **Three Retrieval Approaches**  
  | Approach | Technique | Memory | Best For |
  |----------|-----------|--------|----------|
  | 🟢 Basic RAG | Fixed chunks + similarity | None | Quick factual Q&A |
  | 🔵 MMR Retrieval | Sentence chunks + Maximal Marginal Relevance | 3 turns | Conversational Q&A |
  | 🟣 Hybrid Search | BM25 + FAISS hybrid | 5 turns | Deep, complex queries |

- 🧠 **LLM Integration**  
  - HuggingFace Zephyr-7B-Beta  
  - Groq Llama 3.1 API fallback  

- 🎨 UI 
  - Dark editorial theme  
  - Chat bubbles, KPI cards, pipeline visualizer  
  - Source chunk citations  

---

## 🏗️ Architecture

📄 PDF Upload → 🔀 Chunking → 🔢 Embeddings → 📦 FAISS/BM25 Index
↓                                 ↓
Query Vectorization → Retrieval → LLM Generation → 💬 Answer
