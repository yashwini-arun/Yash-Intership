# config.py
# Central configuration for all settings

import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# API TOKENS
# ─────────────────────────────────────────
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# ─────────────────────────────────────────
# EMBEDDING MODEL (same for all approaches)
# ─────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

# ─────────────────────────────────────────
# LLM MODEL — Llama 3.1-8B via HuggingFace
# ─────────────────────────────────────────
LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"

# ─────────────────────────────────────────
# CHUNKING CONFIGS PER APPROACH
# ─────────────────────────────────────────
CHUNK_CONFIGS = {
    "approach1": {
        "name": "Approach 1 — Basic RAG",
        "chunk_size": 400,
        "chunk_overlap": 40,
        "retrieval_type": "similarity",
        "top_k": 3,
        "color": "#00c853",
        "icon": "🟢",
        "description": "Fixed chunking + similarity search. Fast and reliable."
    },
    "approach2": {
        "name": "Approach 2 — MMR Retrieval",
        "chunk_size": 600,
        "chunk_overlap": 60,
        "retrieval_type": "mmr",
        "top_k": 4,
        "color": "#2979ff",
        "icon": "🔵",
        "description": "Sentence chunking + MMR search. Diverse, non-repetitive answers."
    },
    "approach3": {
        "name": "Approach 3 — Hybrid BM25 + FAISS",
        "chunk_size": 800,
        "chunk_overlap": 80,
        "retrieval_type": "hybrid",
        "top_k": 5,
        "color": "#d500f9",
        "icon": "🟣",
        "description": "Large chunks + BM25 + FAISS hybrid. Best quality answers."
    }
}

# ─────────────────────────────────────────
# STORAGE PATHS
# ─────────────────────────────────────────
STORAGE_DIR = "storage"
STORAGE_PATHS = {
    "approach1": f"{STORAGE_DIR}/approach1/faiss_index",
    "approach2": f"{STORAGE_DIR}/approach2/faiss_index",
    "approach3": f"{STORAGE_DIR}/approach3/faiss_index",
    "bm25":      f"{STORAGE_DIR}/approach3/bm25_index.pkl",
}

# ─────────────────────────────────────────
# LLM GENERATION SETTINGS
# ─────────────────────────────────────────
LLM_SETTINGS = {
    "temperature": 0.2,
    "max_new_tokens": 512,
    "repetition_penalty": 1.1,
    "return_full_text": False
}

# ─────────────────────────────────────────
# MEMORY SETTINGS
# ─────────────────────────────────────────
MEMORY_WINDOW = {
    "approach1": 0,   # No memory (RetrievalQA)
    "approach2": 3,   # Last 3 exchanges
    "approach3": 5,   # Last 5 exchanges
}