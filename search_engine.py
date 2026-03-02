import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
from data import get_jobs

# ─────────────────────────────────────────────
# Load data and models once at startup
# ─────────────────────────────────────────────
JOBS = get_jobs()

# Build a combined text for each job (used for both BM25 and embeddings)
def _build_text(job):
    return f"{job['job_title']} {job['company']} {job['location']} {job['skills']} {job['description']} {job['type']}"

JOB_TEXTS = [_build_text(j) for j in JOBS]

# ── BM25 Setup ──────────────────────────────
tokenized_corpus = [text.lower().split() for text in JOB_TEXTS]
bm25 = BM25Okapi(tokenized_corpus)

# ── Sentence Transformer + FAISS Setup ──────
print("Loading embedding model... (first run may take ~30 seconds)")
model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight & fast
embeddings = model.encode(JOB_TEXTS, show_progress_bar=False).astype("float32")

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product = cosine after norm
index.add(embeddings)
print("Search engine ready!")

# ─────────────────────────────────────────────
# 1. KEYWORD SEARCH (BM25)
# ─────────────────────────────────────────────
def keyword_search(query: str, top_k: int = 5):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({**JOBS[idx], "score": round(float(scores[idx]), 4), "search_type": "Keyword (BM25)"})
    return results

# ─────────────────────────────────────────────
# 2. SEMANTIC SEARCH (FAISS + Sentence Transformers)
# ─────────────────────────────────────────────
def semantic_search(query: str, top_k: int = 5):
    query_vec = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append({**JOBS[idx], "score": round(float(score), 4), "search_type": "Semantic (FAISS)"})
    return results

# ─────────────────────────────────────────────
# 3. HYBRID SEARCH (BM25 + Semantic combined)
# ─────────────────────────────────────────────
def hybrid_search(query: str, top_k: int = 5, alpha: float = 0.6):
    """
    alpha = weight for semantic score (0.6 = favour semantic)
    1 - alpha = weight for keyword score
    """
    # BM25 scores (normalize to 0-1)
    tokens = query.lower().split()
    bm25_scores = np.array(bm25.get_scores(tokens))
    bm25_max = bm25_scores.max()
    bm25_norm = bm25_scores / bm25_max if bm25_max > 0 else bm25_scores

    # Semantic scores
    query_vec = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vec)
    sem_scores_raw, sem_indices = index.search(query_vec, len(JOBS))

    # Map semantic scores back to job index order
    sem_scores = np.zeros(len(JOBS))
    for score, idx in zip(sem_scores_raw[0], sem_indices[0]):
        if idx != -1:
            sem_scores[idx] = score

    # Combine: hybrid_score = alpha * semantic + (1 - alpha) * keyword
    hybrid_scores = alpha * sem_scores + (1 - alpha) * bm25_norm

    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            **JOBS[idx],
            "score": round(float(hybrid_scores[idx]), 4),
            "bm25_score": round(float(bm25_norm[idx]), 4),
            "semantic_score": round(float(sem_scores[idx]), 4),
            "search_type": "Hybrid"
        })
    return results