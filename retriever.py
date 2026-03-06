"""
retriever.py
Hybrid Retrieval: BM25 + FAISS → Cross-Encoder Reranking
1. BM25 fetches top-10 by keyword
2. FAISS fetches top-10 by semantic similarity
3. Combined unique 20 docs passed to Cross-Encoder
4. Cross-Encoder scores each doc against the query as a pair
5. Returns top-5 reranked by Cross-Encoder score
"""

import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

FAISS_PATH   = "faiss_index"
CHUNKS_PATH  = "chunks.pkl"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
CE_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K        = 10
FINAL_TOP_K  = 5

def _short(doc):
    return doc.page_content.strip()[:60].replace("\n", " ") + "..."

def retrieve_with_rrf_details(query: str) -> dict:
    """
    Returns:
      - bm25_results    : top-10 from BM25
      - faiss_results   : top-10 from FAISS
      - rrf_results     : top-5 after Cross-Encoder reranking (named rrf_results for app.py compatibility)
      - rrf_table       : list of dicts for UI visualization
    """

    # ── Load indexes ──────────────────────────────────────────────────────────
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    embeddings   = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    faiss_store  = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    # ── Step 1: BM25 retrieval ────────────────────────────────────────────────
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = TOP_K
    bm25_docs = bm25_retriever.invoke(query)

    # ── Step 2: FAISS retrieval ───────────────────────────────────────────────
    faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": TOP_K})
    faiss_docs = faiss_retriever.invoke(query)

    # ── Step 3: Merge unique docs ─────────────────────────────────────────────
    seen     = set()
    all_docs = []
    for doc in bm25_docs + faiss_docs:
        key = doc.page_content.strip()[:120]
        if key not in seen:
            seen.add(key)
            all_docs.append(doc)

    # ── Step 4: Cross-Encoder reranking ───────────────────────────────────────
    # Cross-Encoder reads (query, chunk) together and gives a relevance score
    ce_model  = CrossEncoder(CE_MODEL)
    ce_inputs = [(query, doc.page_content.strip()) for doc in all_docs]
    ce_scores = ce_model.predict(ce_inputs)   # returns float score per pair

    # Attach scores and sort descending
    scored = sorted(
        zip(ce_scores, all_docs),
        key=lambda x: x[0],
        reverse=True
    )

    reranked_docs = [doc for _, doc in scored][:FINAL_TOP_K]
    reranked_scores = [score for score, _ in scored][:FINAL_TOP_K]

    # ── Step 5: Build visualization table ─────────────────────────────────────
    bm25_rank  = {doc.page_content.strip()[:120]: i + 1 for i, doc in enumerate(bm25_docs)}
    faiss_rank = {doc.page_content.strip()[:120]: i + 1 for i, doc in enumerate(faiss_docs)}

    rrf_table = []
    for final_rank, (doc, score) in enumerate(zip(reranked_docs, reranked_scores), 1):
        key    = doc.page_content.strip()[:120]
        b_rank = bm25_rank.get(key, None)
        f_rank = faiss_rank.get(key, None)

        if b_rank and f_rank:
            boost = "Both ⚡"
        elif b_rank:
            boost = "BM25 only"
        else:
            boost = "FAISS only"

        rrf_table.append({
            "final_rank"  : final_rank,
            "label"       : _short(doc),
            "source"      : doc.metadata.get("source", "unknown"),
            "bm25_rank"   : f"#{b_rank}" if b_rank else "—",
            "faiss_rank"  : f"#{f_rank}" if f_rank else "—",
            "rrf_score"   : round(float(score), 4),
            "doc"         : doc,
            "boost"       : boost,
        })

    return {
        "bm25_results"  : bm25_docs,
        "faiss_results" : faiss_docs,
        "rrf_results"   : reranked_docs,   # kept as rrf_results for app.py compatibility
        "rrf_table"     : rrf_table,
    }


def retrieve(query: str) -> list:
    """Simple wrapper — returns top-5 Cross-Encoder reranked docs."""
    return retrieve_with_rrf_details(query)["rrf_results"]