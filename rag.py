from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from services.vector_store_service import VectorStoreService
from services.prompting_service import build_all_prompts, BUILDERS, PROMPT_META
from services.llm_service import LLMService
import math

router = APIRouter()


class RAGRequest(BaseModel):
    question: str
    collection_name: str           # single collection OR comma-separated for multi-doc
    prompting_strategy: str = "chain_of_thought"
    role: str = "domain expert"
    top_k: int = 3
    run_all_strategies: bool = False


class RAGStatusResponse(BaseModel):
    llm_configured: bool
    available_collections: list


def compute_confidence(chunks: list) -> dict:
    """Compute answer confidence from retrieved chunk similarity scores."""
    if not chunks:
        return {"score": 0, "label": "No Data", "color": "#888", "explanation": "No chunks retrieved."}

    scores = [c["similarity_score"] for c in chunks]
    avg    = sum(scores) / len(scores)
    top    = scores[0]

    # Weighted: 60% top chunk, 40% average of all
    weighted = (top * 0.6) + (avg * 0.4)
    pct = round(weighted * 100, 1)

    if pct >= 75:
        label, color = "High", "#10b981"
        explanation = f"Top chunk matched at {top*100:.1f}%. The document strongly covers this topic."
    elif pct >= 50:
        label, color = "Medium", "#f59e0b"
        explanation = f"Top chunk matched at {top*100:.1f}%. The document partially covers this topic."
    elif pct >= 30:
        label, color = "Low", "#ef4444"
        explanation = f"Top chunk matched at {top*100:.1f}%. The document may not fully answer this question."
    else:
        label, color = "Very Low", "#6b7280"
        explanation = f"Top chunk matched at {top*100:.1f}%. This question may be outside the document's scope."

    return {
        "score":       pct,
        "label":       label,
        "color":       color,
        "explanation": explanation,
        "top_score":   round(top * 100, 1),
        "avg_score":   round(avg * 100, 1),
    }


@router.get("/status")
def rag_status() -> RAGStatusResponse:
    return RAGStatusResponse(
        llm_configured=LLMService.is_configured(),
        available_collections=VectorStoreService.list_collections()
    )


@router.post("/query")
async def rag_query(req: RAGRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    if not LLMService.is_configured():
        raise HTTPException(503, "GROQ_API_KEY not configured.")

    # ── Multi-doc: merge chunks from multiple collections ──────────────────
    collection_names = [c.strip() for c in req.collection_name.split(",") if c.strip()]

    all_chunks = []
    for col_name in collection_names:
        try:
            chunks = VectorStoreService.retrieve(col_name, req.question, top_k=req.top_k)
            # Tag each chunk with its source document
            for ch in chunks:
                ch["source_collection"] = col_name
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Warning: Could not retrieve from {col_name}: {e}")

    if not all_chunks:
        raise HTTPException(404, "No chunks found. Please upload and index a document first.")

    # Sort by similarity score and take top_k overall
    all_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
    retrieved_chunks = all_chunks[:req.top_k]

    # Build context — label source doc if multi-doc
    if len(collection_names) > 1:
        context_parts = []
        for ch in retrieved_chunks:
            doc_label = ch.get("source_collection", "unknown").replace("rag_", "").replace("_", " ")
            context_parts.append(f"[From: {doc_label}]\n{ch['text']}")
        context = "\n\n---\n\n".join(context_parts)
    else:
        context = "\n\n---\n\n".join([c["text"] for c in retrieved_chunks])

    # Confidence score
    confidence = compute_confidence(retrieved_chunks)

    # Run strategies
    if req.run_all_strategies:
        all_prompts = build_all_prompts(context, req.question, req.role)
        results = {}
        for key, prompt_data in all_prompts.items():
            try:
                llm_result = LLMService.complete(prompt_data["system_prompt"], prompt_data["user_prompt"])
                results[key] = {
                    "strategy":    key,
                    "meta":        PROMPT_META[key],
                    "answer":      llm_result["answer"],
                    "tokens_used": llm_result["total_tokens"],
                    "model":       llm_result["model"],
                }
            except Exception as e:
                results[key] = {"strategy": key, "meta": PROMPT_META[key], "error": str(e)}

        return {
            "question":         req.question,
            "retrieved_chunks": retrieved_chunks,
            "context_used":     context,
            "mode":             "all_strategies",
            "results":          results,
            "confidence":       confidence,
            "multi_doc":        len(collection_names) > 1,
            "sources":          collection_names,
        }
    else:
        all_prompts = build_all_prompts(context, req.question, req.role)
        prompt_data = all_prompts[req.prompting_strategy]
        try:
            llm_result = LLMService.complete(prompt_data["system_prompt"], prompt_data["user_prompt"])
        except Exception as e:
            raise HTTPException(500, f"LLM call failed: {str(e)}")

        return {
            "question":         req.question,
            "retrieved_chunks": retrieved_chunks,
            "context_used":     context,
            "mode":             "single_strategy",
            "strategy":         req.prompting_strategy,
            "meta":             PROMPT_META[req.prompting_strategy],
            "answer":           llm_result["answer"],
            "tokens_used":      llm_result["total_tokens"],
            "model":            llm_result["model"],
            "confidence":       confidence,
            "multi_doc":        len(collection_names) > 1,
            "sources":          collection_names,
        }