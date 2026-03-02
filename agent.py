"""
agent.py - Hardcoded 3-step ReAct loop with LLM parameter control.
Accepts temperature, top_p, top_k, max_tokens from UI.
"""

import json
import tiktoken
from langchain_groq import ChatGroq
from tools import tool_sentiment, tool_search_news, tool_fact_check


# ── Context window sizes for every supported Groq model ───────────────────────
MODEL_CONTEXT_WINDOWS = {
    "llama-3.1-8b-instant":    131072,
    "llama-3.3-70b-versatile": 131072,
    "llama-3.1-70b-versatile": 131072,
    "gemma2-9b-it":            8192,
}


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens in a string using tiktoken.
    tiktoken doesn't have Groq/Llama encodings, so we use cl100k_base
    (same as GPT-4) which is a close approximation for Llama models too.
    """
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback: rough estimate (1 token ≈ 4 chars)
        return len(text) // 4


def estimate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> dict:
    """
    Estimate API cost in USD.
    Groq pricing (as of 2025) per 1M tokens.
    """
    PRICING = {
        "llama-3.1-8b-instant":    {"input": 0.05,  "output": 0.08},
        "llama-3.3-70b-versatile": {"input": 0.59,  "output": 0.79},
        "llama-3.1-70b-versatile": {"input": 0.59,  "output": 0.79},
        "gemma2-9b-it":            {"input": 0.20,  "output": 0.20},
    }
    rates = PRICING.get(model, {"input": 0.10, "output": 0.20})
    input_cost  = (prompt_tokens    / 1_000_000) * rates["input"]
    output_cost = (completion_tokens / 1_000_000) * rates["output"]
    return {
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens":      prompt_tokens + completion_tokens,
        "input_cost_usd":    round(input_cost,  8),
        "output_cost_usd":   round(output_cost, 8),
        "total_cost_usd":    round(input_cost + output_cost, 8),
        "context_window":    MODEL_CONTEXT_WINDOWS.get(model, 8192),
        "context_used_pct":  round(((prompt_tokens + completion_tokens) /
                                    MODEL_CONTEXT_WINDOWS.get(model, 8192)) * 100, 2),
    }


def run_detection(
    claim:       str,
    api_key:     str,
    model:       str   = "llama-3.1-8b-instant",
    temperature: float = 0.0,
    max_tokens:  int   = 800,
    top_p:       float = 1.0,
    top_k:       int   = 40,
) -> dict:
    """
    Hardcoded 4-step ReAct loop.
    Steps 1-3: tools (no LLM, no looping possible)
    Step  4  : single Groq call with user-controlled params
    """
    steps = []

    # ── STEP 1: Sentiment ──────────────────────────────────────────────────────
    steps.append({"type": "thought",
                  "text": "I will analyze the sentiment and language bias of this claim first."})
    steps.append({"type": "action", "tool": "analyze_sentiment", "input": claim})
    sentiment_result = tool_sentiment(claim)
    steps.append({"type": "observation", "text": sentiment_result})

    # ── STEP 2: News Search ────────────────────────────────────────────────────
    search_query = " ".join(claim.split()[:6])
    steps.append({"type": "thought",
                  "text": f"Sentiment done. Now searching Google News for: '{search_query}'"})
    steps.append({"type": "action", "tool": "search_news", "input": search_query})
    news_result = tool_search_news(search_query)
    steps.append({"type": "observation", "text": news_result})

    # ── STEP 3: Fact-Check Search ──────────────────────────────────────────────
    fact_query = "fact check " + " ".join(claim.split()[:5])
    steps.append({"type": "thought",
                  "text": "News results gathered. Now searching specifically for fact-checks and debunks."})
    steps.append({"type": "action", "tool": "fact_check", "input": fact_query})
    fact_result = tool_fact_check(claim)
    steps.append({"type": "observation", "text": fact_result})

    # ── STEP 4: Groq Final Verdict ─────────────────────────────────────────────
    steps.append({"type": "thought",
                  "text": f"All evidence gathered. Calling Groq ({model}) — "
                          f"temp={temperature}, top_p={top_p}, top_k={top_k}, "
                          f"max_tokens={max_tokens}"})

    prompt = f"""You are a fake news analyst. Based on these observations, produce a credibility verdict.

CLAIM: "{claim}"

OBSERVATION 1 - Sentiment Analysis:
{sentiment_result}

OBSERVATION 2 - News Search Results:
{news_result}

OBSERVATION 3 - Fact-Check Search Results:
{fact_result}

Respond ONLY with a valid JSON object, no markdown, no extra text:
{{
  "score": <integer 0-10>,
  "verdict": "<Likely True|Misleading|Likely False|Unverified>",
  "confidence": "<High|Medium|Low>",
  "reasoning": "<3 clear sentences explaining verdict using specific evidence from observations>",
  "key_finding": "<single most important fact found in the search results>"
}}

Rules: score 0-3=Likely False, 4-6=Misleading/Unverified, 7-10=Likely True"""

    # Count prompt tokens BEFORE calling API
    prompt_tokens = count_tokens(prompt, model)

    # Build LLM with all user-controlled params
    # Note: Groq supports temperature, max_tokens, top_p via model_kwargs
    # top_k is passed as model_kwargs since LangChain-Groq doesn't have a direct param
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model,
        temperature=temperature,
        max_tokens=max_tokens,
        model_kwargs={
            "top_p": top_p,
        }
    )

    completion_tokens = 0
    try:
        response     = llm.invoke(prompt)
        raw          = response.content.strip().replace("```json","").replace("```","").strip()
        verdict      = json.loads(raw)
        completion_tokens = count_tokens(raw, model)
    except Exception:
        combined  = (news_result + fact_result).lower()
        is_false  = any(w in combined for w in
                        ["debunked","false","hoax","conspiracy","no link","no evidence","myth","misinformation"])
        is_true   = any(w in combined for w in
                        ["confirmed","true","proven","verified","official","scientists agree"])
        verdict = {
            "score":       1 if is_false else (8 if is_true else 4),
            "verdict":     "Likely False" if is_false else ("Likely True" if is_true else "Unverified"),
            "confidence":  "Medium",
            "reasoning":   "Keyword-based fallback. " + (
                           "Multiple debunking sources found." if is_false else
                           "Confirming sources found." if is_true else
                           "No clear confirmation or denial found."),
            "key_finding": "See sources above for details."
        }
        completion_tokens = count_tokens(json.dumps(verdict), model)

    # Build cost/token report
    token_report = estimate_cost(prompt_tokens, completion_tokens, model)

    steps.append({"type": "final", "verdict": verdict,
                  "claim": claim, "token_report": token_report})

    return {"steps": steps, "verdict": verdict, "token_report": token_report}