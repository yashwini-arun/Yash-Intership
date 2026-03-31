from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from agents.multilingual_agent import MultilingualAgent
from agents.search_agent import SearchAgent
from agents.search_verification_agent import SearchVerificationAgent
from agents.summarization_agent import SummarizationAgent
from agents.verification_agent import VerificationAgent
from agents.synthesis_agent import SynthesisAgent

from guardrails.input_guardrail import validate_query, InputGuardrailError
from guardrails.output_guardrail import (
    validate_multilingual_output,
    validate_search_output,
    validate_summary_output,
    validate_verification_output,
    validate_final_report,
    OutputGuardrailError,
)

MAX_SEARCH_RETRIES = 3


# ── State ─────────────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    original_query:     str
    output_language:    str
    detected_language:  str
    english_query:      str
    current_query:      str   # may be refined by search verifier
    raw_results:        dict
    formatted_data:     str
    search_retry_count: int   # tracks how many times search was retried
    search_quality:     dict  # stores search verification result
    summary:            str
    verified:           str
    final_report:       str
    status_callback:    Optional[object]
    guardrail_log:      list


# ── Nodes ─────────────────────────────────────────────────────────────────────

def multilingual_node(state: ResearchState) -> ResearchState:
    if state.get("status_callback"):
        state["status_callback"]("🌍 Agent 1 — Detecting language...")

    agent  = MultilingualAgent()
    result = agent.detect_and_translate(state["original_query"])
    log    = state.get("guardrail_log", [])

    try:
        result = validate_multilingual_output({**result, "original_query": state["original_query"]})
        log.append({"check": "Agent 1 — Language detection", "status": "pass", "detail": f"Detected: {result['detected_language']}"})
    except OutputGuardrailError as e:
        log.append({"check": "Agent 1 — Language detection", "status": "fail", "detail": str(e)})
        raise

    return {
        **state,
        "detected_language": result["detected_language"],
        "english_query":     result["english_query"],
        "current_query":     result["english_query"],
        "guardrail_log":     log,
    }


def search_node(state: ResearchState) -> ResearchState:
    retry = state.get("search_retry_count", 0)
    query = state.get("current_query", state["english_query"])

    if state.get("status_callback"):
        if retry == 0:
            state["status_callback"]("🔍 Agent 2 — Searching arXiv, Wikipedia, CrossRef...")
        else:
            state["status_callback"](f"🔄 Agent 2 — Retry {retry}/{MAX_SEARCH_RETRIES} with refined query...")

    agent          = SearchAgent()
    raw_results    = agent.search(query)
    formatted_data = agent.format_for_llm(raw_results)
    log            = state.get("guardrail_log", [])

    try:
        formatted_data = validate_search_output(raw_results, formatted_data)
        total = sum(len(v) for v in raw_results.values())
        log.append({"check": f"Agent 2 — Search results (attempt {retry+1})", "status": "pass", "detail": f"{total} results across 3 sources"})
    except OutputGuardrailError as e:
        log.append({"check": f"Agent 2 — Search results (attempt {retry+1})", "status": "fail", "detail": str(e)})

    translated_results = agent.translate_results(raw_results, state["output_language"])

    return {
        **state,
        "raw_results":        translated_results,
        "formatted_data":     formatted_data,
        "search_retry_count": retry,
        "guardrail_log":      log,
    }


def search_verification_node(state: ResearchState) -> ResearchState:
    """
    NEW AGENT — Verifies quality of search results.
    If insufficient, refines the query for retry.
    """
    if state.get("status_callback"):
        state["status_callback"]("🔎 Search Verification Agent — Checking result quality...")

    agent   = SearchVerificationAgent()
    quality = agent.verify_search_results(
        state["current_query"],
        state["formatted_data"]
    )
    log = state.get("guardrail_log", [])

    status = "pass" if quality["is_sufficient"] else "fail"
    log.append({
        "check":  "Search Verification — Quality check",
        "status": status,
        "detail": f"Score: {quality['quality_score']}/100 — {quality['reason']}"
    })

    return {
        **state,
        "search_quality": quality,
        "current_query":  quality["refined_query"],
        "guardrail_log":  log,
    }


def summarization_node(state: ResearchState) -> ResearchState:
    if state.get("status_callback"):
        state["status_callback"]("📝 Agent 3 — Summarizing findings...")

    agent   = SummarizationAgent()
    summary = agent.summarize(state["english_query"], state["formatted_data"], language=state["output_language"])
    log     = state.get("guardrail_log", [])

    try:
        summary = validate_summary_output(summary, state["english_query"])
        log.append({"check": "Agent 3 — Summary quality", "status": "pass", "detail": f"{len(summary)} chars — sufficient"})
    except OutputGuardrailError as e:
        log.append({"check": "Agent 3 — Summary quality", "status": "fail", "detail": str(e)})
        raise

    return {**state, "summary": summary, "guardrail_log": log}


def verification_node(state: ResearchState) -> ResearchState:
    if state.get("status_callback"):
        state["status_callback"]("✅ Agent 4 — Verifying and fact-checking...")

    import re
    agent    = VerificationAgent()
    verified = agent.verify(state["english_query"], state["summary"], language=state["output_language"])
    log      = state.get("guardrail_log", [])

    try:
        verified = validate_verification_output(verified)
        score    = re.search(r'\b(\d{1,3})\s*/\s*100\b', str(verified))
        log.append({"check": "Agent 4 — Accuracy score", "status": "pass", "detail": f"Score: {score.group(1)}/100" if score else "Score validated"})
    except OutputGuardrailError as e:
        log.append({"check": "Agent 4 — Accuracy score", "status": "fail", "detail": str(e)})
        raise

    return {**state, "verified": verified, "guardrail_log": log}


def synthesis_node(state: ResearchState) -> ResearchState:
    if state.get("status_callback"):
        state["status_callback"]("📄 Agent 5 — Generating final research report...")

    agent        = SynthesisAgent()
    final_report = agent.synthesize(state["english_query"], state["verified"], language=state["output_language"])
    log          = state.get("guardrail_log", [])

    try:
        final_report = validate_final_report(final_report, state["output_language"])
        log.append({"check": "Agent 5 — Final report", "status": "pass", "detail": f"{len(final_report)} chars — complete"})
    except OutputGuardrailError as e:
        log.append({"check": "Agent 5 — Final report", "status": "fail", "detail": str(e)})
        raise

    return {**state, "final_report": final_report, "guardrail_log": log}


# ── Conditional Edge — Retry Logic ────────────────────────────────────────────

def should_retry_search(state: ResearchState) -> str:
    """
    Decision node after Search Verification Agent.
    - If quality is insufficient AND retries remaining → go back to search
    - Otherwise → proceed to summarization
    """
    quality = state.get("search_quality", {})
    retry   = state.get("search_retry_count", 0)

    if not quality.get("is_sufficient", True) and retry < MAX_SEARCH_RETRIES:
        # Increment retry count and loop back
        state["search_retry_count"] = retry + 1
        return "retry_search"
    else:
        return "proceed"


# ── Build LangGraph ───────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    # Add all nodes
    graph.add_node("multilingual",         multilingual_node)
    graph.add_node("search",               search_node)
    graph.add_node("search_verification",  search_verification_node)
    graph.add_node("summarize",            summarization_node)
    graph.add_node("verify",               verification_node)
    graph.add_node("synthesize",           synthesis_node)

    # Entry point
    graph.set_entry_point("multilingual")

    # Sequential edges
    graph.add_edge("multilingual", "search")
    graph.add_edge("search",       "search_verification")

    # Conditional edge — retry loop
    graph.add_conditional_edges(
        "search_verification",
        should_retry_search,
        {
            "retry_search": "search",      # ← loop back to search
            "proceed":      "summarize",   # ← move forward
        }
    )

    # Rest of pipeline
    graph.add_edge("summarize",  "verify")
    graph.add_edge("verify",     "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run_pipeline(query: str, status_callback=None, output_language: str = "English") -> dict:
    guardrail_log = []

    # INPUT GUARDRAIL
    try:
        validated_query = validate_query(query)
        guardrail_log.append({"check": "Input — Query validation", "status": "pass", "detail": f"Accepted: '{validated_query[:60]}'"})
    except InputGuardrailError as e:
        guardrail_log.append({"check": "Input — Query validation", "status": "fail", "detail": str(e)})
        raise

    initial_state: ResearchState = {
        "original_query":     validated_query,
        "output_language":    output_language,
        "detected_language":  "",
        "english_query":      "",
        "current_query":      "",
        "raw_results":        {},
        "formatted_data":     "",
        "search_retry_count": 0,
        "search_quality":     {},
        "summary":            "",
        "verified":           "",
        "final_report":       "",
        "status_callback":    status_callback,
        "guardrail_log":      guardrail_log,
    }

    app         = build_graph()
    final_state = app.invoke(initial_state)

    if status_callback:
        status_callback("🌐 Translating report back to your language...")

    return {
        "agent1_multilingual": {
            "original_query":    validated_query,
            "detected_language": final_state["detected_language"],
            "english_query":     final_state["english_query"],
        },
        "agent2_search":          final_state["raw_results"],
        "search_quality":         final_state.get("search_quality", {}),
        "search_retry_count":     final_state.get("search_retry_count", 0),
        "agent3_summary":         final_state["summary"],
        "agent4_verified":        final_state["verified"],
        "agent5_final_report":    final_state["final_report"],
        "detected_language":      final_state["detected_language"],
        "english_query":          final_state["english_query"],
        "guardrail_log":          final_state.get("guardrail_log", []),
    }