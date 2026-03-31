from typing import Optional

from agents.multilingual_agent import MultilingualAgent
from agents.search_agent import SearchAgent
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


# ── Main Pipeline ────────────────────────────────────────────────────────────

def run_pipeline(query: str, status_callback: Optional[object] = None, output_language: str = "English") -> dict:
    guardrail_log = []

    # ── INPUT GUARDRAIL ───────────────────────────────────────────────────────
    try:
        validated_query = validate_query(query)
        guardrail_log.append({
            "check": "Input — Query validation",
            "status": "pass",
            "detail": f"Query accepted: '{validated_query[:60]}'"
        })
    except InputGuardrailError as e:
        guardrail_log.append({
            "check": "Input — Query validation",
            "status": "fail",
            "detail": str(e)
        })
        raise

    # ── Agent 1: Multilingual ────────────────────────────────────────────────
    if status_callback:
        status_callback("🌍 Agent 1 — Detecting language...")

    multilingual_agent = MultilingualAgent()
    result = multilingual_agent.detect_and_translate(validated_query)

    try:
        result = validate_multilingual_output({**result, "original_query": validated_query})
        guardrail_log.append({
            "check": "Agent 1 Output — Language detected",
            "status": "pass",
            "detail": f"Detected: {result['detected_language']}"
        })
    except OutputGuardrailError as e:
        guardrail_log.append({
            "check": "Agent 1 Output — Language detected",
            "status": "fail",
            "detail": str(e)
        })
        raise

    english_query = result["english_query"]
    detected_language = result["detected_language"]

    # ── Agent 2: Search ──────────────────────────────────────────────────────
    if status_callback:
        status_callback("🔍 Agent 2 — Searching arXiv, Wikipedia, Semantic Scholar...")

    search_agent = SearchAgent()
    raw_results = search_agent.search(english_query)
    formatted_data = search_agent.format_for_llm(raw_results)

    try:
        formatted_data = validate_search_output(raw_results, formatted_data)
        total = sum(len(v) for v in raw_results.values())
        guardrail_log.append({
            "check": "Agent 2 Output — Search results",
            "status": "pass",
            "detail": f"{total} results fetched across 3 sources"
        })
    except OutputGuardrailError as e:
        guardrail_log.append({
            "check": "Agent 2 Output — Search results",
            "status": "fail",
            "detail": str(e)
        })
        raise

    translated_results = search_agent.translate_results(raw_results, output_language)

    # ── Agent 3: Summarization ───────────────────────────────────────────────
    if status_callback:
        status_callback("📝 Agent 3 — Summarizing findings...")

    summarization_agent = SummarizationAgent()
    summary = summarization_agent.summarize(english_query, formatted_data, language=output_language)

    try:
        summary = validate_summary_output(summary, english_query)
        guardrail_log.append({
            "check": "Agent 3 Output — Summary quality",
            "status": "pass",
            "detail": f"{len(summary)} chars — sufficient content"
        })
    except OutputGuardrailError as e:
        guardrail_log.append({
            "check": "Agent 3 Output — Summary quality",
            "status": "fail",
            "detail": str(e)
        })
        raise

    # ── Agent 4: Verification ───────────────────────────────────────────────
    if status_callback:
        status_callback("✅ Agent 4 — Verifying and fact-checking...")

    import re
    verification_agent = VerificationAgent()
    verified = verification_agent.verify(english_query, summary, language=output_language)

    try:
        verified = validate_verification_output(verified)
        score_match = re.search(r'\b(\d{1,3})\s*/\s*100\b', str(verified))
        score = score_match.group(1) if score_match else "N/A"
        guardrail_log.append({
            "check": "Agent 4 Output — Accuracy score",
            "status": "pass",
            "detail": f"Score extracted: {score}/100"
        })
    except OutputGuardrailError as e:
        guardrail_log.append({
            "check": "Agent 4 Output — Accuracy score",
            "status": "fail",
            "detail": str(e)
        })
        raise

    # ── Agent 5: Synthesis ──────────────────────────────────────────────────
    if status_callback:
        status_callback("📄 Agent 5 — Generating final research report...")

    synthesis_agent = SynthesisAgent()
    final_report = synthesis_agent.synthesize(english_query, verified, language=output_language)

    try:
        final_report = validate_final_report(final_report, output_language)
        guardrail_log.append({
            "check": "Agent 5 Output — Final report",
            "status": "pass",
            "detail": f"{len(final_report)} chars — complete report generated"
        })
    except OutputGuardrailError as e:
        guardrail_log.append({
            "check": "Agent 5 Output — Final report",
            "status": "fail",
            "detail": str(e)
        })
        raise

    if status_callback:
        status_callback("🌐 Translating report back to your language...")

    # ── Final Output ─────────────────────────────────────────────────────────
    return {
        "agent1_multilingual": {
            "original_query": validated_query,
            "detected_language": detected_language,
            "english_query": english_query,
        },
        "agent2_search": translated_results,
        "agent3_summary": summary,
        "agent4_verified": verified,
        "agent5_final_report": final_report,
        "detected_language": detected_language,
        "english_query": english_query,
        "guardrail_log": guardrail_log,
    }
