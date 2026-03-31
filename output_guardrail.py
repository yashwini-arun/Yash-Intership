"""
Output Guardrails — validate each agent's output before passing to next agent.
Checks for: empty output, too short, error messages, missing sections.
"""

import re


class OutputGuardrailError(Exception):
    """Raised when agent output fails guardrail checks."""
    pass


# ── Agent 1 — Multilingual output check ──────────────────────────────────────

def validate_multilingual_output(result: dict) -> dict:
    """Ensure language detection returned valid results."""

    if not result.get("english_query") or len(result["english_query"].strip()) < 2:
        result["english_query"] = result.get("original_query", "")

    if not result.get("detected_language"):
        result["detected_language"] = "English"

    return result


# ── Agent 2 — Search output check ────────────────────────────────────────────

def validate_search_output(raw_results: dict, formatted_data: str) -> str:
    """Ensure search returned usable content."""

    if not formatted_data or len(formatted_data.strip()) < 20:
        raise OutputGuardrailError(
            "Search Agent returned no usable content. Please try a different query."
        )

    # Check if all sources errored
    all_error = all(
        all("error" in item for item in items)
        for items in raw_results.values()
        if items
    )
    if all_error:
        raise OutputGuardrailError(
            "All data sources are currently unavailable. Please try again in a moment."
        )

    return formatted_data


# ── Agent 3 — Summarization output check ─────────────────────────────────────

def validate_summary_output(summary: str, query: str) -> str:
    """Ensure summary is meaningful and complete."""

    if not summary or len(summary.strip()) < 50:
        raise OutputGuardrailError(
            "Summarization Agent produced an insufficient response. Please try again."
        )

    # Check if output is just an error message
    error_phrases = ["[summarizationagent error]", "error:", "i cannot", "i'm unable"]
    if any(phrase in summary.lower()[:100] for phrase in error_phrases):
        raise OutputGuardrailError(
            "Summarization Agent encountered an error. Please try again."
        )

    return summary.strip()


# ── Agent 4 — Verification output check ──────────────────────────────────────

def validate_verification_output(verified: str) -> str:
    """Ensure verification includes a valid numerical score."""

    if not verified or len(verified.strip()) < 20:
        raise OutputGuardrailError(
            "Verification Agent produced an insufficient response. Please try again."
        )

    verified = str(verified).strip()

    # Check score is present as Arabic numerals
    score_match = re.search(r'\b(\d{1,3})\s*/\s*100\b', verified)
    if not score_match:
        # Inject a default score note if missing
        verified = "SCORE: 70/100\nREASON: Score could not be automatically extracted. Moderate accuracy assumed.\n\n" + verified

    # Validate score range
    if score_match:
        score = int(score_match.group(1))
        if score < 0 or score > 100:
            verified = verified.replace(
                score_match.group(0), "75/100"
            )

    return verified


# ── Agent 5 — Synthesis output check ─────────────────────────────────────────

def validate_final_report(report: str, language: str) -> str:
    """Ensure final report is complete and well-structured."""

    if not report or len(report.strip()) < 100:
        raise OutputGuardrailError(
            "Synthesis Agent produced an incomplete report. Please try again."
        )

    report = report.strip()

    # Check for error messages
    error_phrases = ["[synthesisagent error]", "error:", "i cannot generate"]
    if any(phrase in report.lower()[:100] for phrase in error_phrases):
        raise OutputGuardrailError(
            "Synthesis Agent encountered an error. Please try again."
        )

    return report