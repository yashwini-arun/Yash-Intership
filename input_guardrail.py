"""
Input Guardrail — validates user query before pipeline starts.
Checks for: empty input, too short, harmful content, gibberish.
"""

BLOCKED_KEYWORDS = [
    "bomb", "weapon", "kill", "hack", "exploit", "malware",
    "drug synthesis", "poison", "suicide method", "illegal"
]


class InputGuardrailError(Exception):
    """Raised when input fails guardrail checks."""
    pass


def validate_query(query: str) -> str:
    """
    Validate and clean the user query.
    Returns cleaned query or raises InputGuardrailError.
    """

    # Check 1 — empty or whitespace only
    if not query or not query.strip():
        raise InputGuardrailError("Query cannot be empty. Please enter a research question.")

    query = query.strip()

    # Check 2 — too short (less than 3 chars)
    if len(query) < 3:
        raise InputGuardrailError("Query is too short. Please enter a meaningful research question.")

    # Check 3 — too long (over 500 chars)
    if len(query) > 500:
        raise InputGuardrailError("Query is too long. Please keep it under 500 characters.")

    # Check 4 — harmful/blocked keywords
    query_lower = query.lower()
    for keyword in BLOCKED_KEYWORDS:
        if keyword in query_lower:
            raise InputGuardrailError(
                f"Query contains restricted content. Please enter a valid research question."
            )

    # Check 5 — gibberish (all special characters or numbers only)
    alpha_count = sum(c.isalpha() for c in query)
    if alpha_count < 2:
        raise InputGuardrailError("Query appears to be invalid. Please enter a proper research question.")

    return query