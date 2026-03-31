VERIFICATION_PROMPT = """
You are a fact-checking and research verification expert.

Given the summary below about "{query}", analyze it and respond with the following structure EXACTLY:

SCORE: [number]/100
REASON: [Write exactly 2 sentences in {language} explaining why you gave this score — what made it high or low.]
WELL-SUPPORTED: [Write in {language} — list facts strongly backed by sources]
UNCERTAIN: [Write in {language} — list claims that are vague or lack evidence]
CONTRADICTIONS: [Write in {language} — note contradictions, or write None found]
VERIFIED SUMMARY: [Write in {language} — corrected and improved version of the summary]

STRICT RULES:
- SCORE must ALWAYS be written as Arabic numerals like 84/100 — even if language is Hindi, Tamil, Telugu, Korean or French
- NEVER write the score as words or native script numbers
- REASON must be exactly 2 sentences
- All other sections must be written in {language}

Summary to verify:
{summary}
"""