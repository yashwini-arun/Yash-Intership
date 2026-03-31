DETECT_AND_TRANSLATE_PROMPT = """
You are a multilingual expert. Given the user query below:
1. Detect the language it is written in.
2. Translate it to English if it is not already in English.

Respond in this exact JSON format:
{{
  "detected_language": "<language name>",
  "english_query": "<translated or original query in English>"
}}

User Query: {query}
"""

TRANSLATE_BACK_PROMPT = """
You are a multilingual expert. Translate the following research report from English into {target_language}.
Keep all formatting, headings, and structure intact.

Report:
{report}
"""