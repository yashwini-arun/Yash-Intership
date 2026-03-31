from groq import Groq
from config.settings import GROQ_API_KEY, GROQ_MODEL, MAX_TOKENS, TEMPERATURE


class BaseAgent:
    """All agents inherit from this. Handles Groq API calls."""

    def __init__(self, name: str):
        self.name = name
        self.client = Groq(api_key=GROQ_API_KEY)

    def call_llm(self, prompt: str) -> str:
        """Send a prompt to Groq and return the text response."""
        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[{self.name} Error]: {str(e)}"