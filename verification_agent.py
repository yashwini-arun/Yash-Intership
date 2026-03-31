from agents.base_agent import BaseAgent
from prompts.verification import VERIFICATION_PROMPT


class VerificationAgent(BaseAgent):
    def __init__(self):
        super().__init__("VerificationAgent")

    def verify(self, query: str, summary: str, language: str = "English") -> str:
        prompt = VERIFICATION_PROMPT.format(query=query, summary=summary, language=language)
        result = self.call_llm(prompt)
        # Always return a clean string — never a dict
        if isinstance(result, dict):
            return str(result)
        return result