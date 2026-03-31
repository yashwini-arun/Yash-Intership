from agents.base_agent import BaseAgent
from prompts.synthesis import SYNTHESIS_PROMPT


class SynthesisAgent(BaseAgent):
    def __init__(self):
        super().__init__("SynthesisAgent")

    def synthesize(self, query: str, verified_summary: str, language: str = "English") -> str:
        prompt = SYNTHESIS_PROMPT.format(
            query=query, verified_summary=verified_summary, language=language
        )
        return self.call_llm(prompt)