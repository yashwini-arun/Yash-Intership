from agents.base_agent import BaseAgent
from prompts.summarization import SUMMARIZATION_PROMPT


class SummarizationAgent(BaseAgent):
    def __init__(self):
        super().__init__("SummarizationAgent")

    def summarize(self, query: str, raw_data: str, language: str = "English") -> str:
        prompt = SUMMARIZATION_PROMPT.format(query=query, raw_data=raw_data, language=language)
        return self.call_llm(prompt)