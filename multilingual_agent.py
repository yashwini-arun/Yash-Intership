import json
from agents.base_agent import BaseAgent
from prompts.multilingual import DETECT_AND_TRANSLATE_PROMPT, TRANSLATE_BACK_PROMPT


class MultilingualAgent(BaseAgent):
    def __init__(self):
        super().__init__("MultilingualAgent")

    def detect_and_translate(self, query: str) -> dict:
        """Detect language and translate query to English."""
        prompt = DETECT_AND_TRANSLATE_PROMPT.format(query=query)
        response = self.call_llm(prompt)
        try:
            # Extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            json_str = response[start:end]
            result = json.loads(json_str)
            return {
                "detected_language": result.get("detected_language", "English"),
                "english_query": result.get("english_query", query),
            }
        except Exception:
            return {"detected_language": "English", "english_query": query}

    def translate_back(self, report: str, target_language: str) -> str:
        """Translate the final report back to the user's original language."""
        if target_language.lower() == "english":
            return report
        prompt = TRANSLATE_BACK_PROMPT.format(
            target_language=target_language, report=report
        )
        return self.call_llm(prompt)