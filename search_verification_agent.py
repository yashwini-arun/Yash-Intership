from agents.base_agent import BaseAgent


class SearchVerificationAgent(BaseAgent):
    """
    Verifies the quality of search results from the Search Agent.
    If results are poor, it refines the query for retry.
    """

    def __init__(self):
        super().__init__("SearchVerificationAgent")

    def verify_search_results(self, query: str, formatted_data: str) -> dict:
        """
        Evaluates search results quality.
        Returns:
            - is_sufficient: True/False
            - reason: why it passed or failed
            - refined_query: better query if failed
            - quality_score: 0-100
        """
        prompt = f"""You are a search quality evaluator.

Given this research query: "{query}"
And these search results:
{formatted_data[:2000]}

Evaluate the search results and respond in this EXACT format:
QUALITY_SCORE: [number 0-100]
IS_SUFFICIENT: [YES or NO]
REASON: [1 sentence explaining why results are sufficient or insufficient]
REFINED_QUERY: [If NO — write a better search query. If YES — write NONE]

Rules:
- Score >= 60 means IS_SUFFICIENT: YES
- Score < 60 means IS_SUFFICIENT: NO
- If results are empty, error messages, or completely unrelated → score < 40
- If results have real content related to the query → score >= 70
"""
        response = self.call_llm(prompt)

        # Parse response
        result = {
            "is_sufficient": True,
            "reason": "Results accepted",
            "refined_query": query,
            "quality_score": 90,
        }

        try:
            lines = response.strip().split("\n")
            for line in lines:
                if line.startswith("QUALITY_SCORE:"):
                    score = int(''.join(filter(str.isdigit, line.split(":", 1)[1][:5])))
                    result["quality_score"] = min(100, max(0, score))
                elif line.startswith("IS_SUFFICIENT:"):
                    val = line.split(":", 1)[1].strip().upper()
                    result["is_sufficient"] = "YES" in val
                elif line.startswith("REASON:"):
                    result["reason"] = line.split(":", 1)[1].strip()
                elif line.startswith("REFINED_QUERY:"):
                    rq = line.split(":", 1)[1].strip()
                    if rq and rq.upper() != "NONE":
                        result["refined_query"] = rq
        except Exception:
            pass

        return result