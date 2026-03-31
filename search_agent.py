from agents.base_agent import BaseAgent
from services.arxiv import fetch_arxiv
from services.wikipedia import fetch_wikipedia
from services.crossref import fetch_crossref


class SearchAgent(BaseAgent):
    def __init__(self):
        super().__init__("SearchAgent")

    def search(self, query: str) -> dict:
        arxiv_results    = fetch_arxiv(query)
        wiki_results     = fetch_wikipedia(query)
        crossref_results = fetch_crossref(query)
        return {
            "arxiv":            arxiv_results,
            "wikipedia":        wiki_results,
            "semantic_scholar": crossref_results,
        }

    def translate_results(self, results: dict, language: str) -> dict:
        if language.lower() == "english":
            return results
        translated = {}
        for source, items in results.items():
            translated_items = []
            for item in items:
                if "error" in item:
                    translated_items.append(item)
                    continue
                title   = item.get("title", "")
                summary = item.get("summary", "")
                prompt  = f"""Translate the following title and summary to {language}.
Keep proper nouns, author names, and URLs unchanged.
Respond in this exact format:
TITLE: <translated title>
SUMMARY: <translated summary>

Title: {title}
Summary: {summary}"""
                response = self.call_llm(prompt)
                try:
                    t_title   = title
                    t_summary = summary
                    for line in response.strip().split("\n"):
                        if line.startswith("TITLE:"):
                            t_title = line.replace("TITLE:", "").strip()
                        elif line.startswith("SUMMARY:"):
                            t_summary = line.replace("SUMMARY:", "").strip()
                    translated_items.append({**item, "title": t_title, "summary": t_summary})
                except Exception:
                    translated_items.append(item)
            translated[source] = translated_items
        return translated

    def format_for_llm(self, results: dict) -> str:
        formatted = []
        for source, items in results.items():
            formatted.append(f"\n=== {source.upper()} ===")
            for item in items:
                if "error" in item:
                    formatted.append(f"Error: {item['error']}")
                    continue
                formatted.append(f"Title: {item.get('title', 'N/A')}")
                formatted.append(f"Content: {item.get('summary', 'N/A')}")
                if item.get("url"):
                    formatted.append(f"URL: {item['url']}")
                formatted.append("")
        return "\n".join(formatted)