import arxiv
import time
from config.settings import MAX_ARXIV_RESULTS


def fetch_arxiv(query: str) -> list:
    """Fetch research papers from arXiv with retry on rate limit."""
    for attempt in range(3):
        try:
            time.sleep(2)  # wait before each attempt
            client = arxiv.Client(
                page_size=MAX_ARXIV_RESULTS,
                delay_seconds=3,
                num_retries=3
            )
            search = arxiv.Search(
                query=query,
                max_results=MAX_ARXIV_RESULTS,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            results = []
            for paper in client.results(search):
                results.append({
                    "title": paper.title,
                    "summary": paper.summary[:800],
                    "url": paper.entry_id,
                    "authors": [a.name for a in paper.authors[:3]],
                    "source": "arXiv",
                })
            return results if results else [{"error": "No results found.", "source": "arXiv"}]

        except Exception as e:
            if "429" in str(e) or "too many" in str(e).lower():
                wait = 10 * (attempt + 1)
                time.sleep(wait)
                continue
            return [{"error": str(e), "source": "arXiv"}]

    return [{"error": "arXiv rate limited. Please wait a moment and try again.", "source": "arXiv"}]