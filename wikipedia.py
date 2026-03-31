import wikipediaapi
import requests
from config.settings import MAX_WIKI_SENTENCES


def search_wikipedia_titles(query: str) -> list:
    try:
        # Use last 2-3 keywords for better Wikipedia matching
        keywords = " ".join(query.split()[-3:])
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "opensearch",
                "search": keywords,
                "limit": 5,
                "namespace": 0,
                "format": "json",
            },
            timeout=10,
        )
        data = resp.json()
        return data[1] if len(data) > 1 else []
    except Exception:
        return []


def fetch_wikipedia(query: str) -> list:
    try:
        wiki = wikipediaapi.Wikipedia(
            language="en",
            user_agent="ResearchAssistant/1.0"
        )
        titles = search_wikipedia_titles(query)
        page = None
        for title in titles:
            p = wiki.page(title)
            if p.exists() and len(p.summary) > 100:
                page = p
                break
        if not page:
            cleaned = query.lower()
            for w in ["what is", "what are", "how does", "impact of",
                      "effects of", "tell me about", "explain", "define",
                      "impact on", "role of", "importance of"]:
                cleaned = cleaned.replace(w, " ")
            cleaned = " ".join(cleaned.split()).title()
            p = wiki.page(cleaned)
            if p.exists():
                page = p
        if page and page.exists():
            sentences = page.summary.split(". ")
            summary = ". ".join(sentences[:MAX_WIKI_SENTENCES]) + "."
            return [{"title": page.title, "summary": summary, "url": page.fullurl, "source": "Wikipedia"}]
        return [{"title": query, "summary": "No Wikipedia article found for this topic.", "url": "", "source": "Wikipedia"}]
    except Exception as e:
        return [{"error": str(e), "source": "Wikipedia"}]
