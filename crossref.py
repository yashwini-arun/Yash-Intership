import requests
from config.settings import MAX_SEMANTIC_RESULTS


def fetch_crossref(query: str) -> list:
    """Fetch research papers from CrossRef API — free, no key needed."""
    try:
        url = "https://api.crossref.org/works"
        params = {
            "query.bibliographic": query,
            "rows": MAX_SEMANTIC_RESULTS + 5,
            "select": "title,abstract,URL,author,published-print,container-title,subject",
        }
        headers = {
            "User-Agent": "ResearchAssistant/1.0 (mailto:research@assistant.com)"
        }
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("message", {}).get("items", []):
            title = item.get("title", ["Untitled"])[0] if item.get("title") else "Untitled"

            abstract = item.get("abstract", "")
            for tag in ["<jats:p>", "</jats:p>", "<jats:italic>", "</jats:italic>",
                        "<jats:bold>", "</jats:bold>", "<jats:sup>", "</jats:sup>",
                        "<jats:sub>", "</jats:sub>", "<jats:title>", "</jats:title>"]:
                abstract = abstract.replace(tag, "")
            abstract = abstract.strip()

            if not abstract:
                continue

            url_link = item.get("URL", "")
            authors = []
            for a in item.get("author", [])[:3]:
                name = f"{a.get('given', '')} {a.get('family', '')}".strip()
                if name:
                    authors.append(name)

            year = ""
            if item.get("published-print"):
                year = str(item["published-print"].get("date-parts", [[""]])[0][0])

            journal = ""
            if item.get("container-title"):
                journal = item["container-title"][0]

            results.append({
                "title": title,
                "summary": abstract[:800],
                "url": url_link,
                "authors": authors,
                "year": year,
                "journal": journal,
                "source": "CrossRef",
            })

            if len(results) >= MAX_SEMANTIC_RESULTS:
                break

        return results if results else [{"summary": "No results with abstracts found.", "source": "CrossRef"}]

    except Exception as e:
        return [{"error": str(e), "source": "CrossRef"}]