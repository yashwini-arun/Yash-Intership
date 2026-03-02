"""
tools.py - Plain functions (no @tool decorator). Called directly by agent.py.
"""

import requests
import urllib.parse
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from textblob import TextBlob


def tool_sentiment(text: str) -> str:
    """Analyze sentiment and subjectivity of the claim."""
    try:
        blob = TextBlob(str(text)[:500])
        p = round(blob.sentiment.polarity, 2)
        s = round(blob.sentiment.subjectivity, 2)
        pol = "Positive" if p > 0.3 else ("Negative" if p < -0.3 else "Neutral")
        sub = "Highly Subjective" if s > 0.6 else ("Moderate" if s > 0.3 else "Objective")
        warn = " ⚠ Emotionally charged language detected." if s > 0.5 and abs(p) > 0.3 else " Language appears balanced."
        return f"Polarity: {p} ({pol}) | Subjectivity: {s} ({sub}).{warn}"
    except Exception as e:
        return f"Sentiment error: {e}"


def _google_news_rss(query: str, max_results: int = 4) -> list:
    """Fetch articles from Google News RSS — no API key needed."""
    encoded = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    results = []
    for item in root.findall(".//item")[:max_results]:
        title  = item.findtext("title", "").strip()
        link   = item.findtext("link",  "").strip()
        source = item.findtext("source","").strip()
        desc   = item.findtext("description","").strip()
        if desc:
            desc = BeautifulSoup(desc, "lxml").get_text()[:200]
        results.append({"title": title, "source": source, "url": link, "desc": desc})
    return results


def tool_search_news(query: str) -> str:
    """Search Google News for articles about the claim."""
    try:
        items = _google_news_rss(query)
        if not items:
            return "No results found."
        lines = []
        for i, r in enumerate(items, 1):
            src = f" ({r['source']})" if r['source'] else ""
            lines.append(f"[{i}] {r['title']}{src}\n    {r['desc']}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Search failed: {e}"


def tool_fact_check(claim: str) -> str:
    """Search Google News specifically for fact-checks about the claim."""
    query = "fact check " + " ".join(claim.split()[:5])
    try:
        items = _google_news_rss(query)
        if not items:
            return "No fact-check results found."
        lines = []
        for i, r in enumerate(items, 1):
            src = f" ({r['source']})" if r['source'] else ""
            lines.append(f"[{i}] {r['title']}{src}\n    {r['desc']}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Fact-check search failed: {e}"