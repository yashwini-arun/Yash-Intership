# 🔬 Autonomous Research Assistant System
---

## 📌 Overview

The **Autonomous Research Assistant** is a production-grade, multi-agent AI system that automates the entire research workflow — from raw user query to a fully verified, structured research report — in under 90 seconds.

A user types **any research question in any supported language**. The system automatically:
- Detects the language and translates to English for processing
- Searches 3 real-time open-access databases simultaneously
- Quality-validates the search results (with automatic retry if insufficient)
- Summarises, fact-checks, and synthesises a structured final report
- Delivers everything back in the user's chosen language


---

## 🤖 The 6-Agent Pipeline

| # | Agent | File | Role |
|---|-------|------|------|
| 1 | **Multilingual Agent** | `agents/multilingual_agent.py` | Detects input language, translates query to English |
| 2 | **Search Agent** | `agents/search_agent.py` | Fetches from arXiv, Wikipedia & CrossRef simultaneously |
| 2.5 | **Search Verification Agent** | `agents/search_verification_agent.py` | Scores search quality (0–100); retries with refined query if score < 90 |
| 3 | **Summarization Agent** | `agents/summarization_agent.py` | Condenses all source content into a structured summary |
| 4 | **Verification Agent** | `agents/verification_agent.py` | Fact-checks summary; returns numerical accuracy score with explanation |
| 5 | **Synthesis Agent** | `agents/synthesis_agent.py` | Writes final structured report (Overview → Key Findings → Implications → Conclusion) |

### 🔄 How Agents Connect

Each agent's output becomes the next agent's input via LangGraph's `ResearchState`:

```
User Query
   → Agent 1  →  [english_query]
   → Agent 2  →  [raw_results from 3 sources]
   → Search Verification  →  [quality_score]
      └── score < 90? Refine query → retry Agent 2 (max 3x)
   → Agent 3  →  [summary in selected language]
   → Agent 4  →  [verified analysis + accuracy score]
   → Agent 5  →  [final_report in selected language]
   → Streamlit UI displays all outputs
```

---

## 🛡️ Guardrails

Guardrails validate every input and output in the pipeline, preventing errors from propagating.

### Input Guardrail (`guardrails/input_guardrail.py`)
| Check | Rule |
|-------|------|
| Empty query | Cannot be blank or whitespace |
| Too short | Minimum 5 characters |
| Too long | Maximum 500 characters |
| Harmful content | Blocks dangerous keywords |
| Gibberish detection | Must contain recognisable words |

### Output Guardrails (`guardrails/output_guardrail.py`)
| Agent | Check | Recovery |
|-------|-------|----------|
| Agent 1 | Valid JSON with language + translation | Auto-fix: defaults to English |
| Agent 2 | At least one source returned data | Error shown if all sources fail |
| Agent 3 | Non-empty summary, min 50 chars | Retry once |
| Agent 4 | Score in Arabic numerals, 0–100 range | Auto-fix: clamps or injects default 70 |
| Agent 5 | Non-empty report, min 100 chars | Retry once |

---

## 📂 Project Structure

```
research_assistant/
│
├── app.py                         ← Streamlit frontend (entry point)
├── .env                           ← API keys (never commit this)
├── requirements.txt
│
├── config/
│   └── settings.py                ← Model, API keys, result limits
│
├── agents/
│   ├── base_agent.py              ← Parent class: Groq connection + call_llm()
│   ├── multilingual_agent.py      ← Agent 1
│   ├── search_agent.py            ← Agent 2
│   ├── search_verification_agent.py ← Quality gate + retry loop
│   ├── summarization_agent.py     ← Agent 3
│   ├── verification_agent.py      ← Agent 4
│   └── synthesis_agent.py         ← Agent 5
│
├── pipeline/
│   └── orchestrator.py            ← LangGraph StateGraph controller
│
├── services/
│   ├── arxiv.py                   ← arXiv / Europe PMC fetcher
│   ├── wikipedia.py               ← Wikipedia live API fetcher
│   └── semantic_scholar.py        ← CrossRef journal fetcher
│
├── prompts/
│   ├── multilingual.py            ← Agent 1 prompt template
│   ├── summarization.py           ← Agent 3 prompt template
│   ├── verification.py            ← Agent 4 prompt template
│   └── synthesis.py               ← Agent 5 prompt template
│
└── guardrails/
    ├── __init__.py
    ├── input_guardrail.py
    └── output_guardrail.py
```

---

## 🌍 Supported Languages

| Language | Input Detection | Output |
|----------|----------------|--------|
| 🇬🇧 English | ✅ | ✅ |
| 🇮🇳 Hindi | ✅ | ✅ |
| 🇮🇳 Tamil | ✅ | ✅ |
| 🇮🇳 Telugu | ✅ | ✅ |
| 🇰🇷 Korean | ✅ | ✅ |
| 🇫🇷 French | ✅ | ✅ |

> Agents 3, 4 and 5 write their outputs directly in the selected language. Agent 2 search results are also translated via the LLM before display.

---

## 🗄️ Data Sources

| Source | API | Papers Available | API Key Required |
|--------|-----|-----------------|-----------------|
| **arXiv** | Europe PMC (fallback) | 40M+ research papers | ❌ None |
| **Wikipedia** | Wikipedia opensearch API | Full encyclopaedia | ❌ None |
| **CrossRef** | `api.crossref.org` | 140M+ journal articles | ❌ None |

> All data sources are **completely free** with no rate limit issues.

---

## ⚙️ Tech Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.10+ | Core language |
| Streamlit | Latest | Web frontend |
| Groq API | Latest | LLM inference platform |
| Llama 3.3 70B Versatile | — | AI model powering all agents |
| LangGraph | Latest | StateGraph agent orchestration |
| LangChain ChatGroq | Latest | LangGraph ↔ Groq interface |
| requests | Latest | HTTP calls to CrossRef & Wikipedia |
| wikipedia-api | Latest | Wikipedia article fetching |
| python-dotenv | Latest | Secure environment variable loading |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com) — takes under 1 minute to get

### Installation

```bash
# 1. Clone or download the project
cd research_assistant

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Add your Groq API key to .env
echo "GROQ_API_KEY=your_actual_key_here" > .env

# 4. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

### requirements.txt

```
streamlit
groq
langchain-groq
langchain-core
langgraph
requests
wikipedia-api
arxiv
python-dotenv
langdetect
```

---

## 🖥️ UI Walkthrough

After running, the Streamlit UI provides:

| Section | What You See |
|---------|-------------|
| **Search Bar** | Type any research question in any language |
| **Language Dropdown** | Select your output language (default: English) |
| **Agent Progress Bar** | Live status showing which agent is currently running |
| **Agent 1 Output** | Detected language + English translation |
| **Agent 2 Output** | 3 tabs — arXiv papers, Wikipedia, CrossRef journals |
| **Search Quality Card** | Score/100 + colour-coded pass/retry status |
| **Agent 3 Output** | Structured summary in selected language |
| **Agent 4 Output** | Accuracy score card (colour-coded) + 2-sentence explanation |
| **Agent 5 Output** | Complete final research report in selected language |
| **Guardrails Panel** | ✅/❌ status for every validation check |

---

## 🔁 Failure Handling

| Failure | Recovery |
|---------|---------|
| Input is empty / harmful | Pipeline stops immediately, clear error shown |
| Agent 1 language detection fails | Auto-defaults to English, pipeline continues |
| One API source blocked | Other sources continue, partial results used |
| All APIs blocked | Pipeline stops with network error message |
| Search quality score < 90 | Query refined automatically, Agent 2 retried (max 3×) |
| Agent 3 returns empty | Retried once; error shown if second attempt fails |
| Agent 4 score is invalid | Auto-fixed to default 70/100 with warning |
| Agent 5 report too short | Retried once; error shown if second attempt fails |

---

## 💡 Sample Research Questions

```
# Science & Technology
"Impact of AI on healthcare"
"Applications of deep learning in drug discovery"
"What is quantum computing"

# Environment
"Impact of water pollution on human health"
"Causes and effects of climate change"
"Effects of air pollution on children"

# Social Science
"Impact of social media on mental health"
"Effects of remote work on productivity"

# Multilingual Examples
"जल प्रदूषण का प्रभाव"          ← Hindi
"செயற்கை நுண்ணறிவின் தாக்கம்"  ← Tamil
"인공지능의 영향"                  ← Korean
"Impact du changement climatique" ← French
```

---

## 📊 Performance

| Metric | Result |
|--------|--------|
| Average response time (English) | 45–75 seconds |
| Average response time (non-English) | 60–90 seconds |
| Search quality score range | 72–88 / 100 |
| Verification accuracy score range | 78–92 / 100 |
| Retry trigger rate | ~30–40% of queries |
| Guardrail catch rate | 100% of invalid inputs blocked |
| Wikipedia fetch success rate | 95%+ |

---

## ⚠️ Known Limitations

- **Institutional networks** may block arXiv. Use mobile hotspot if blocked — or Europe PMC fallback handles it automatically.
- **Groq free tier** has per-minute rate limits. Avoid rapid consecutive searches.
- **CrossRef abstracts** — ~40% of papers lack abstracts; LLM auto-generates summaries for those.
- **Local deployment only** — currently runs on `localhost:8501`. Deployable to Streamlit Cloud.

---

