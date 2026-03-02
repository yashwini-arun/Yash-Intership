# 🔍 Fake News Detector

---

## 🧠 Overview

**Fake News Detector** is a full-stack AI web application that takes any news headline or claim and returns a structured credibility verdict. It uses the **ReAct (Reasoning + Acting)** agent pattern — gathering evidence from multiple sources before passing everything to a large language model for final analysis.

The entire pipeline runs in under 10 seconds and costs a fraction of a cent per check.

### What it does

- Analyzes the **sentiment and language bias** of the claim
- Searches **Google News** for related articles in real time
- Searches for **fact-checks** from Snopes, PolitiFact, AFP, and similar sources
- Feeds all evidence into an **LLM (LLaMA 3 via Groq)** for a structured verdict
- Displays the full **chain of thought** — every Thought, Action, and Observation
- Shows **token usage and cost in USD** for every run

---

## 🎬 Demo

```
Claim: "Drinking coffee cures cancer completely"

THOUGHT   → Analyzing sentiment and language bias first
ACTION    → analyze_sentiment["Drinking coffee cures cancer completely"]
RESULT    → Polarity: 0.35 (Positive) | Subjectivity: 0.71 (Highly Subjective) ⚠

THOUGHT   → Searching Google News for related articles
ACTION    → search_news["Drinking coffee cures cancer"]
RESULT    → [1] Coffee May Reduce Some Cancer Risk, But 'Cure' Claims Overstated (Reuters)
            [2] Scientists Warn Against Miracle Cure Headlines (BBC Health)
            ...

THOUGHT   → Searching specifically for fact-checks
ACTION    → fact_check["Drinking coffee cures cancer"]
RESULT    → [1] Fact Check: Coffee Does Not Cure Cancer (Snopes)
            [2] No Evidence Coffee Completely Eliminates Cancer (PolitiFact)
            ...

VERDICT   → Score: 1/10 | Likely False | Confidence: High
REASONING → Multiple fact-checking sources confirm no scientific basis for
            the claim that coffee completely cures cancer. Studies show
            correlation with reduced risk in specific cancers only.
```

---

## ⚙️ How It Works

The app follows a fixed **4-step ReAct loop** — no dynamic tool selection, which keeps it fast and predictable.

```
Step 1  →  Sentiment Analysis    (TextBlob — no internet needed)
Step 2  →  News Search           (Google News RSS — free, no API key)
Step 3  →  Fact-Check Search     (Google News RSS — "fact check" prefix)
Step 4  →  LLM Final Verdict     (LLaMA 3 via Groq API — paid, very cheap)
```

### Step 1 — Sentiment Analysis

- Uses **TextBlob** to score the claim's language
- **Polarity**: −1.0 (very negative) → +1.0 (very positive)
- **Subjectivity**: 0.0 (purely factual) → 1.0 (purely opinion-based)
- Raises a warning flag if the claim is both emotionally charged AND highly polarized — a common pattern in fake news

### Step 2 — News Search

- Takes the **first 6 words** of the claim as the search query
- Calls the **Google News RSS feed** — a free, public Google URL
- Google returns an **XML document** with up to 4 matching news articles
- XML is parsed using Python's built-in `ElementTree` library
- **BeautifulSoup** strips HTML tags from article descriptions
- Returns: article title, publisher name, and a 200-character snippet

### Step 3 — Fact-Check Search

- Same pipeline as Step 2, but the query is prefixed with `"fact check"`
- Example: `"fact check vaccines cause autism"`
- This causes Google to surface articles from **Snopes, PolitiFact, AFP Fact Check, Full Fact**, and similar sources
- Returns 4 fact-checking article headlines and snippets

### Step 4 — LLM Final Verdict

- All three observations are combined into a **single structured prompt**
- **Tiktoken** counts the prompt tokens before the API call
- **ChatGroq** sends the prompt to Groq's API using the selected model and parameters
- The LLM responds with a structured **JSON verdict**
- **Tiktoken** counts response tokens after the API call
- If JSON parsing fails, a **keyword-based fallback** verdict is used (no LLM dependency)

---

## 🏗️ Architecture

```
USER INPUT (claim)
        │
        ▼
┌─────────────────────────────────────────┐
│            app.py  (Streamlit UI)       │
│                                         │
│  Sidebar  →  Model selector             │
│             Temperature slider          │
│             Top-P slider                │
│             Top-K slider                │
│             Max Tokens slider           │
│             API Key input               │
│                                         │
│  Main     →  Claim text box             |
│             Example buttons             │
│             Live token counter          │
│             Results display             │
└─────────────────────────────────────────┘
        │
        │  calls run_detection()
        ▼
┌─────────────────────────────────────────┐
│            agent.py  (Brain)            │
│                                         │
│  1. tool_sentiment()                    │
│  2. tool_search_news()                  │
│  3. tool_fact_check()                   │
│  4. ChatGroq → LLM Final Verdict        │
│  5. estimate_cost() via Tiktoken        │
└─────────────────────────────────────────┘
        │              │              │
        ▼              ▼              ▼
   TextBlob     Google News RSS  Google News RSS
  (local NLP)   (news articles)  (fact-checks)
```

---

## 🛠️ Tools & Libraries

| Library | Version | Purpose |
|---|---|---|
| `streamlit` | 1.32.0 | Builds the entire web UI in pure Python — no HTML or JavaScript needed |
| `langchain-groq` | 0.1.3 | LangChain's integration for Groq API — provides the `ChatGroq` class |
| `requests` | 2.31.0 | Makes HTTP requests to fetch the Google News RSS feed |
| `beautifulsoup4` | 4.12.3 | Parses and strips HTML tags from RSS article descriptions |
| `textblob` | 0.18.0 | Pre-trained NLP library for sentiment polarity and subjectivity scoring |
| `python-dotenv` | 1.0.1 | Loads the Groq API key securely from a `.env` file |
| `lxml` | 5.2.1 | Fast XML/HTML parser backend used by BeautifulSoup |
| `tiktoken` | 0.7.0 | OpenAI's tokenizer — counts tokens and estimates API cost in USD |

### External Services

| Service | Cost | Purpose |
|---|---|---|
| **Groq API** | Paid (very cheap) | Ultra-fast LLaMA 3 inference. Free tier available at console.groq.com |
| **Google News RSS** | Free | Public Google endpoint — no API key needed. Returns XML of news articles |

---

## 💡 Key Concepts

### Context Window

The context window is the **maximum number of tokens** a model can process at once — input prompt + generated output combined. If this limit is exceeded, the model begins losing earlier parts of the conversation.

| Model | Context Window | Equivalent |
|---|---|---|
| `llama-3.1-8b-instant` | 131,072 tokens | ~98,000 words — entire novels |
| `llama-3.3-70b-versatile` | 131,072 tokens | ~98,000 words — entire novels |
| `gemma2-9b-it` | 8,192 tokens | ~6,000 words — a short article |

---

### Temperature

Controls **how creative vs. deterministic** the model's output is. At `0.0`, the model always picks the most probable next token. Higher values flatten the probability distribution, allowing more varied responses.

| Value | Behavior | Best For |
|---|---|---|
| `0.0` | Fully deterministic — same answer every time | JSON output, fact-checking |
| `0.1 – 0.3` | Near-deterministic with slight variation | Analysis, summarization |
| `0.5 – 0.7` | Balanced creativity | General Q&A, explanations |
| `1.0 – 2.0` | Highly creative and unpredictable | Brainstorming, creative writing |

---

### Top-P (Nucleus Sampling)

Restricts the model to only consider the **smallest set of tokens whose combined probability reaches P**. For example, `Top-P = 0.9` means only words that together account for 90% of the probability mass are considered — removing unlikely fringe words while preserving natural diversity.

| Value | Behavior |
|---|---|
| `0.1` | Very focused — only the top 1–2 most likely words considered |
| `0.5` | Moderate — roughly half the probability mass |
| `0.9` | Wide range — standard for most tasks (recommended) |
| `1.0` | No filtering — entire vocabulary is considered |

---

### Top-K

The simplest sampling filter — restricts the model to only the **top K most probable tokens** at each step, regardless of their probability distribution.

| Value | Behavior |
|---|---|
| `1` | Greedy decoding — always picks the single most likely token |
| `10` | Focused — only top 10 candidates considered |
| `40` | Balanced — default for most models |
| `100+` | Very diverse — rare words can appear |

> **Top-P vs Top-K:** Top-P filters by cumulative probability (variable number of candidates). Top-K filters by rank (fixed number of candidates). Generally, tune one at a time, not both.

---

### Tiktoken — Token Counting & Cost Estimation

Tiktoken converts text into **tokens** — the fundamental unit LLMs process. Tokens are sub-word chunks, not whole words. On average: `1 token ≈ 4 characters` in English.

```python
# How cost is calculated in agent.py
input_cost  = (prompt_tokens    / 1_000_000) × input_rate_per_1M
output_cost = (completion_tokens / 1_000_000) × output_rate_per_1M
total_cost  = input_cost + output_cost
```

**Groq Pricing (as of 2025):**

| Model | Input | Output |
|---|---|---|
| `llama-3.1-8b-instant` | $0.05 / 1M tokens | $0.08 / 1M tokens |
| `llama-3.3-70b-versatile` | $0.59 / 1M tokens | $0.79 / 1M tokens |
| `gemma2-9b-it` | $0.20 / 1M tokens | $0.20 / 1M tokens |

> **Note:** Tiktoken uses OpenAI's `cl100k_base` encoding as an approximation for LLaMA models. Actual token counts may vary slightly.

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- `pip` (Python package manager)
- A free Groq API key from [console.groq.com](https://console.groq.com)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```

### Step 2 — Create a Virtual Environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Create the `.env` File

Create a file named `.env` in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> ⚠️ **Never commit your `.env` file to Git.** Add `.env` to your `.gitignore` immediately.

```bash
echo ".env" >> .gitignore
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Your Groq API key. Get it free at [console.groq.com](https://console.groq.com) |

### Supported Models

| Model | Context Window | Speed | Cost |
|---|---|---|---|
| `llama-3.1-8b-instant` | 131,072 | ⚡ Fastest | 💚 Cheapest |
| `llama-3.3-70b-versatile` | 131,072 | 🐢 Slower | 🔴 Higher |
| `llama-3.1-70b-versatile` | 131,072 | 🐢 Slower | 🔴 Higher |
| `gemma2-9b-it` | 8,192 | ⚡ Fast | 💛 Moderate |

### Customizing Example Claims

To change the 5 example buttons shown in the UI, edit the `EXAMPLES` list in `app.py` (line 394):

```python
EXAMPLES = [
    "The moon landing in 1969 was faked by NASA",
    "A new vaccine causes autism in children",
    "5G towers are being used to spread COVID-19",
    "Climate change is causing more frequent hurricanes",
    "Drinking coffee cures cancer completely",
]
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

The app opens automatically at:

```
http://localhost:8501
```

### Usage Guide

1. Enter your **Groq API key** in the sidebar (or set it in `.env` — it will load automatically)
2. Select a **model** from the dropdown
3. Adjust **LLM parameters** with the sliders or leave the defaults
4. **Type a claim** in the text box — or click one of the 5 example buttons
5. Click **"🚀 Run ReAct Agent"**
6. Watch each **Thought → Action → Observation** step appear
7. Review the **final verdict card** with score, reasoning, and key finding
8. Check the **token usage and cost report** at the bottom

---

## 🎛️ LLM Parameters

All parameters are adjustable in the sidebar and passed directly to the LLM.

| Parameter | Range | Default | Effect |
|---|---|---|---|
| `temperature` | 0.0 – 2.0 | `0.0` | Lower = more deterministic. Use 0.0 for consistent JSON output |
| `top_p` | 0.0 – 1.0 | `1.0` | Nucleus sampling. Lower = more focused vocabulary |
| `top_k` | 1 – 100 | `40` | Limits candidate tokens per step |
| `max_tokens` | 100 – 2000 | `800` | Maximum length of the LLM response |

> **Recommended for fact-checking:** `temperature=0.0`, `top_p=0.9`, `top_k=40`, `max_tokens=800`

---

