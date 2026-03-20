# 🔬 Autonomous Research Assistant System
### Multi-Agent AI System with Multilingual Support

> A production-ready multi-agent AI system that autonomously researches any topic, verifies facts, and delivers structured reports — powered by **Groq API + Llama 3.3 70B**.

---

## 📌 Quick Stats

| Property | Details |
|---|---|
| **AI Model** | Llama 3.3 70B Versatile (via Groq API) |
| **Framework** | Python + Streamlit |
| **Agents** | 5-agent sequential pipeline |
| **Languages** | 6 supported (EN, HI, TA, TE, KO, FR) |
| **Data Sources** | arXiv · Wikipedia · CrossRef |
| **API Keys Required** | Groq only (free) |

---

## 📖 Overview

The Autonomous Research Assistant automates end-to-end research workflows. A user submits any research question in any language, and the system orchestrates five specialized AI agents to **search**, **summarize**, **verify**, and **synthesize** a complete research report.

### Business Objective
> Accelerate research workflows and improve quality of insights by automating multi-step reasoning, cross-source validation, and knowledge synthesis.

### Key Capabilities
- Multi-step reasoning across 5 specialized agents
- Cross-source validation using 3 live data sources
- Knowledge synthesis into structured research reports
- Multilingual support — input and output in 6 languages
- Real-time agent progress tracking on the UI

---

## 🛠 Technology Stack

| Tool | Purpose |
|---|---|
| **Python 3.10+** | Core backend language |
| **Streamlit** | Web frontend — UI, language selector, output rendering |
| **Groq API** | AI inference platform — runs Llama 3.3 70B |
| **arXiv API** | Free research paper database — no key required |
| **Wikipedia API** | Free encyclopedia — fetches articles via opensearch |
| **CrossRef API** | Free journal database — 140M+ papers — no key required |
| **python-dotenv** | Secure API key management via .env file |
| **langdetect** | Automatic input language detection |

---

## 🤖 Agent Pipeline

The system runs 5 agents in strict sequence. Each agent receives the output of the previous agent as its input.

| # | Agent | Role |
|---|---|---|
| 01 | 🌍 **Multilingual Agent** | Detects input language · translates query to English · translates final output to selected language |
| 02 | 🔍 **Search Agent** | Searches arXiv, Wikipedia, and CrossRef simultaneously · fetches relevant content |
| 03 | 📝 **Summarization Agent** | Reads all fetched content · produces a structured summary of key facts and findings |
| 04 | ✅ **Verification Agent** | Fact-checks the summary · flags inaccurate claims · validates across sources |
| 05 | 📄 **Synthesis Agent** | Writes the final report — Overview, Key Findings, Current State, Implications, Conclusion |

---

## ⚙️ Installation & Setup

### Step 1 — Navigate to your project folder
```bash
cd path/to/research_assistant
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Add your Groq API key
Open the `.env` file and paste your key:
```
GROQ_API_KEY=your_groq_api_key_here
```
> Get a free key at 👉 https://console.groq.com

### Step 4 — Run the application
```bash
streamlit run app.py
```
> The app opens automatically at **http://localhost:8501**

---

## 🌍 Supported Languages

| Language | Native Script | Input | Output |
|---|---|---|---|
| English | English | ✅ | ✅ |
| Hindi | हिंदी | ✅ | ✅ |
| Tamil | தமிழ் | ✅ | ✅ |
| Telugu | తెలుగు | ✅ | ✅ |
| Korean | 한국어 | ✅ | ✅ |
| French | Français | ✅ | ✅ |

---

## 📚 Data Sources

| Source | Description | Requires Key |
|---|---|---|
| **arXiv** | Cornell University open-access research repository | ❌ No |
| **Wikipedia** | Free encyclopedia with live opensearch API | ❌ No |
| **CrossRef** | Scholarly metadata for 140M+ journal publications | ❌ No |

> **Note:** Raw source papers are always in English (published by researchers worldwide). Agent 3 onwards translates all AI-generated outputs to your selected language.

---

## 📦 Requirements

```
streamlit==1.42.0
groq==0.13.1
requests==2.32.3
python-dotenv==1.0.1
langdetect==1.0.9
wikipedia-api==0.7.1
arxiv==2.1.3
```

---

## 🔄 How It Works

```
User types question + selects language
            ↓
      Streamlit UI (app.py)
            ↓
      Orchestrator (orchestrator.py)
            ↓
   ┌─────────────────────────┐
   │     5-Agent Pipeline    │
   │  01 → Multilingual      │
   │  02 → Search            │
   │  03 → Summarization     │
   │  04 → Verification      │
   │  05 → Synthesis         │
   └─────────────────────────┘
            ↓
   Groq API — Llama 3.3 70B
            ↓
   Final Research Report
   (in selected language)
```

---

## ⚠️ Important Notes

**Groq API Key**
- Required to run the application
- Get for free at https://console.groq.com
- Add to `.env` file — never commit this file to GitHub

**CrossRef / arXiv**
- Both are free APIs with no registration required
- If CrossRef returns no results, wait a few seconds and try again (rate limit)

**Language Behavior**
- Agents 1–2: query is processed in English internally
- Agents 3–5: all outputs are generated in your selected language
- Final report is always in the language you selected

---

