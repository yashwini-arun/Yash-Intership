# FoodJudge API

**LLM-as-a-Judge evaluation backend for AI food assistant responses**

FoodJudge API is a production-ready evaluation service that judges the quality of AI-generated food assistant responses using a multi-layered evaluation pipeline. It combines three complementary techniques — **LLM-as-a-Judge**, **G-Eval (Chain-of-Thought Scoring)**, and **BERTScore (Semantic Similarity)** — to deliver structured, explainable, and quantifiable quality scores across five food-specific domains.

Built with FastAPI and powered by Groq's ultra-fast LLaMA 3.1 inference, FoodJudge is designed for developers, researchers, and product teams building or benchmarking AI food assistants.

---

## How It Works

FoodJudge evaluates an AI response by acting as a judge itself — sending the response to a second LLM (Groq/LLaMA 3.1) along with a structured evaluation prompt. This is the **LLM-as-a-Judge** paradigm.

The judge is further guided by the **G-Eval** methodology, which instructs the model to reason step-by-step through domain-specific criteria *before* assigning any numeric score. This chain-of-thought approach produces more calibrated, human-aligned scores compared to direct scoring.

Optionally, when a reference (gold standard) answer is supplied, **BERTScore** computes semantic similarity between the AI response and the reference using contextual embeddings — providing an objective, model-free signal to complement the LLM judge.

```
User AI Response
       │
       ▼
┌──────────────────────────────────────────────┐
│              FoodJudge API                   │
│                                              │
│  ┌─────────────────────────────────────┐     │
│  │   Domain Router (Auto / Explicit)   │     │
│  └──────────────────┬──────────────────┘     │
│                     │                        │
│         ┌───────────┴───────────┐            │
│         ▼                       ▼            │
│  ┌─────────────┐       ┌──────────────────┐  │
│  │   G-Eval    │       │   BERTScore      │  │
│  │  Prompt     │       │  (if reference   │  │
│  │  Builder    │       │   provided)      │  │
│  └──────┬──────┘       └────────┬─────────┘  │
│         │                       │            │
│         ▼                       │            │
│  ┌─────────────┐                │            │
│  │  Groq API   │                │            │
│  │ LLaMA 3.1   │                │            │
│  └──────┬──────┘                │            │
│         │                       │            │
│         └──────────┬────────────┘            │
│                    ▼                         │
│           ┌─────────────────┐                │
│           │  EvalResponse   │                │
│           │  (score, verdict│                │
│           │  strengths,     │                │
│           │  issues, ...)   │                │
│           └─────────────────┘                │
└──────────────────────────────────────────────┘
```

---

## Evaluation Architecture

FoodJudge uses three evaluation layers working in concert:

### 1. LLM-as-a-Judge
A second LLM instance acts as an expert judge, reviewing the AI response against the user's question and domain context. The judge is role-prompted as a domain expert (culinary expert, food safety specialist, registered dietitian, professional chef) to maximise evaluation quality.

### 2. G-Eval (Chain-of-Thought Scoring)
Before assigning any score, the LLM judge is instructed to reason through a domain-specific 5-step thought process. Each step targets a different evaluation dimension — accuracy, risk, completeness, audience suitability, and so on. Only after completing this reasoning does the model assign per-criterion scores (1–10). This methodology is directly inspired by the [G-Eval paper (Liu et al., 2023)](https://arxiv.org/abs/2303.16634).

### 3. BERTScore (Semantic Similarity)
When a `reference_answer` is provided, BERTScore measures the semantic overlap between the AI response and the reference using BERT contextual embeddings. It produces Precision, Recall, and F1 scores, with `F1 >= 0.85` as the pass threshold. This is an objective, reference-based signal that works independently of the LLM judge.

---

## Features

- Five specialized evaluation domains — Recipe, Allergy & Safety, Ingredient Substitution, Meal Plan, Cooking Technique
- Intelligent auto-routing endpoint that detects the domain from keywords
- G-Eval methodology with step-by-step chain-of-thought reasoning before scoring
- Optional BERTScore semantic similarity against a reference answer
- Per-criterion scores with human-readable reasoning for each
- Structured JSON responses with overall score, verdict, strengths, issues, and improvement suggestion
- High-stakes safety mode for allergy evaluations with stricter scoring
- Groq-powered inference for low-latency LLM calls
- CORS middleware enabled for direct frontend integration
- Graceful BERTScore degradation — evaluation proceeds even if `bert-score` is not installed

---

## Tech Stack

| Component | Technology | Version |
|---|---|---|
| Web Framework | FastAPI | 0.115.0 |
| ASGI Server | Uvicorn | 0.30.6 |
| LLM Provider | Groq API (LLaMA 3.1 8B Instant) | — |
| HTTP Client | HTTPX | 0.27.2 |
| Semantic Scoring | BERTScore | 0.3.13 |
| ML Backend | PyTorch | ≥ 2.0.0 |
| Transformers | HuggingFace Transformers | ≥ 4.30.0 |
| Data Validation | Pydantic | v2.9.2 |
| Config Management | python-dotenv | 1.0.1 |

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- A [Groq API key](https://console.groq.com/) (free tier available)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/foodjudge-api.git
cd foodjudge-api

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

> **Note:** Installing `torch` and `transformers` for BERTScore may take a few minutes. If you want to skip BERTScore support and run a lightweight version, you can install only the core dependencies:
> ```bash
> pip install fastapi uvicorn httpx pydantic python-dotenv
> ```
> BERTScore will be automatically disabled and evaluations will proceed using G-Eval only.

### Running the Server

```bash
uvicorn main:app --reload
```

| URL | Description |
|---|---|
| `http://localhost:8000` | API root — service info and endpoint list |
| `http://localhost:8000/docs` | Interactive Swagger UI |
| `http://localhost:8000/redoc` | ReDoc documentation |

---

## Configuration

Create a `.env` file in the project root before starting the server:

```env
GROQ_API_KEY=your-groq-api-key-here
```

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | API key from [console.groq.com](https://console.groq.com/) |

---

## API Reference

### `GET /`

Returns service metadata including the active model, provider, available endpoints, and BERTScore availability status.

**Response example:**
```json
{
  "service": "FoodJudge API",
  "version": "2.0.0",
  "model": "llama-3.1-8b-instant",
  "provider": "Groq",
  "features": ["G-Eval (chain-of-thought scoring)", "BERTScore (semantic similarity)"],
  "bert_score_available": true,
  "endpoints": [
    "POST /evaluate/recipe",
    "POST /evaluate/allergy",
    "POST /evaluate/substitution",
    "POST /evaluate/mealplan",
    "POST /evaluate/technique",
    "POST /evaluate/auto"
  ]
}
```

---

### Common Request Fields

All evaluation endpoints share these base fields:

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `user_question` | string | Yes | — | The original question posed by the user |
| `ai_response` | string | Yes | — | The AI-generated response to evaluate |
| `dietary_context` | string | No | `"none"` | Dietary restrictions, e.g. `"vegan"`, `"gluten-free"`, `"halal"` |
| `audience` | string | No | `"home_cook"` | Target audience, e.g. `"home_cook"`, `"professional_chef"`, `"beginner"` |
| `reference_answer` | string | No | `null` | Gold standard answer — activates BERTScore when provided |

---

## Evaluation Domains

### `POST /evaluate/recipe`

Evaluates AI-generated recipe instructions from a culinary expert perspective.

**Additional fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `cuisine_type` | string | `"general"` | Cuisine context, e.g. `"Italian"`, `"Indian"`, `"Middle Eastern"` |

**Criteria scored:** `accuracy` · `clarity` · `safety` · `completeness`

**Example request:**
```json
{
  "user_question": "How do I make a classic shakshuka?",
  "ai_response": "Heat olive oil in a pan. Sauté onions, garlic, and bell peppers...",
  "cuisine_type": "Middle Eastern",
  "dietary_context": "vegetarian",
  "audience": "home_cook",
  "reference_answer": "Optional gold standard answer for BERTScore"
}
```

---

### `POST /evaluate/allergy`

Evaluates AI allergy and food safety responses. **This is a high-stakes domain** — the judge is explicitly prompted to be very strict on safety criteria, as incorrect advice can cause anaphylaxis.

**Additional fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `allergen` | string | `null` | The specific allergen in question, e.g. `"peanuts"`, `"gluten"` |
| `severity` | string | `"unknown"` | Allergy severity, e.g. `"mild"`, `"severe"`, `"anaphylactic"` |

**Criteria scored:** `medical_accuracy` · `risk_communication` · `safety` · `actionability`

---

### `POST /evaluate/substitution`

Evaluates AI ingredient substitution suggestions from a culinary science perspective.

**Additional fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `original_ingredient` | string | `null` | The ingredient being replaced, e.g. `"butter"`, `"eggs"` |
| `recipe_type` | string | `"general"` | Type of recipe the substitution applies to |

**Criteria scored:** `culinary_accuracy` · `taste_impact` · `diet_compliance` · `practicality`

---

### `POST /evaluate/mealplan`

Evaluates AI-generated meal plans from a registered dietitian perspective.

**Additional fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `duration_days` | integer | `3` | Number of days the meal plan covers |
| `calorie_target` | integer | `null` | Daily calorie target in kcal |

**Criteria scored:** `nutrition_balance` · `variety` · `diet_compliance` · `practicality`

---

### `POST /evaluate/technique`

Evaluates AI cooking technique explanations from a professional chef perspective.

**Additional fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `technique_name` | string | `null` | The technique being explained, e.g. `"braising"`, `"emulsifying"` |
| `skill_level` | string | `"beginner"` | Skill level of the target audience |

**Criteria scored:** `technical_accuracy` · `clarity` · `safety_tips` · `helpfulness`

---

### `POST /evaluate/auto`

Automatically detects the evaluation domain from keywords in the question and response, then routes to the appropriate evaluator. Accepts the same base request fields — no domain-specific fields required.

**Detection priority:**

| Priority | Domain | Trigger Keywords |
|---|---|---|
| 1 | Allergy | `allergy`, `allergic`, `intolerance`, `anaphylaxis`, `celiac`, `safe to eat` |
| 2 | Substitution | `substitute`, `replace`, `instead of`, `swap`, `alternative`, `without` |
| 3 | Meal Plan | `meal plan`, `weekly plan`, `daily plan`, `diet plan`, `days of food` |
| 4 | Technique | `how to`, `technique`, `sear`, `blanch`, `braise`, `emulsify`, `knead` |
| 5 (default) | Recipe | *(any other food query)* |

---

## Response Schema

All endpoints return a unified `EvalResponse` object:

```json
{
  "eval_type": "recipe",
  "overall_score": 8,
  "passed": true,
  "verdict": "The response provides accurate and well-structured instructions suitable for a home cook, though it omits key food safety notes around egg handling.",
  "strengths": [
    "Correct ingredient ratios and cooking temperatures",
    "Clear step-by-step structure"
  ],
  "issues": [
    "No mention of safe internal temperature for poultry",
    "Missing tips for common failure points"
  ],
  "improvement": "Add a note about cooking chicken to an internal temperature of 74°C (165°F) for food safety.",
  "criteria": {
    "accuracy": {
      "score": 9,
      "reason": "Ingredient ratios and cooking times are correct for this cuisine."
    },
    "clarity": {
      "score": 8,
      "reason": "Instructions are easy to follow but could benefit from numbered steps."
    },
    "safety": {
      "score": 6,
      "reason": "No food safety guidance provided for raw protein handling."
    },
    "completeness": {
      "score": 8,
      "reason": "All major steps are present; garnish and plating suggestions are missing."
    }
  },
  "bert_score": {
    "precision": 0.921,
    "recall": 0.908,
    "f1": 0.914,
    "passed": true
  }
}
```

| Field | Type | Description |
|---|---|---|
| `eval_type` | string | The evaluation domain used |
| `overall_score` | integer | Aggregate score from 1–10 |
| `passed` | boolean | `true` if `overall_score >= 7` |
| `verdict` | string | 2–3 sentence narrative summary |
| `strengths` | string[] | What the response does well |
| `issues` | string[] | Problems or gaps identified |
| `improvement` | string | One concrete, actionable fix |
| `criteria` | object | Per-criterion scores with reasoning |
| `bert_score` | object \| null | Semantic similarity result (null if no reference provided or BERTScore not installed) |

---

## BERTScore

BERTScore is activated automatically by including a `reference_answer` in any request. It computes semantic similarity between the AI response and the reference using BERT contextual token embeddings — not surface-level string matching.

| Metric | Description |
|---|---|
| `precision` | Proportion of the AI response semantically covered by the reference |
| `recall` | Proportion of the reference semantically covered by the AI response |
| `f1` | Harmonic mean of precision and recall — primary evaluation metric |
| `passed` | `true` when `f1 >= 0.85` |

BERTScore is fully optional. If `bert-score`, `torch`, or `transformers` are not installed, the field is returned as `null` and a warning is logged at startup. All other evaluation functionality remains unaffected.

---

## Project Structure

```
foodjudge-api/
├── main.py              # Application entry — routes, prompt builders, scoring logic
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (not committed to version control)
└── README.md
```
