from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="FoodJudge API",
    description="LLM-as-Judge backend for evaluating AI food assistant responses (powered by Groq) with BERTScore + G-Eval",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")
MODEL = "llama-3.1-8b-instant"

# ─────────────────────────────────────────
# BERT SCORE SETUP
# ─────────────────────────────────────────

try:
    from bert_score import score as bert_score_fn
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("WARNING: bert-score not installed. BERTScore will be skipped. Run: pip install bert-score")


# ─────────────────────────────────────────
# REQUEST SCHEMAS
# ─────────────────────────────────────────

class BaseEvalRequest(BaseModel):
    user_question: str
    ai_response: str
    dietary_context: Optional[str] = "none"
    audience: Optional[str] = "home_cook"
    reference_answer: Optional[str] = None   


class RecipeEvalRequest(BaseEvalRequest):
    cuisine_type: Optional[str] = "general"


class AllergyEvalRequest(BaseEvalRequest):
    allergen: Optional[str] = None
    severity: Optional[str] = "unknown"


class SubstitutionEvalRequest(BaseEvalRequest):
    original_ingredient: Optional[str] = None
    recipe_type: Optional[str] = "general"


class MealPlanEvalRequest(BaseEvalRequest):
    duration_days: Optional[int] = 3
    calorie_target: Optional[int] = None


class TechniqueEvalRequest(BaseEvalRequest):
    technique_name: Optional[str] = None
    skill_level: Optional[str] = "beginner"


class AutoEvalRequest(BaseEvalRequest):
    pass


# ─────────────────────────────────────────
# RESPONSE SCHEMAS
# ─────────────────────────────────────────

class CriterionScore(BaseModel):
    score: int
    reason: str


class BertScoreResult(BaseModel):          
    precision: float
    recall: float
    f1: float
    passed: bool


class EvalResponse(BaseModel):
    eval_type: str
    overall_score: int
    passed: bool
    verdict: str
    strengths: list[str]
    issues: list[str]
    improvement: str
    criteria: dict[str, CriterionScore]
    bert_score: Optional[BertScoreResult] = None   


# ─────────────────────────────────────────
# BERT SCORE
# ─────────────────────────────────────────

def compute_bert_score(ai_response: str, reference: str) -> BertScoreResult:
    """
    Computes BERTScore between ai_response and a reference (gold standard) answer.
    Returns precision, recall, F1, and a pass/fail based on F1 >= 0.85 threshold.
    """
    if not BERT_AVAILABLE:
        return None
    try:
        P, R, F1 = bert_score_fn(
            [ai_response],
            [reference],
            lang="en",
            verbose=False
        )
        f1_val = round(F1.item(), 3)
        return BertScoreResult(
            precision=round(P.item(), 3),
            recall=round(R.item(), 3),
            f1=f1_val,
            passed=f1_val >= 0.85
        )
    except Exception as e:
        print(f"BERTScore computation failed: {e}")
        return None


# ─────────────────────────────────────────
# GROQ CALLER
# ─────────────────────────────────────────

async def call_groq(prompt: str) -> dict:
    """Call Groq API and parse JSON response."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict food AI quality evaluator. Always respond with valid JSON only. No markdown, no backticks, no explanation outside the JSON object."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 1500,   
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(GROQ_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"]
        clean = raw_text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)


# ─────────────────────────────────────────
# SHARED RESPONSE BUILDER
# ─────────────────────────────────────────

def build_eval_response(
    eval_type: str,
    data: dict,
    criteria_keys: list[str],
    ai_response: str = "",
    reference_answer: Optional[str] = None
) -> EvalResponse:
    criteria = {}
    for key in criteria_keys:
        raw = data.get(key, {})
        criteria[key] = CriterionScore(
            score=int(raw.get("score", 5)),
            reason=raw.get("reason", "")
        )
    overall = int(data.get("overall_score", 5))

    # Compute BERTScore if reference answer is provided
    bert_result = None
    if reference_answer and ai_response:
        bert_result = compute_bert_score(ai_response, reference_answer)

    return EvalResponse(
        eval_type=eval_type,
        overall_score=overall,
        passed=overall >= 7,
        verdict=data.get("verdict", ""),
        strengths=data.get("strengths", []),
        issues=data.get("issues", []),
        improvement=data.get("improvement", ""),
        criteria=criteria,
        bert_score=bert_result,
    )


# ─────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────

def recipe_prompt(req: RecipeEvalRequest) -> tuple[str, list[str]]:
    keys = ["accuracy", "clarity", "safety", "completeness"]
    prompt = f"""You are a culinary expert evaluating an AI recipe response.

User Question: {req.user_question}
AI Response: {req.ai_response}
Cuisine: {req.cuisine_type}
Dietary Context: {req.dietary_context}
Audience: {req.audience}

--- G-EVAL: Think step by step before scoring ---
Step 1: What would an ideal recipe response contain for this specific question and cuisine?
Step 2: What is present in the AI response that is correct, well-explained, and helpful?
Step 3: What is missing, incorrect, or potentially unsafe in the response?
Step 4: How would a {req.audience} be affected by any gaps or errors in this response?
Step 5: Based on your reasoning above, now assign scores for each criterion.
-------------------------------------------------

Score each criterion 1-10 (be strict, 10 = exceptional):
- accuracy: correct ingredients, ratios, temperatures, times
- clarity: steps are clear, ordered, easy to follow
- safety: food safety rules followed, allergens noted
- completeness: all steps, quantities, tips included

Return ONLY this JSON:
{{"accuracy":{{"score":7,"reason":"..."}},"clarity":{{"score":7,"reason":"..."}},"safety":{{"score":7,"reason":"..."}},"completeness":{{"score":7,"reason":"..."}},"overall_score":7,"verdict":"2-3 sentence verdict","strengths":["s1","s2"],"issues":["i1","i2"],"improvement":"one specific fix"}}"""
    return prompt, keys


def allergy_prompt(req: AllergyEvalRequest) -> tuple[str, list[str]]:
    keys = ["medical_accuracy", "risk_communication", "safety", "actionability"]
    allergen_info = f"Allergen: {req.allergen}" if req.allergen else ""
    prompt = f"""You are a food safety specialist. This is HIGH STAKES — wrong advice can cause anaphylaxis.

User Question: {req.user_question}
AI Response: {req.ai_response}
{allergen_info}
Severity: {req.severity}
Dietary Context: {req.dietary_context}

--- G-EVAL: Think step by step before scoring ---
Step 1: What is the actual medical/scientific truth about this allergen or food safety concern?
Step 2: Does the AI response align with that medical truth? Where does it agree or differ?
Step 3: Is the severity of the risk clearly and accurately communicated, or is it understated?
Step 4: Would a doctor or allergist approve of this advice? What would they change?
Step 5: Based on your reasoning above, now assign scores — be VERY strict on safety.
-------------------------------------------------

Score each criterion 1-10 (be VERY strict on safety):
- medical_accuracy: is allergy/intolerance info medically correct?
- risk_communication: is risk clearly stated, not understated?
- safety: does it avoid putting user at risk, recommend doctor if needed?
- actionability: does it give clear practical steps?

Return ONLY this JSON:
{{"medical_accuracy":{{"score":7,"reason":"..."}},"risk_communication":{{"score":7,"reason":"..."}},"safety":{{"score":7,"reason":"..."}},"actionability":{{"score":7,"reason":"..."}},"overall_score":7,"verdict":"2-3 sentence verdict","strengths":["s1","s2"],"issues":["i1","i2"],"improvement":"one specific fix"}}"""
    return prompt, keys


def substitution_prompt(req: SubstitutionEvalRequest) -> tuple[str, list[str]]:
    keys = ["culinary_accuracy", "taste_impact", "diet_compliance", "practicality"]
    ingredient_info = f"Original ingredient: {req.original_ingredient}" if req.original_ingredient else ""
    prompt = f"""You are a culinary scientist evaluating ingredient substitution advice.

User Question: {req.user_question}
AI Response: {req.ai_response}
{ingredient_info}
Recipe Type: {req.recipe_type}
Dietary Context: {req.dietary_context}
Audience: {req.audience}

--- G-EVAL: Think step by step before scoring ---
Step 1: Does this substitution work chemically and culinarily? Think about ratios, binding, moisture, fat content.
Step 2: What flavor and texture changes will occur? Does the AI response accurately describe these?
Step 3: Does the substitution fully comply with the stated dietary restrictions?
Step 4: Is the substitute easy to find in a regular grocery store and practical for a {req.audience}?
Step 5: Based on your reasoning above, now assign scores.
-------------------------------------------------

Score each criterion 1-10:
- culinary_accuracy: is the substitution scientifically/culinarily sound?
- taste_impact: does it accurately describe flavor/texture changes?
- diet_compliance: does it respect dietary restrictions?
- practicality: is the substitute easy to find and use?

Return ONLY this JSON:
{{"culinary_accuracy":{{"score":7,"reason":"..."}},"taste_impact":{{"score":7,"reason":"..."}},"diet_compliance":{{"score":7,"reason":"..."}},"practicality":{{"score":7,"reason":"..."}},"overall_score":7,"verdict":"2-3 sentence verdict","strengths":["s1","s2"],"issues":["i1","i2"],"improvement":"one specific fix"}}"""
    return prompt, keys


def mealplan_prompt(req: MealPlanEvalRequest) -> tuple[str, list[str]]:
    keys = ["nutrition_balance", "variety", "diet_compliance", "practicality"]
    calorie_info = f"Calorie target: {req.calorie_target} kcal/day" if req.calorie_target else ""
    prompt = f"""You are a registered dietitian evaluating an AI-generated meal plan.

User Question: {req.user_question}
AI Response: {req.ai_response}
Duration: {req.duration_days} days
{calorie_info}
Dietary Context: {req.dietary_context}
Audience: {req.audience}

--- G-EVAL: Think step by step before scoring ---
Step 1: Are macronutrients (protein, carbs, fats) and micronutrients (vitamins, minerals) balanced across all {req.duration_days} days?
Step 2: Is there enough meal variety to avoid monotony and prevent nutritional gaps?
Step 3: Does every single meal fully comply with the dietary requirements: {req.dietary_context}?
Step 4: Is this meal plan realistic and practical for a {req.audience} to actually prepare daily?
Step 5: Based on your reasoning above, now assign scores.
-------------------------------------------------

Score each criterion 1-10:
- nutrition_balance: balanced macros and micros across the plan?
- variety: enough variety to avoid monotony and nutrient gaps?
- diet_compliance: fully complies with dietary requirements?
- practicality: realistic to prepare for stated audience?

Return ONLY this JSON:
{{"nutrition_balance":{{"score":7,"reason":"..."}},"variety":{{"score":7,"reason":"..."}},"diet_compliance":{{"score":7,"reason":"..."}},"practicality":{{"score":7,"reason":"..."}},"overall_score":7,"verdict":"2-3 sentence verdict","strengths":["s1","s2"],"issues":["i1","i2"],"improvement":"one specific fix"}}"""
    return prompt, keys


def technique_prompt(req: TechniqueEvalRequest) -> tuple[str, list[str]]:
    keys = ["technical_accuracy", "clarity", "safety_tips", "helpfulness"]
    technique_info = f"Technique: {req.technique_name}" if req.technique_name else ""
    prompt = f"""You are a professional chef evaluating a cooking technique explanation.

User Question: {req.user_question}
AI Response: {req.ai_response}
{technique_info}
Skill Level: {req.skill_level}
Dietary Context: {req.dietary_context}

--- G-EVAL: Think step by step before scoring ---
Step 1: Is the technique description scientifically and culinarily correct? Check temperatures, timing, and equipment.
Step 2: Can a {req.skill_level} cook successfully follow these instructions without prior knowledge of this technique?
Step 3: What could go wrong if someone follows this advice? Are those risks mentioned in the response?
Step 4: After following this advice, will the user actually succeed at the technique? What is missing?
Step 5: Based on your reasoning above, now assign scores.
-------------------------------------------------

Score each criterion 1-10:
- technical_accuracy: correct temperatures, timing, equipment?
- clarity: easy to follow for the stated skill level?
- safety_tips: relevant safety warnings included?
- helpfulness: does it actually help the user succeed?

Return ONLY this JSON:
{{"technical_accuracy":{{"score":7,"reason":"..."}},"clarity":{{"score":7,"reason":"..."}},"safety_tips":{{"score":7,"reason":"..."}},"helpfulness":{{"score":7,"reason":"..."}},"overall_score":7,"verdict":"2-3 sentence verdict","strengths":["s1","s2"],"issues":["i1","i2"],"improvement":"one specific fix"}}"""
    return prompt, keys


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "FoodJudge API",
        "version": "2.0.0",
        "model": MODEL,
        "provider": "Groq",
        "features": ["G-Eval (chain-of-thought scoring)", "BERTScore (semantic similarity)"],
        "bert_score_available": BERT_AVAILABLE,
        "endpoints": [
            "POST /evaluate/recipe",
            "POST /evaluate/allergy",
            "POST /evaluate/substitution",
            "POST /evaluate/mealplan",
            "POST /evaluate/technique",
            "POST /evaluate/auto",
        ]
    }


@app.post("/evaluate/recipe", response_model=EvalResponse)
async def evaluate_recipe(req: RecipeEvalRequest):
    """Evaluate AI-generated recipe instructions. Supports G-Eval + optional BERTScore via reference_answer."""
    try:
        prompt, keys = recipe_prompt(req)
        data = await call_groq(prompt)
        return build_eval_response("recipe", data, keys, req.ai_response, req.reference_answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate/allergy", response_model=EvalResponse)
async def evaluate_allergy(req: AllergyEvalRequest):
    """Evaluate AI allergy & food safety responses (HIGH STAKES). Supports G-Eval + optional BERTScore."""
    try:
        prompt, keys = allergy_prompt(req)
        data = await call_groq(prompt)
        return build_eval_response("allergy", data, keys, req.ai_response, req.reference_answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate/substitution", response_model=EvalResponse)
async def evaluate_substitution(req: SubstitutionEvalRequest):
    """Evaluate AI ingredient substitution suggestions. Supports G-Eval + optional BERTScore."""
    try:
        prompt, keys = substitution_prompt(req)
        data = await call_groq(prompt)
        return build_eval_response("substitution", data, keys, req.ai_response, req.reference_answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate/mealplan", response_model=EvalResponse)
async def evaluate_mealplan(req: MealPlanEvalRequest):
    """Evaluate AI-generated meal plans. Supports G-Eval + optional BERTScore."""
    try:
        prompt, keys = mealplan_prompt(req)
        data = await call_groq(prompt)
        return build_eval_response("mealplan", data, keys, req.ai_response, req.reference_answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate/technique", response_model=EvalResponse)
async def evaluate_technique(req: TechniqueEvalRequest):
    """Evaluate AI cooking technique explanations. Supports G-Eval + optional BERTScore."""
    try:
        prompt, keys = technique_prompt(req)
        data = await call_groq(prompt)
        return build_eval_response("technique", data, keys, req.ai_response, req.reference_answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate/auto", response_model=EvalResponse)
async def evaluate_auto(req: AutoEvalRequest):
    """Auto-detects eval type from question keywords and routes accordingly. Supports G-Eval + optional BERTScore."""
    combined = (req.user_question + " " + req.ai_response).lower()

    allergy_keywords = ["allergy", "allergic", "intolerance", "anaphylaxis", "safe to eat", "celiac", "gluten free"]
    substitution_keywords = ["substitute", "replace", "instead of", "swap", "alternative", "without"]
    mealplan_keywords = ["meal plan", "weekly plan", "daily plan", "diet plan", "days of food"]
    technique_keywords = ["how to", "technique", "sear", "blanch", "braise", "emulsify", "knead", "caramelize"]

    if any(k in combined for k in allergy_keywords):
        prompt, keys = allergy_prompt(AllergyEvalRequest(**req.dict()))
        eval_type = "allergy"
    elif any(k in combined for k in substitution_keywords):
        prompt, keys = substitution_prompt(SubstitutionEvalRequest(**req.dict()))
        eval_type = "substitution"
    elif any(k in combined for k in mealplan_keywords):
        prompt, keys = mealplan_prompt(MealPlanEvalRequest(**req.dict()))
        eval_type = "mealplan"
    elif any(k in combined for k in technique_keywords):
        prompt, keys = technique_prompt(TechniqueEvalRequest(**req.dict()))
        eval_type = "technique"
    else:
        prompt, keys = recipe_prompt(RecipeEvalRequest(**req.dict()))
        eval_type = "recipe"

    try:
        data = await call_groq(prompt)
        return build_eval_response(eval_type, data, keys, req.ai_response, req.reference_answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))