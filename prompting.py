from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.prompting_service import build_all_prompts

router = APIRouter()


class PromptRequest(BaseModel):
    context: str
    question: str
    role: str = "domain expert"


@router.post("/generate")
async def generate_prompts(req: PromptRequest):
    if not req.context.strip():
        raise HTTPException(400, "Context cannot be empty.")
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    strategies = build_all_prompts(req.context, req.question, req.role)
    return {
        "question": req.question,
        "context_preview": req.context[:300] + "..." if len(req.context) > 300 else req.context,
        "strategies": strategies,
    }