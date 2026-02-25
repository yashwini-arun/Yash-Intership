from typing import Dict, Any


PROMPT_META = {
    "zero_shot": {
        "name": "Zero-Shot",
        "color": "#8b5cf6",
        "icon": "⚡",
        "pros": ["Simple & fast", "No examples needed", "Low token cost"],
        "cons": ["Less format control", "May miss nuance"],
        "use_when": "Simple factual Q&A, quick answers",
        "format_note": "Short, direct paragraph answer."
    },
    "few_shot": {
        "name": "Few-Shot",
        "color": "#0ea5e9",
        "icon": "🎯",
        "pros": ["Consistent output format", "Controls tone & style"],
        "cons": ["More tokens", "Example quality matters"],
        "use_when": "When output format consistency matters",
        "format_note": "Bullet-point list format."
    },
    "chain_of_thought": {
        "name": "Chain-of-Thought",
        "color": "#f97316",
        "icon": "🧠",
        "pros": ["Best accuracy", "Shows reasoning", "Handles complexity"],
        "cons": ["Slower", "More verbose"],
        "use_when": "Complex analysis, multi-step reasoning",
        "format_note": "Step-by-step reasoning then final answer."
    },
    "role_based": {
        "name": "Role-Based",
        "color": "#ec4899",
        "icon": "🎭",
        "pros": ["Domain expertise depth", "Tailored vocabulary", "Authority"],
        "cons": ["Can over-claim expertise", "Role must match domain"],
        "use_when": "Domain-specific docs: legal, medical, technical",
        "format_note": "Expert analysis with technical depth."
    },
}


def build_zero_shot(context: str, question: str) -> Dict[str, Any]:
    system = (
        "You are a concise AI assistant. "
        "Answer ONLY using the provided context. "
        "Give a SHORT, DIRECT answer in 2-3 sentences maximum. "
        "Do NOT use bullet points or numbered lists. Write in plain prose only."
    )
    user = f"""Context:
\"\"\"{context}\"\"\"

Question: {question}

Give a short direct answer in 2-3 sentences:"""
    return {"strategy": "zero_shot", "system_prompt": system, "user_prompt": user}


def build_few_shot(context: str, question: str) -> Dict[str, Any]:
    system = (
        "You are a structured AI assistant. "
        "You ALWAYS answer in bullet points — never in paragraphs. "
        "Each bullet must be a complete, specific point from the context. "
        "Use exactly 4-6 bullet points. Start each bullet with •"
    )
    user = f"""Here are two examples of the EXACT format you must follow:

Example 1:
Context: "The heart pumps blood through the body. It has four chambers. It beats about 100,000 times a day."
Question: What does the heart do?
Answer:
• Pumps blood continuously throughout the entire body
• Contains four chambers that work together
• Beats approximately 100,000 times every day
• Essential for delivering oxygen to all organs

Example 2:
Context: "Solar panels convert sunlight into electricity. They use photovoltaic cells. They work best in direct sunlight."
Question: How do solar panels work?
Answer:
• Convert sunlight directly into electrical energy
• Use photovoltaic cells as the core technology
• Perform best under direct sunlight conditions
• Provide a renewable source of electricity

Now answer using the SAME bullet format:

Context:
\"\"\"{context}\"\"\"

Question: {question}
Answer:"""
    return {"strategy": "few_shot", "system_prompt": system, "user_prompt": user}


def build_chain_of_thought(context: str, question: str) -> Dict[str, Any]:
    system = (
        "You are an analytical AI assistant. "
        "You MUST show your reasoning process before answering. "
        "Always follow this EXACT structure:\n"
        "STEP 1 - IDENTIFY: State which parts of the context are relevant\n"
        "STEP 2 - ANALYZE: What do those parts tell us?\n"
        "STEP 3 - CONNECT: How do they relate to the question?\n"
        "FINAL ANSWER: Your conclusion based on the above steps\n"
        "Never skip steps. Never merge steps."
    )
    user = f"""Context:
\"\"\"{context}\"\"\"

Question: {question}

Work through this step by step:

STEP 1 - IDENTIFY (which parts of the context relate to the question?):
STEP 2 - ANALYZE (what do those parts tell us?):
STEP 3 - CONNECT (how does this answer the question?):
FINAL ANSWER:"""
    return {"strategy": "chain_of_thought", "system_prompt": system, "user_prompt": user}


def build_role_based(context: str, question: str, role: str = "domain expert") -> Dict[str, Any]:
    system = (
        f"You are a senior {role} with 15+ years of experience. "
        f"You speak with authority and use technical/professional language appropriate to your field. "
        f"Structure your answer as: 1) Expert Overview, 2) Key Technical Details, 3) Professional Recommendation. "
        f"Use domain-specific terminology. Be detailed and thorough — minimum 4-5 sentences."
    )
    user = f"""As a senior {role}, provide an expert analysis based strictly on this document:

Document:
\"\"\"{context}\"\"\"

Question: {question}

Provide your expert analysis with: Overview, Key Details, and Professional Recommendation:"""
    return {"strategy": "role_based", "system_prompt": system, "user_prompt": user}


BUILDERS = {
    "zero_shot":        build_zero_shot,
    "few_shot":         build_few_shot,
    "chain_of_thought": build_chain_of_thought,
    "role_based":       build_role_based,
}


def build_all_prompts(context: str, question: str, role: str = "domain expert") -> Dict[str, Any]:
    results = {}
    for key, fn in BUILDERS.items():
        if key == "role_based":
            data = fn(context, question, role)
        else:
            data = fn(context, question)
        full = f"[SYSTEM]\n{data['system_prompt']}\n\n[USER]\n{data['user_prompt']}"
        results[key] = {
            **data,
            "full_prompt":     full,
            "token_estimate":  len(full.split()),
            "meta":            PROMPT_META[key],
        }
    return results