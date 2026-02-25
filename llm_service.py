import os
from groq import Groq
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama3-8b-8192")


class LLMService:
    _client: Optional[Groq] = None

    @classmethod
    def get_client(cls) -> Groq:
        if cls._client is None:
            if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
                raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")
            cls._client = Groq(api_key=GROQ_API_KEY)
        return cls._client

    @classmethod
    def complete(cls, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> Dict[str, Any]:
        """Send a prompt to Groq and return the response."""
        client = cls.get_client()

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=1024,
        )

        return {
            "answer": response.choices[0].message.content,
            "model": response.model,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    @classmethod
    def is_configured(cls) -> bool:
        return bool(GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here")