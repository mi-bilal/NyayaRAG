from __future__ import annotations

from groq import Groq

from nyayarag.config import Settings


class GroqGenerator:
    def __init__(self, settings: Settings):
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing. Add it to .env.")
        self.settings = settings
        self.client = Groq(api_key=settings.groq_api_key)

    def complete_json(self, system: str, user: str, temperature: float = 0.1) -> str:
        response = self.client.chat.completions.create(
            model=self.settings.groq_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or "{}"
