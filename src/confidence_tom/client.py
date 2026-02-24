import os
from typing import Optional, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Wrapper for interacting with OpenAI or OpenRouter."""

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 2048) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Check if it's an OpenRouter model (contains a slash, e.g., google/gemma-2-9b-it)
        if "/" in self.model:
            base_url = "https://openrouter.ai/api/v1"
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
        else:
            base_url = "https://api.openai.com/v1"
            api_key = os.environ.get("OPENAI_API_KEY", "")

        if not api_key:
            pass  # Could raise warning, but defer it to standard error if API gets called.

        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate_parsed(
        self, messages: list[dict[str, str]], response_model: Type[T]
    ) -> Optional[T]:
        """Generates a structured response strictly matching the Pydantic schema."""
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format=response_model,
            )
            return response.choices[0].message.parsed  # type: ignore[no-any-return]
        except Exception as e:
            print(f"Error generating parsed LLM response: {e}")
            return None

    def generate_text(self, messages: list[dict[str, str]]) -> str:
        """Generates a plain text response."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
            return content if content else ""
        except Exception as e:
            print(f"Error generating text LLM response: {e}")
            return ""
