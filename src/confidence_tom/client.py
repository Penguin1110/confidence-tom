import os
from typing import Optional, Type, TypeVar

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI, RateLimitError
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Wrapper for interacting with OpenAI or OpenRouter."""

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 2048) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        base_url = "https://openrouter.ai/api/v1"
        api_key = os.environ.get("OPENROUTER_API_KEY", "")

        if not api_key:
            print("WARNING: OPENROUTER_API_KEY not found in environment!")

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.aclient = AsyncOpenAI(base_url=base_url, api_key=api_key)

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

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((RateLimitError, Exception)),
    )  # type: ignore
    async def agenerate_parsed(
        self, messages: list[dict[str, str]], response_model: Type[T]
    ) -> Optional[T]:
        """Asynchronously generates a structured response matching the Pydantic schema."""
        try:
            response = await self.aclient.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format=response_model,
            )
            return response.choices[0].message.parsed  # type: ignore[no-any-return]
        except RateLimitError as e:
            # We explicitly raise this to trigger tenacity retry
            print(f"Rate limit hit for {self.model}, retrying... ({e})")
            raise e
        except Exception as e:
            if "429" in str(e):
                print(f"Rate limit hit for {self.model}, retrying... ({e})")
                raise e
            print(f"Error agenerating parsed LLM response: {e}")
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
