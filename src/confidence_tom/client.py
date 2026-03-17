import json as _json
import os
import re as _re
import uuid
from typing import Any, Optional, Type, TypeVar

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI, RateLimitError
from pydantic import BaseModel
from tenacity import RetryError, retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class _RateLimitOrQuota(Exception):
    """Sentinel used by tenacity: raised only when we actually want a retry."""

from confidence_tom.task_models import ApiTrace

load_dotenv()

T = TypeVar("T", bound=BaseModel)


def _normalize_chat_message(message: dict[str, Any]) -> dict[str, Any]:
    """Ensure provider-facing chat messages always have a string content field."""
    normalized = dict(message)
    if normalized.get("role") == "assistant" and normalized.get("tool_calls"):
        normalized["content"] = normalized.get("content") or ""
    elif "content" in normalized and normalized["content"] is None:
        normalized["content"] = ""
    return normalized


def _extract_trace(response: object) -> ApiTrace:
    """Pull all metadata fields out of a raw OpenAI/OpenRouter response object."""
    usage = getattr(response, "usage", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    choices = getattr(response, "choices", [])
    msg = choices[0].message if choices else None

    # reasoning content is exposed by some models (DeepSeek-R1, etc.)
    reasoning_content = getattr(msg, "reasoning", None) or ""
    raw_content = getattr(msg, "content", None) or ""
    if isinstance(raw_content, list):
        raw_content = _json.dumps(raw_content, ensure_ascii=False)
    else:
        raw_content = str(raw_content)

    # cache_read: OpenRouter puts it in prompt_tokens_details.cached_tokens
    # cache_write: prompt_tokens_details.cache_write_tokens
    cache_read = (
        getattr(prompt_details, "cached_tokens", 0)
        or getattr(usage, "cache_read_tokens", 0)
        or 0
    )
    cache_write = (
        getattr(prompt_details, "cache_write_tokens", 0)
        or getattr(usage, "cache_write_tokens", 0)
        or 0
    )

    return ApiTrace(
        model_id=getattr(response, "model", ""),
        request_id=getattr(response, "id", ""),
        reasoning_tokens=getattr(completion_details, "reasoning_tokens", 0) or 0,
        reasoning_content=reasoning_content,
        response_content=raw_content,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )


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

    async def agenerate_parsed(
        self, messages: list[dict[str, str]], response_model: Type[T]
    ) -> Optional[T]:
        """Asynchronously generates a structured response matching the Pydantic schema."""
        try:
            return await self._agenerate_parsed_inner(messages, response_model)
        except RetryError:
            print(f"Rate limit retries exhausted for {self.model}, skipping sample.")
            return None

    @retry(
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=2, min=5, max=120),
        retry=retry_if_exception_type(_RateLimitOrQuota),
    )  # type: ignore
    async def _agenerate_parsed_inner(
        self, messages: list[dict[str, str]], response_model: Type[T]
    ) -> Optional[T]:
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
            print(f"Rate limit hit for {self.model}, retrying... ({e})")
            raise _RateLimitOrQuota(str(e)) from e
        except Exception as e:
            if "429" in str(e):
                print(f"Rate limit hit for {self.model}, retrying... ({e})")
                raise _RateLimitOrQuota(str(e)) from e
            print(f"Error agenerating parsed LLM response: {e}")
            return None

    async def agenerate_with_trace(
        self, messages: list[dict[str, str]], response_model: Type[T]
    ) -> tuple[Optional[T], ApiTrace]:
        """Like agenerate_parsed but also returns the full ApiTrace metadata."""
        try:
            return await self._agenerate_with_trace_inner(messages, response_model)
        except RetryError as e:
            print(f"Rate limit retries exhausted for {self.model}, skipping sample.")
            return None, ApiTrace()

    @retry(
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=2, min=5, max=120),
        retry=retry_if_exception_type(_RateLimitOrQuota),
    )  # type: ignore
    async def _agenerate_with_trace_inner(
        self, messages: list[dict[str, str]], response_model: Type[T]
    ) -> tuple[Optional[T], ApiTrace]:
        try:
            response = await self.aclient.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format=response_model,
            )
            parsed = response.choices[0].message.parsed  # type: ignore[no-any-return]
            return parsed, _extract_trace(response)
        except RateLimitError as e:
            print(f"Rate limit hit for {self.model}, retrying... ({e})")
            raise _RateLimitOrQuota(str(e)) from e
        except Exception as e:
            if "429" in str(e):
                print(f"Rate limit hit for {self.model}, retrying... ({e})")
                raise _RateLimitOrQuota(str(e)) from e
            print(f"Error in agenerate_with_trace for {self.model}: {e}")
            return None, ApiTrace()

    async def agenerate_tool_message(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> tuple[Optional[dict[str, Any]], ApiTrace]:
        """Generate one assistant message with OpenAI-compatible tool calling."""
        try:
            return await self._agenerate_tool_message_inner(messages, tools)
        except RetryError:
            print(f"Rate limit retries exhausted for {self.model}, skipping tool step.")
            return None, ApiTrace()

    @retry(
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=2, min=5, max=120),
        retry=retry_if_exception_type(_RateLimitOrQuota),
    )  # type: ignore
    async def _agenerate_tool_message_inner(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> tuple[Optional[dict[str, Any]], ApiTrace]:
        try:
            normalized_messages = [_normalize_chat_message(message) for message in messages]
            response = await self.aclient.chat.completions.create(
                model=self.model,
                messages=normalized_messages,
                tools=tools,
                tool_choice="auto",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            message = response.choices[0].message
            return _normalize_chat_message(message.model_dump()), _extract_trace(response)
        except RateLimitError as e:
            print(f"Rate limit hit for {self.model}, retrying... ({e})")
            raise _RateLimitOrQuota(str(e)) from e
        except Exception as e:
            if "429" in str(e):
                print(f"Rate limit hit for {self.model}, retrying... ({e})")
                raise _RateLimitOrQuota(str(e)) from e
            print(f"Error in agenerate_tool_message for {self.model}: {e}")
            return None, ApiTrace()

    async def agenerate_react_message(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> tuple[Optional[dict[str, Any]], "ApiTrace"]:
        """Text-based fallback for models without native tool-calling support.

        Injects tool schemas into the system prompt, calls plain-text completion,
        parses a JSON action from the response, and returns an OpenAI-compatible
        message dict (with synthetic tool_calls if the model picked a tool).
        """
        try:
            return await self._agenerate_react_message_inner(messages, tools)
        except RetryError:
            print(f"Rate limit retries exhausted for {self.model}, skipping tool step.")
            return None, ApiTrace()

    @retry(
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=2, min=5, max=120),
        retry=retry_if_exception_type(_RateLimitOrQuota),
    )  # type: ignore
    async def _agenerate_react_message_inner(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> tuple[Optional[dict[str, Any]], "ApiTrace"]:
        # Build a compact tool reference for the injected system instruction
        tool_lines: list[str] = []
        for t in tools:
            fn = t.get("function", {})
            name = fn.get("name", "")
            desc = fn.get("description", "").strip()
            params = fn.get("parameters", {})
            props = params.get("properties", {}) if isinstance(params, dict) else {}
            required = params.get("required", []) if isinstance(params, dict) else []
            args_desc = ", ".join(
                f"{k}{'*' if k in required else ''}: {v.get('type', 'str')}"
                for k, v in props.items()
            )
            tool_lines.append(f"  {name}({args_desc}) — {desc}")

        react_instruction = (
            "TOOL USE INSTRUCTIONS:\n"
            "You must respond with exactly one JSON object on a single line.\n"
            'To call a tool:   {"name": "<tool_name>", "arguments": {"key": "value", ...}}\n'
            'To reply to user: {"name": "respond", "arguments": {"content": "<your message>"}}\n\n'
            "Available tools:\n" + "\n".join(tool_lines) + "\n"
        )

        # Sanitize message history: convert tool-call artifacts to plain text so
        # that models without native tool support don't get rejected at the API level.
        sanitized: list[dict[str, Any]] = []
        for msg in messages:
            msg = _normalize_chat_message(msg)
            role = msg.get("role", "")
            if role == "tool":
                # Tool result → user message so the model sees the observation
                sanitized.append({
                    "role": "user",
                    "content": f"Tool result ({msg.get('name', 'tool')}): {msg.get('content', '')}",
                })
            elif role == "assistant" and msg.get("tool_calls"):
                # Synthesized tool call → plain assistant text so the model sees its own action
                tool_call = msg["tool_calls"][0]
                fn = tool_call.get("function", {})
                sanitized.append({
                    "role": "assistant",
                    "content": f"[called {fn.get('name', 'tool')}({fn.get('arguments', '')})]",
                })
            else:
                sanitized.append(msg)

        # Inject instruction at the top of the system message (non-destructive copy)
        injected = list(sanitized)
        if injected and injected[0].get("role") == "system":
            injected[0] = {
                **injected[0],
                "content": react_instruction + "\n\n" + str(injected[0].get("content", "")),
            }
        else:
            injected.insert(0, {"role": "system", "content": react_instruction})

        try:
            response = await self.aclient.chat.completions.create(
                model=self.model,
                messages=injected,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            raw = (response.choices[0].message.content or "").strip()
            trace = _extract_trace(response)

            # Extract first JSON object (may be wrapped in ```json ... ```)
            json_match = _re.search(r'\{[^{}]*\}', raw, _re.DOTALL)
            if not json_match:
                return {"role": "assistant", "content": raw, "tool_calls": None}, trace

            try:
                parsed = _json.loads(json_match.group())
            except _json.JSONDecodeError:
                return {"role": "assistant", "content": raw, "tool_calls": None}, trace

            action_name = str(parsed.get("name", "respond"))
            arguments = parsed.get("arguments", {})

            if action_name == "respond":
                return {
                    "role": "assistant",
                    "content": arguments.get("content", raw),
                    "tool_calls": None,
                }, trace

            # Synthesize OpenAI-compatible tool_calls
            tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": action_name,
                        "arguments": _json.dumps(arguments),
                    },
                }],
            }, trace

        except RateLimitError as e:
            print(f"Rate limit hit for {self.model}, retrying... ({e})")
            raise _RateLimitOrQuota(str(e)) from e
        except Exception as e:
            if "429" in str(e):
                print(f"Rate limit hit for {self.model}, retrying... ({e})")
                raise _RateLimitOrQuota(str(e)) from e
            print(f"Error in agenerate_react_message for {self.model}: {e}")
            return None, ApiTrace()

    async def aelicit_confidence(
        self,
        messages: list[dict[str, Any]],
    ) -> Optional[float]:
        """Ask the model to self-report its confidence on the just-completed task.

        Appends a single confidence question to the conversation and parses the
        integer reply (0-100) into a [0, 1] float. Returns None on failure.
        """
        # Keep only system + last 8 messages to avoid blowing the context window
        # on long multi-step benchmarks (plancraft, tau_bench with many turns).
        elicit_raw = list(messages)
        system_msgs = [m for m in elicit_raw if m.get("role") == "system"]
        non_system  = [m for m in elicit_raw if m.get("role") != "system"]
        elicit_raw  = system_msgs + non_system[-8:]

        elicit_messages = [_normalize_chat_message(m) for m in elicit_raw]
        # Strip tool artifacts so the elicitation call works on any model
        sanitized: list[dict[str, Any]] = []
        for msg in elicit_messages:
            role = msg.get("role", "")
            if role == "tool":
                sanitized.append({
                    "role": "user",
                    "content": f"Tool result ({msg.get('name', 'tool')}): {msg.get('content', '')}",
                })
            elif role == "assistant" and msg.get("tool_calls"):
                fn = (msg["tool_calls"][0] or {}).get("function", {})
                sanitized.append({
                    "role": "assistant",
                    "content": f"[called {fn.get('name', 'tool')}({fn.get('arguments', '')})]",
                })
            else:
                # Only keep standard fields — vendor extras like `reasoning`,
                # `annotations`, `reasoning_details` cause API rejection.
                sanitized.append({"role": role, "content": msg.get("content", "")})

        sanitized.append({
            "role": "user",
            "content": (
                "Before we finish: imagine you attempted this exact task 10 times "
                "independently from scratch. How many of those 10 attempts do you "
                "think would succeed? Express your answer as a percentage (0–100). "
                "Reply with a single integer only, nothing else."
            ),
        })
        try:
            response = await self.aclient.chat.completions.create(
                model=self.model,
                messages=sanitized,
                temperature=0.0,
                max_tokens=16384,  # reasoning models may use thousands of tokens for <think>
            )
            raw = (response.choices[0].message.content or "").strip()
            finish_reason = getattr(response.choices[0], "finish_reason", "?")
            import logging as _logging
            _log = _logging.getLogger(__name__)
            _log.info(
                "aelicit_confidence finish=%s raw_tail=%r",
                finish_reason, raw[-80:],
            )
            # 1. Strip <think>...</think> blocks (Qwen3/DeepSeek) and check the remainder.
            stripped = _re.sub(r'<think>.*?</think>', '', raw, flags=_re.DOTALL).strip()
            matches = _re.findall(r'\b(\d{1,3})\b', stripped)
            if matches:
                value = int(matches[-1])
                return round(max(0.0, min(1.0, value / 100.0)), 4)
            # 2. Thinking block may be truncated (finish_reason=length): search the tail
            #    of the raw text where the answer would appear if thinking ran long.
            tail = raw[-200:]
            matches = _re.findall(r'\b(\d{1,3})\b', tail)
            if matches:
                value = int(matches[-1])
                return round(max(0.0, min(1.0, value / 100.0)), 4)
            # 3. Full scan of raw (may pick a number from inside the think block,
            #    but better than returning None).
            matches = _re.findall(r'\b(\d{1,3})\b', raw)
            if matches:
                value = int(matches[-1])
                return round(max(0.0, min(1.0, value / 100.0)), 4)
            # No integer anywhere — log and return None
            _log.warning(
                "aelicit_confidence: no integer in response for %s. "
                "finish_reason=%s raw=%r",
                self.model, finish_reason, raw[:200],
            )
        except Exception as e:
            print(f"Confidence elicitation failed for {self.model}: {e}")
        return None

    async def aelicit_run_summary(
        self,
        messages: list[dict[str, Any]],
        trace_text: str = "",
    ) -> Optional[Any]:
        """Ask the model to produce a structured RunSummary after completing a task.

        Returns a RunSummary instance, or None on failure.
        """
        from confidence_tom.task_models import RunSummary

        # Keep system + last 8 messages to avoid blowing context
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]
        trimmed = system_msgs + non_system[-8:]

        sanitized: list[dict[str, Any]] = []
        for msg in trimmed:
            msg = _normalize_chat_message(msg)
            role = msg.get("role", "")
            if role == "tool":
                sanitized.append({
                    "role": "user",
                    "content": f"Tool result ({msg.get('name', 'tool')}): {msg.get('content', '')}",
                })
            elif role == "assistant" and msg.get("tool_calls"):
                fn = (msg["tool_calls"][0] or {}).get("function", {})
                sanitized.append({
                    "role": "assistant",
                    "content": f"[called {fn.get('name', 'tool')}({fn.get('arguments', '')})]",
                })
            else:
                # Only keep standard fields — vendor extras like `reasoning`,
                # `annotations`, `reasoning_details` cause API rejection.
                sanitized.append({"role": role, "content": msg.get("content", "")})

        trace_section = f"\n\nYour execution trace:\n{trace_text}\n" if trace_text else ""
        sanitized.append({
            "role": "user",
            "content": (
                f"You just completed a task.{trace_section}\n"
                "Now produce a structured reflection covering:\n"
                "- plan: Your initial strategy before starting.\n"
                "- trajectory: For each action you took, provide thought, action, "
                "observation, and step_confidence (0-100: confidence that this step "
                "was correct at the time you took it).\n"
                "- summary: Final synthesis of all observations.\n"
                "- final_answer: Your final declared answer or task result.\n"
                "- final_confidence: If you attempted this exact task 10 times "
                "independently from scratch, what percentage would succeed? (0-100)"
            ),
        })
        result = await self.agenerate_parsed(sanitized, RunSummary)
        if result is not None:
            return result

        # Fallback for models that don't support structured output (e.g. Qwen via OpenRouter):
        # ask a single plain-text question and extract the confidence integer.
        # Do NOT include trace_section here — it can be 10k+ tokens and cause context overflow.
        confidence_prompt = sanitized[:-1] + [{
            "role": "user",
            "content": (
                "You just completed a task. "
                "If you attempted this exact task 10 times independently from scratch, "
                "how many out of 10 would succeed? "
                "Reply with a single integer between 0 and 10, nothing else."
            ),
        }]
        raw = await self.agenerate_text(confidence_prompt, max_tokens=16, temperature=0.0)
        print(f"[confidence_fallback] model={self.model} raw={raw!r}", flush=True)
        match = _re.search(r"\b(\d+)\b", raw.strip())
        if match:
            n = max(0, min(10, int(match.group(1))))
            print(f"[confidence_fallback] parsed n={n} -> final_confidence={n * 10}", flush=True)
            return RunSummary(
                plan="",
                trajectory=[],
                summary="",
                final_answer="",
                final_confidence=n * 10,
            )
        print(f"[confidence_fallback] no integer found in {raw!r}, returning None", flush=True)
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

    async def agenerate_text(self, messages: list[dict[str, Any]], max_tokens: int = 256, temperature: Optional[float] = None) -> str:
        """Async plain-text generation, no structured output required."""
        try:
            response = await self.aclient.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            return content if content else ""
        except Exception as e:
            print(f"Error in agenerate_text for {self.model}: {e}")
            return ""

    async def agenerate_text_with_trace(
        self,
        messages: list[dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> tuple[str, ApiTrace]:
        """Async plain-text generation that also returns raw API metadata."""
        try:
            response = await self.aclient.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            )
            content = response.choices[0].message.content or ""
            return content, _extract_trace(response)
        except Exception as e:
            print(f"Error in agenerate_text_with_trace for {self.model}: {e}")
            return "", ApiTrace()
