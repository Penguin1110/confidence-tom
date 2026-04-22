from __future__ import annotations

import asyncio
import inspect
import json
import logging
import random
import re
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional, cast

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from confidence_tom.data.dataset_models import StaticTask
from confidence_tom.data.scale_dataset import load_livebench_reasoning, load_olympiadbench
from confidence_tom.eval.static_evaluators import build_static_evaluator
from confidence_tom.infra.client import LLMClient
from confidence_tom.intervention import (
    ExtractedFinalAnswerOutput,
    ModelPricing,
    PrefixOracleGainStepResult,
    PrefixOracleGainTaskResult,
    PrefixSegment,
    SegmentedTraceOutput,
    parse_with_llm_fallback,
    trace_to_cost,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

_FULL_TRACE_SYSTEM_PROMPT = """You are a careful math reasoning assistant.
Solve the problem naturally and completely.
Organize the reasoning into meaningful chunks if natural.
If numbered chunks feel natural, use 1., 2., 3., ...
Keep each chunk semantically self-contained when possible.
Do not use JSON.
Do not force a fixed number of chunks.
End with a final line exactly in the form:
Final Answer: <answer>
"""

_SMALL_CONTINUE_SYSTEM_PROMPT = """You are the same small worker continuing an
existing reasoning prefix.
Continue naturally from the given prefix and finish the problem.
Do not restart from scratch unless the prefix is unusable.
Do not restate the full problem or rebuild the derivation from the beginning if
the prefix already contains useful structure.
Do not use JSON.
End with a final line exactly in the form:
Final Answer: <answer>
"""

_LARGE_TAKEOVER_SYSTEM_PROMPT = """You are a stronger reasoning assistant taking
over from a partial reasoning prefix.
Continue from the given prefix and finish the problem.
You may correct earlier mistakes if needed, but preserve useful progress when possible.
Prefer minimal edits to the existing prefix unless a correction is necessary.
Do not use JSON.
End with a final line exactly in the form:
Final Answer: <answer>
"""


def _client_kwargs_from_cfg(worker_cfg: DictConfig) -> dict[str, Any]:
    raw_kwargs = {
        "model": str(worker_cfg.model),
        "temperature": float(worker_cfg.get("temperature", 0.0)),
        "max_tokens": int(worker_cfg.get("max_tokens", 2048)),
        "reasoning_effort": worker_cfg.get("reasoning_effort"),
        "backend": str(worker_cfg.get("backend", "openrouter")),
        "local_model_name": worker_cfg.get("local_model_name"),
        "top_p": worker_cfg.get("top_p"),
        "top_k": worker_cfg.get("top_k"),
        "seed": worker_cfg.get("seed"),
        "num_ctx": worker_cfg.get("num_ctx"),
        "num_predict": worker_cfg.get("num_predict"),
        "enable_thinking": worker_cfg.get("enable_thinking"),
    }
    # Keep compatibility with older LLMClient signatures in heterogeneous envs
    # (e.g. Colab image with stale package cache).
    valid = set(inspect.signature(LLMClient.__init__).parameters.keys())
    valid.discard("self")
    return {k: v for k, v in raw_kwargs.items() if k in valid}


def _sanitize_label(text: str) -> str:
    return text.replace("/", "_").replace(":", "_").replace("-", "_").replace(".", "_")


def _pricing_from_cfg(cfg: DictConfig, model_name: str) -> Optional[ModelPricing]:
    item = cfg.pricing.get(model_name)
    if not item:
        return None
    pricing = ModelPricing(
        input_per_1k=float(item.get("input_per_1k", 0.0)),
        output_per_1k=float(item.get("output_per_1k", 0.0)),
        reasoning_per_1k=float(item.get("reasoning_per_1k", 0.0)),
    )
    if (
        pricing.input_per_1k == 0.0
        and pricing.output_per_1k == 0.0
        and pricing.reasoning_per_1k == 0.0
    ):
        return None
    return pricing


class ResultStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = self._load()
        self.index = {row["task_id"]: i for i, row in enumerate(self.rows)}

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            return cast(list[dict[str, Any]], json.loads(self.path.read_text()))
        except Exception:
            return []

    def has(self, task_id: str) -> bool:
        return task_id in self.index

    def save(self, result: PrefixOracleGainTaskResult) -> None:
        row = result.model_dump()
        existing = self.index.get(result.task_id)
        if existing is None:
            self.index[result.task_id] = len(self.rows)
            self.rows.append(row)
        else:
            self.rows[existing] = row
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.rows, ensure_ascii=False, indent=2))
        tmp.replace(self.path)


class PartialTaskStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, task_id: str) -> Path:
        return self.root / f"{task_id}.json"

    def load(self, task_id: str) -> dict[str, Any] | None:
        path = self.path_for(task_id)
        if not path.exists():
            return None
        try:
            return cast(dict[str, Any], json.loads(path.read_text()))
        except Exception:
            return None

    def save(self, task_id: str, payload: dict[str, Any]) -> None:
        tmp = self.root / f"{task_id}.tmp"
        final = self.path_for(task_id)
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        tmp.replace(final)

    def clear(self, task_id: str) -> None:
        path = self.path_for(task_id)
        if path.exists():
            path.unlink()


def _extract_final_answer(text: str) -> str:
    if not text:
        return ""

    def _extract_balanced_group(src: str, start: int) -> str:
        if start >= len(src) or src[start] != "{":
            return ""
        depth = 0
        buf: list[str] = []
        for ch in src[start:]:
            if ch == "{":
                depth += 1
                if depth > 1:
                    buf.append(ch)
                continue
            if ch == "}":
                depth -= 1
                if depth == 0:
                    return "".join(buf).strip()
                if depth < 0:
                    return ""
                buf.append(ch)
                continue
            if depth >= 1:
                buf.append(ch)
        return ""

    matches = re.findall(r"(?im)^(?:the\s+)?final answer:\s*(.+?)\s*$", text)
    if matches:
        candidate = cast(str, matches[-1]).strip()
        if candidate and candidate not in {"$", "$$", r"\[", r"\]"}:
            return candidate
    boxed_payloads: list[str] = []
    search_pos = 0
    marker = r"\boxed"
    while True:
        idx = text.find(marker, search_pos)
        if idx == -1:
            break
        brace_idx = idx + len(marker)
        while brace_idx < len(text) and text[brace_idx].isspace():
            brace_idx += 1
        payload = _extract_balanced_group(text, brace_idx)
        if payload:
            boxed_payloads.append(payload)
        search_pos = idx + len(marker)
    if boxed_payloads:
        return boxed_payloads[-1]
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        if re.fullmatch(r"(?i)(#+\s*)?(?:the\s+)?final answer:?", line):
            for follow in lines[idx + 1 :]:
                if follow not in {"$", "$$", r"\[", r"\]"}:
                    return follow.strip()
    tail = text.strip().splitlines()
    if not tail:
        return ""
    last = tail[-1].strip()
    if re.search(r"(sqrt|\\sqrt|\\boxed|boxed|[0-9])", last):
        return last
    return ""


def _looks_like_bad_answer(answer: str) -> bool:
    text = (answer or "").strip()
    if not text:
        return True
    if text in {"$", "$$", r"\[", r"\]"}:
        return True
    if re.fullmatch(r"(?i)(#+\s*)?final answer:?", text):
        return True
    return False


async def _extract_answer_with_fallback(raw_text: str, extract_client: Optional[LLMClient]) -> str:
    parser_candidate = ""
    if extract_client is not None:
        parsed, _ = await parse_with_llm_fallback(
            raw_text, ExtractedFinalAnswerOutput, extract_client
        )
        if parsed is not None and not _looks_like_bad_answer(parsed.final_answer):
            parser_candidate = parsed.final_answer.strip()
    if parser_candidate:
        return parser_candidate
    candidate = _extract_final_answer(raw_text)
    return "" if _looks_like_bad_answer(candidate) else candidate


def _split_numbered_chunks(text: str) -> list[str]:
    pattern = re.compile(r"(?m)^\s*(\d+)\.\s+")
    matches = list(pattern.finditer(text))
    if not matches:
        return []
    chunks: list[str] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _split_paragraph_chunks(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) >= 2:
        return paras
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 3:
        grouped: list[str] = []
        current: list[str] = []
        for line in lines:
            current.append(line)
            if len(" ".join(current)) > 300:
                grouped.append("\n".join(current).strip())
                current = []
        if current:
            grouped.append("\n".join(current).strip())
        return [g for g in grouped if g]
    return [text.strip()] if text.strip() else []


def _segment_trace(text: str) -> list[PrefixSegment]:
    body = re.sub(r"(?im)^final answer:\s*.+?$", "", text).strip()
    numbered = _split_numbered_chunks(body)
    raw_segments = numbered if numbered else _split_paragraph_chunks(body)
    return [
        PrefixSegment(segment_id=f"seg_{i}", index=i, text=seg.strip())
        for i, seg in enumerate(raw_segments, start=1)
        if seg.strip()
    ]


def _clean_segments(segments: list[PrefixSegment]) -> list[PrefixSegment]:
    def _is_formula_block(text: str) -> bool:
        stripped = text.strip()
        return bool(
            stripped
            and len(stripped) <= 240
            and (
                stripped.startswith("$$")
                or stripped.startswith(r"\[")
                or stripped.startswith(r"\(")
            )
        )

    def _is_header_only(text: str) -> bool:
        stripped = text.strip()
        return bool(
            re.fullmatch(r"(?:#{1,6}\s+.+|Step\s+\d+\s*:?.+)", stripped, flags=re.IGNORECASE)
        )

    def _is_short_transition(text: str) -> bool:
        stripped = text.strip()
        if not stripped or len(stripped) > 80:
            return False
        if stripped.endswith(":"):
            return True
        return stripped.lower() in {
            "simplifying:",
            "therefore:",
            "thus:",
            "so:",
            "hence:",
            "the given polynomial is:",
            "the discriminant is:",
            "the total number of ways is:",
            "the probability is:",
            "final answer:",
        }

    cleaned: list[str] = []
    pending_final_header = False
    for seg in segments:
        text = seg.text.strip()
        if not text or text == "---":
            continue
        if re.fullmatch(r"#+\s*final answer:?\s*", text, flags=re.IGNORECASE):
            pending_final_header = True
            continue
        if pending_final_header:
            text = f"Final Answer:\n{text}"
            pending_final_header = False
        cleaned.append(text)

    merged: list[str] = []
    for text in cleaned:
        if merged and (
            _is_header_only(text) or _is_short_transition(text) or _is_formula_block(text)
        ):
            merged[-1] = f"{merged[-1]}\n{text}".strip()
            continue
        if (
            merged
            and len(text) <= 50
            and not text.lower().startswith("final answer:")
            and not re.search(r"[.?!]\s*$", text)
        ):
            merged[-1] = f"{merged[-1]}\n{text}".strip()
            continue
        merged.append(text)

    return [
        PrefixSegment(segment_id=f"seg_{i}", index=i, text=text)
        for i, text in enumerate(merged, start=1)
    ]


def _prefix_text(segments: list[PrefixSegment]) -> str:
    return "\n\n".join(seg.text for seg in segments if seg.text.strip())


async def _generate_text(
    client: LLMClient,
    messages: list[dict[str, str]],
    timeout_sec: int,
    tag: str,
    task_id: str,
    retry_attempts: int = 1,
    retry_backoff_sec: float = 2.0,
) -> tuple[str, Any]:
    last_exc: Exception | None = None
    attempts = max(1, retry_attempts)
    for attempt in range(1, attempts + 1):
        try:
            raw, trace = await asyncio.wait_for(
                client.agenerate_text_with_trace(
                    messages,
                    max_tokens=client.max_tokens,
                    temperature=client.temperature,
                ),
                timeout=timeout_sec,
            )
            logger.info(
                "%s.raw task=%s raw_len=%d raw_head=%r",
                tag,
                task_id,
                len(raw or ""),
                (raw or "")[:200],
            )
            return raw, trace
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            backoff = retry_backoff_sec * (2 ** (attempt - 1))
            jitter = backoff * 0.1 * random.random()
            wait_sec = backoff + jitter
            logger.warning(
                "%s.retry task=%s attempt=%d/%d wait=%.2fs error=%s",
                tag,
                task_id,
                attempt,
                attempts,
                wait_sec,
                exc.__class__.__name__,
            )
            await asyncio.sleep(wait_sec)
    assert last_exc is not None
    raise last_exc


async def _run_full_trace(
    task: StaticTask,
    small_client: LLMClient,
    extract_client: Optional[LLMClient],
    timeout_sec: int,
    retry_attempts: int,
    retry_backoff_sec: float,
) -> tuple[str, str, Any]:
    messages = [
        {"role": "system", "content": _FULL_TRACE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem:\n{task.question}"},
    ]
    raw, trace = await _generate_text(
        small_client,
        messages,
        timeout_sec,
        "full_trace",
        task.id,
        retry_attempts=retry_attempts,
        retry_backoff_sec=retry_backoff_sec,
    )
    return raw, await _extract_answer_with_fallback(raw, extract_client), trace


async def _segment_full_trace(
    raw_text: str,
    extract_client: Optional[LLMClient],
    task_id: str,
) -> tuple[list[PrefixSegment], str, bool]:
    parsed: Optional[SegmentedTraceOutput] = None
    if extract_client is not None:
        parsed, _ = await parse_with_llm_fallback(raw_text, SegmentedTraceOutput, extract_client)
        logger.info("segment_trace.extract_fallback task=%s parsed=%s", task_id, parsed is not None)
    if parsed is not None:
        segments = _clean_segments(parsed.segments)
        final_answer = parsed.final_answer.strip() or _extract_final_answer(raw_text)
        return segments, final_answer, parsed.parse_incomplete
    segments = _clean_segments(_segment_trace(raw_text))
    return segments, _extract_final_answer(raw_text), False


async def _run_continue(
    *,
    task: StaticTask,
    client: LLMClient,
    extract_client: Optional[LLMClient],
    system_prompt: str,
    prefix_text: str,
    timeout_sec: int,
    tag: str,
    retry_attempts: int,
    retry_backoff_sec: float,
) -> tuple[str, str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Problem:\n{task.question}\n\n"
                f"Reasoning prefix:\n{prefix_text}\n\n"
                "Continue from this prefix and finish the task."
            ),
        },
    ]
    raw, trace = await _generate_text(
        client,
        messages,
        timeout_sec,
        tag,
        task.id,
        retry_attempts=retry_attempts,
        retry_backoff_sec=retry_backoff_sec,
    )
    return raw, await _extract_answer_with_fallback(raw, extract_client), trace


async def _map_task(task: StaticTask, cfg: DictConfig) -> PrefixOracleGainTaskResult:
    partial_store = PartialTaskStore(Path(to_absolute_path(str(cfg.output_dir))) / "partials")
    small_client = LLMClient(**_client_kwargs_from_cfg(cfg.small_worker))
    large_client = LLMClient(**_client_kwargs_from_cfg(cfg.large_worker))
    extract_cfg = cfg.get("extractor", {})
    extract_client = None
    if bool(extract_cfg.get("enabled", False)):
        extract_client = LLMClient(**_client_kwargs_from_cfg(extract_cfg))
    evaluator = build_static_evaluator(task)
    trace_id = f"{task.id}_{uuid.uuid4().hex[:8]}"
    execution_cfg = cfg.get("execution", {})
    retry_attempts = int(execution_cfg.get("retry_attempts", 1))
    retry_backoff_sec = float(execution_cfg.get("retry_backoff_sec", 2.0))
    resume_from_partials = bool(execution_cfg.get("resume_from_partials", True))
    retain_partials = bool(execution_cfg.get("retain_partials", True))

    resume_payload = partial_store.load(task.id) if resume_from_partials else None
    full_text = ""
    full_answer = ""
    full_trace_api = None
    parse_incomplete = False
    segments: list[PrefixSegment] = []
    oracle_steps: list[PrefixOracleGainStepResult] = []

    if resume_payload and resume_payload.get("status") in {"full_trace_done", "prefix_step_done"}:
        logger.info("partial.resume task=%s status=%s", task.id, resume_payload.get("status"))
        trace_id = str(resume_payload.get("trace_id") or trace_id)
        full_text = str(resume_payload.get("full_trace_text") or "")
        full_answer = str(resume_payload.get("full_trace_answer") or "")
        parse_incomplete = bool(resume_payload.get("parse_incomplete", False))
        segments = [PrefixSegment.model_validate(s) for s in resume_payload.get("segments", [])]
        oracle_steps = [
            PrefixOracleGainStepResult.model_validate(s)
            for s in resume_payload.get("prefix_oracle_steps", [])
        ]
        full_trace_api = resume_payload.get("full_trace_api_trace")
    else:
        full_text, full_answer, full_trace_api = await _run_full_trace(
            task,
            small_client,
            extract_client,
            int(cfg.timeouts.full_trace_sec),
            retry_attempts,
            retry_backoff_sec,
        )
        segments, parsed_final_answer, parse_incomplete = await _segment_full_trace(
            full_text, extract_client, task.id
        )
        if parsed_final_answer:
            full_answer = parsed_final_answer
        full_eval = evaluator(full_answer, task) if full_answer else evaluator("", task)
        logger.info(
            "full_trace.summary task=%s segments=%d final_answer=%r correct=%s parse_incomplete=%s",
            task.id,
            len(segments),
            full_answer,
            full_eval.is_correct,
            parse_incomplete,
        )
        partial_store.save(
            task.id,
            {
                "task_id": task.id,
                "status": "full_trace_done",
                "small_model": str(cfg.small_worker.model),
                "large_model": str(cfg.large_worker.model),
                "trace_id": trace_id,
                "full_trace_text": full_text,
                "full_trace_answer": full_answer,
                "full_trace_correct": full_eval.is_correct,
                "full_trace_api_trace": full_trace_api.model_dump()
                if full_trace_api is not None
                else None,
                "parse_incomplete": parse_incomplete,
                "segments": [s.model_dump() for s in segments],
                "prefix_oracle_steps": [],
            },
        )

    small_pricing = _pricing_from_cfg(cfg, str(cfg.small_worker.model))
    large_pricing = _pricing_from_cfg(cfg, str(cfg.large_worker.model))
    full_eval = evaluator(full_answer, task) if full_answer else evaluator("", task)
    start_step_index = len(oracle_steps) + 1
    if start_step_index > 1:
        logger.info(
            "partial.resume.steps task=%s completed_steps=%d remaining_steps=%d",
            task.id,
            len(oracle_steps),
            max(0, len(segments) - len(oracle_steps)),
        )

    for step_index in range(start_step_index, len(segments) + 1):
        prefix_segments = segments[:step_index]
        prefix_text = _prefix_text(prefix_segments)
        prefix_id = f"{trace_id}_p{step_index}"
        parent_prefix_id = f"{trace_id}_p{step_index - 1}" if step_index > 1 else ""
        logger.info("prefix.step task=%s step=%d/%d", task.id, step_index, len(segments))

        try:
            small_text, small_answer, small_api = await _run_continue(
                task=task,
                client=small_client,
                extract_client=extract_client,
                system_prompt=_SMALL_CONTINUE_SYSTEM_PROMPT,
                prefix_text=prefix_text,
                timeout_sec=int(cfg.timeouts.small_worker_sec),
                tag="small_continue_prefix",
                retry_attempts=retry_attempts,
                retry_backoff_sec=retry_backoff_sec,
            )
        except Exception:
            logger.error(
                "small_continue_prefix.error task=%s step=%d\n%s",
                task.id,
                step_index,
                traceback.format_exc(),
            )
            small_text, small_answer, small_api = "", "", None

        try:
            large_text, large_answer, large_api = await _run_continue(
                task=task,
                client=large_client,
                extract_client=extract_client,
                system_prompt=_LARGE_TAKEOVER_SYSTEM_PROMPT,
                prefix_text=prefix_text,
                timeout_sec=int(cfg.timeouts.large_worker_sec),
                tag="large_takeover_prefix",
                retry_attempts=retry_attempts,
                retry_backoff_sec=retry_backoff_sec,
            )
        except Exception:
            logger.error(
                "large_takeover_prefix.error task=%s step=%d\n%s",
                task.id,
                step_index,
                traceback.format_exc(),
            )
            large_text, large_answer, large_api = "", "", None

        small_eval = evaluator(small_answer, task) if small_answer else evaluator("", task)
        large_eval = evaluator(large_answer, task) if large_answer else evaluator("", task)

        oracle_steps.append(
            PrefixOracleGainStepResult(
                prefix_id=prefix_id,
                parent_prefix_id=parent_prefix_id,
                step_index=step_index,
                prefix_segments=prefix_segments,
                prefix_text=prefix_text,
                small_continue_answer=small_answer,
                small_continue_correct=small_eval.is_correct,
                large_takeover_answer=large_answer,
                large_takeover_correct=large_eval.is_correct,
                delta_correctness=float((large_eval.score or 0.0) - (small_eval.score or 0.0)),
                small_continue_cost=trace_to_cost(small_api, small_pricing),
                large_takeover_cost=trace_to_cost(large_api, large_pricing),
                small_continue_text=small_text,
                large_takeover_text=large_text,
                small_continue_api_trace=small_api,
                large_takeover_api_trace=large_api,
            )
        )
        logger.info(
            "prefix.step.summary task=%s step=%d small_correct=%s large_correct=%s delta=%.3f",
            task.id,
            step_index,
            small_eval.is_correct,
            large_eval.is_correct,
            float((large_eval.score or 0.0) - (small_eval.score or 0.0)),
        )
        partial_store.save(
            task.id,
            {
                "task_id": task.id,
                "status": "prefix_step_done",
                "completed_step_index": step_index,
                "small_model": str(cfg.small_worker.model),
                "large_model": str(cfg.large_worker.model),
                "trace_id": trace_id,
                "full_trace_text": full_text,
                "full_trace_answer": full_answer,
                "full_trace_correct": full_eval.is_correct,
                "full_trace_api_trace": (
                    full_trace_api.model_dump()
                    if full_trace_api is not None and hasattr(full_trace_api, "model_dump")
                    else full_trace_api
                ),
                "parse_incomplete": parse_incomplete,
                "segments": [s.model_dump() for s in segments],
                "prefix_oracle_steps": [s.model_dump() for s in oracle_steps],
            },
        )

    result = PrefixOracleGainTaskResult(
        task_id=task.id,
        benchmark=task.source,
        small_model=str(cfg.small_worker.model),
        large_model=str(cfg.large_worker.model),
        trace_id=trace_id,
        full_trace_text=full_text,
        full_trace_answer=full_answer,
        full_trace_correct=full_eval.is_correct,
        segments=segments,
        prefix_oracle_steps=oracle_steps,
        metadata={
            "reference_answer": task.reference_answer,
            "evaluator_name": task.evaluator_name,
            "category": task.category,
            "external_difficulty": task.external_difficulty,
            "takeover_mode": "prefix_conditioned_resolve",
            "full_trace_api_trace": (
                full_trace_api.model_dump()
                if full_trace_api is not None and hasattr(full_trace_api, "model_dump")
                else full_trace_api
            ),
        },
    )
    if not retain_partials:
        partial_store.clear(task.id)
    return result


@hydra.main(
    version_base=None,
    config_path="../../../../configs",
    config_name="prefix_oracle_gain_mapping",
)
def main(cfg: DictConfig) -> None:
    output_dir = Path(to_absolute_path(str(cfg.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_name = str(cfg.dataset.benchmark)
    if benchmark_name == "olympiadbench":
        questions = load_olympiadbench(num_samples=int(cfg.dataset.olympiadbench))
    elif benchmark_name == "livebench_reasoning":
        livebench_count = int(
            cfg.dataset.get("livebench_reasoning", cfg.dataset.get("livebench", 0))
        )
        questions = load_livebench_reasoning(num_samples=livebench_count)
    else:
        raise ValueError(f"Unsupported prefix oracle-gain benchmark: {benchmark_name}")

    if cfg.dataset.limit:
        questions = questions[: int(cfg.dataset.limit)]

    out_path = output_dir / (
        f"{_sanitize_label(str(cfg.small_worker.label))}_to_"
        f"{_sanitize_label(str(cfg.large_worker.label))}.json"
    )
    store = ResultStore(out_path)
    logger.info("Loaded %d tasks for prefix oracle gain mapping", len(questions))

    async def _run_all() -> None:
        execution_cfg = cfg.get("execution", {})
        concurrency = max(1, int(execution_cfg.get("task_concurrency", 1)))
        sem = asyncio.Semaphore(concurrency)
        save_lock = asyncio.Lock()

        async def _run_one(i: int, task: StaticTask) -> None:
            if store.has(task.id):
                return
            async with sem:
                logger.info("[%d/%d] %s", i, len(questions), task.id)
                try:
                    result = await asyncio.wait_for(
                        _map_task(task, cfg),
                        timeout=float(cfg.timeouts.task_sec),
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "task.timeout task=%s timeout_sec=%s", task.id, cfg.timeouts.task_sec
                    )
                    return
                except Exception:
                    logger.error("task.error task=%s\n%s", task.id, traceback.format_exc())
                    return
                async with save_lock:
                    store.save(result)

        pending = [
            asyncio.create_task(_run_one(i, task))
            for i, task in enumerate(questions, start=1)
            if not store.has(task.id)
        ]
        if pending:
            await asyncio.gather(*pending)

    asyncio.run(_run_all())


if __name__ == "__main__":
    main()
