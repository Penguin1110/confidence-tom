"""Phase 2: Run the scale generator experiment.

For each model in the scale sequence (Gemma 4B/12B/27B), answer all questions
K=10 times and compute behavioral confidence.

Features:
  - All models run in parallel (async concurrency)
  - Thread-safe real-time checkpoint saving (asyncio.Lock per file)
  - Full resume from checkpoint (skip already-processed questions)
  - tqdm progress bar per model
  - Graceful Ctrl+C handling (saves current state)

Usage:
    uv run python experiments/run_scale_generator.py

    # Override config values:
    uv run python experiments/run_scale_generator.py generator.k_samples=5
    uv run python experiments/run_scale_generator.py concurrency.max_per_model=10
"""

import asyncio
import collections
import json
import logging
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from confidence_tom.client import LLMClient
from confidence_tom.dataset_models import StaticTask
from confidence_tom.parsing import (
    get_parse_stats,
    normalize_confidence,
    parse_mc_response,
    parse_static_response,
    reset_parse_stats,
)
from confidence_tom.scale_dataset import load_scale_experiment_dataset
from confidence_tom.static_evaluators import build_static_evaluator
from confidence_tom.task_models import ApiTrace, StaticTrace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def _extract_answer_explicit(raw_text: str) -> str | None:
    """Recover answer only from explicit final-answer style statements.

    Important: We intentionally DO NOT infer from option text mentions,
    because models often discuss multiple options during reasoning.
    """
    patterns = [
        r"\bfinal answer\b\s*[:\-]?\s*\(?([A-Ja-j])\)?",
        r"\bmy answer\b\s*[:\-]?\s*\(?([A-Ja-j])\)?",
        r"\banswer is\b\s*[:\-]?\s*\(?([A-Ja-j])\)?",
        r"\bi (?:choose|select)\b\s*\(?([A-Ja-j])\)?",
        r"\btherefore\b.*?\b(?:answer|option|choice)\b\s*(?:is|:)?\s*\(?([A-Ja-j])\)?",
    ]
    matches: list[str] = []
    for pat in patterns:
        for m in re.finditer(pat, raw_text, re.IGNORECASE | re.DOTALL):
            matches.append(m.group(1).upper())

    if not matches:
        return None

    unique = sorted(set(matches))
    if len(unique) == 1:
        return unique[0]
    # Conflicting explicit answers -> treat as unresolved.
    return None


def _extract_confidence_loose(raw_text: str) -> float:
    """Recover confidence from unstructured text; fallback to neutral 0.5."""
    patterns = [
        r'["\']?[Cc]onfidence["\']?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%?',
        r'["\']?[Cc]onfident["\']?\s*[:=]?\s*(\d+(?:\.\d+)?)',
        r"(\d+(?:\.\d+)?)\s*/\s*100\b",
        r"(\d+(?:\.\d+)?)\s*%\s*confiden",
        r"(\d+(?:\.\d+)?)\s*%?\s*confident",
    ]
    for pat in patterns:
        m = re.search(pat, raw_text, re.IGNORECASE)
        if m:
            try:
                return normalize_confidence(float(m.group(1)))
            except ValueError:
                pass

    lower = raw_text.lower()
    if any(p in lower for p in ["extremely confident", "very certain", "almost certain"]):
        return 0.95
    if any(p in lower for p in ["very confident", "high confidence", "highly confident"]):
        return 0.85
    if any(p in lower for p in ["confident", "fairly confident", "reasonably confident"]):
        return 0.70
    if any(p in lower for p in ["somewhat confident", "moderately confident"]):
        return 0.55
    if any(p in lower for p in ["uncertain", "not sure", "low confidence", "not very confident"]):
        return 0.35
    if any(p in lower for p in ["guess", "pure guess", "wild guess"]):
        return 0.15
    return 0.50


def _has_explicit_confidence(raw_text: str) -> bool:
    """Whether the raw response explicitly contains confidence signal."""
    numeric_patterns = [
        r'["\']?[Cc]onfidence["\']?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%?',
        r'["\']?[Cc]onfident["\']?\s*[:=]?\s*(\d+(?:\.\d+)?)',
        r"(\d+(?:\.\d+)?)\s*/\s*100\b",
        r"(\d+(?:\.\d+)?)\s*%\s*confiden",
        r"(\d+(?:\.\d+)?)\s*%?\s*confident",
    ]
    for pat in numeric_patterns:
        if re.search(pat, raw_text, re.IGNORECASE):
            return True

    lower = raw_text.lower()
    qualitative = [
        "extremely confident",
        "very certain",
        "almost certain",
        "very confident",
        "high confidence",
        "highly confident",
        "fairly confident",
        "reasonably confident",
        "somewhat confident",
        "moderately confident",
        "not very confident",
        "low confidence",
        "uncertain",
        "not sure",
        "guess",
    ]
    return any(p in lower for p in qualitative)


def _extract_strategy(raw_text: str, reasoning_text: str) -> str:
    """Best-effort extraction of a short high-level plan from static reasoning."""
    strategy_match = re.search(
        r'["\']?[Ss]trategy["\']?\s*[:=]\s*["\']?(.*?)(?:["\']?\s*,\s*["\']?[Rr]easoning["\']?|$)',
        raw_text,
        re.IGNORECASE | re.DOTALL,
    )
    if strategy_match:
        return strategy_match.group(1).strip().strip('"').strip("'")

    for line in reasoning_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        return stripped[:240]
    return ""


def _build_static_prompt(task: StaticTask) -> tuple[str, str]:
    """Create the worker prompt based on answer format."""
    if task.answer_format == "multiple_choice":
        choices_str = "\n".join(task.choices)
        system_prompt = (
            "You are answering a multiple-choice question.\n"
            "Read the question carefully, think step by step, "
            "and choose the best answer.\n\n"
            "You MUST respond in this EXACT JSON format:\n"
            '{"strategy": "...", "reasoning": "...", "answer": "B", "confidence": 45}\n\n'
            "Output policy (STRICT):\n"
            "- Output ONLY a single valid JSON object.\n"
            "- answer MUST be exactly one choice letter.\n"
            "- confidence must be an integer 0-100.\n"
            "- strategy: 1-2 sentences describing your high-level plan.\n"
            "- reasoning: brief step-by-step reasoning.\n"
        )
        user_prompt = f"Question: {task.question}\n\n{choices_str}"
        return system_prompt, user_prompt

    system_prompt = (
        "You are answering a single-turn reasoning question.\n"
        "Think step by step, then provide your final answer succinctly.\n\n"
        "You MUST respond in this EXACT JSON format:\n"
        '{"strategy": "...", "reasoning": "...", "answer": "...", "confidence": 45}\n\n'
        "Output policy (STRICT):\n"
        "- Output ONLY a single valid JSON object.\n"
        "- answer should be your final answer string only, concise and direct.\n"
        "- confidence must be an integer 0-100.\n"
        "- strategy: 1-2 sentences describing your high-level plan.\n"
        "- reasoning: brief step-by-step reasoning.\n"
    )
    return system_prompt, f"Question: {task.question}"


# ---- Pydantic models for structured output ----


# NOTE: Structured output removed — all models now use text + parsing
# to ensure fair comparison (Gemma 12B couldn't use structured output).


# ---- Thread-safe checkpoint manager ----


class CheckpointManager:
    """Thread-safe checkpoint for real-time saving and resume.

    Each model gets its own JSON file and asyncio.Lock to prevent
    concurrent writes from corrupting the file.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}
        self._results: dict[str, list[dict[str, Any]]] = {}
        self._processed: dict[str, set[str]] = {}
        self._counters: dict[str, dict[str, int]] = {}

    def _get_lock(self, model_label: str) -> asyncio.Lock:
        if model_label not in self._locks:
            self._locks[model_label] = asyncio.Lock()
        return self._locks[model_label]

    def _file_path(self, model_label: str) -> Path:
        return self.output_dir / f"{model_label}.json"

    def load_checkpoint(self, model_label: str) -> int:
        """Load existing checkpoint for a model. Returns count of loaded items."""
        self._results[model_label] = []
        self._processed[model_label] = set()
        self._counters[model_label] = {"success": 0, "failed": 0, "skipped": 0}

        fp = self._file_path(model_label)
        if fp.exists():
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                self._results[model_label] = existing
                self._processed[model_label] = {r["question_id"] for r in existing}
                self._counters[model_label]["skipped"] = len(existing)
                return len(existing)
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"  [{model_label}] Corrupt checkpoint, starting fresh")
        return 0

    def is_processed(self, model_label: str, question_id: str) -> bool:
        return question_id in self._processed.get(model_label, set())

    async def save_result(self, model_label: str, result: dict[str, Any]) -> None:
        """Thread-safe append + save to disk."""
        lock = self._get_lock(model_label)
        async with lock:
            self._results[model_label].append(result)
            self._processed[model_label].add(result["question_id"])
            self._counters[model_label]["success"] += 1

            # Atomic-ish write: write to tmp then rename
            fp = self._file_path(model_label)
            tmp_fp = fp.with_suffix(".json.tmp")
            with open(tmp_fp, "w", encoding="utf-8") as f:
                json.dump(
                    self._results[model_label],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            tmp_fp.rename(fp)

    async def record_failure(self, model_label: str) -> None:
        lock = self._get_lock(model_label)
        async with lock:
            self._counters[model_label]["failed"] += 1

    def get_counts(self, model_label: str) -> dict[str, int]:
        return self._counters.get(model_label, {})

    def get_total_done(self, model_label: str) -> int:
        c = self._counters.get(model_label, {})
        return c.get("success", 0) + c.get("skipped", 0)


# ---- Core experiment logic ----


async def solve_question_k_times(
    client: LLMClient,
    model_id: str,
    question: StaticTask,
    k: int = 10,
    extract_client: LLMClient | None = None,
    store_raw_samples: bool = False,
    raw_truncate_chars: int = 800,
) -> dict[str, Any] | None:
    """Run a model on a single static task K times.

    Returns a result dict with behavioral confidence, reported confidence,
    majority answer, etc. — all in the unified format.
    """
    system_prompt, user_prompt = _build_static_prompt(question)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    samples: list[dict[str, Any]] = []
    sample_traces: list[dict[str, Any]] = []
    unresolved_samples = 0
    n_extract_success = 0
    n_extract_attempts = 0

    def _extractor_messages(raw_text: str, valid_labels: str) -> list[dict[str, str]]:
        answer_rule = (
            f"- answer must be one of: {valid_labels}\n"
            if question.answer_format == "multiple_choice"
            else "- answer must be a concise final answer string.\n"
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are a strict information extractor.\n"
                    "From the model output below, extract ONLY:\n"
                    "1) final answer\n"
                    "2) confidence (0-100)\n\n"
                    "Return ONLY valid JSON:\n"
                    '{"answer":"...", "confidence":50}\n\n'
                    f"Rules:\n{answer_rule}"
                    "- confidence must be numeric in [0,100]\n"
                    "- If unresolved, still output best extraction."
                ),
            },
            {
                "role": "user",
                "content": f"Raw model output:\n\n{raw_text}",
            },
        ]

    async def fetch_one(i: int) -> dict[str, Any] | None:
        """Fetch a single sample via text generation + robust parsing.

        All models use the same pipeline (text → parse) for fairness.
        Structured output was removed because Gemma 12B couldn't
        use it, causing systematic bias.
        """
        result: dict[str, Any] = {
            "sample_index": i,
            "parse_method": "request_error",
            "answer": None,
            "confidence": None,
            "reasoning": "",
            "raw_text": "",
            "extract_attempted": False,
            "extract_model": str(extract_client.model) if extract_client is not None else "",
            "extract_raw": "",
            "extract_answer": None,
            "extract_confidence": None,
            "strategy": "",
            "original_has_explicit_answer": False,
            "original_has_confidence": False,
            "missing_answer_in_original": True,
            "missing_confidence_in_original": True,
            "note": "",
            "api_trace": ApiTrace().model_dump(),
            "extract_api_trace": ApiTrace().model_dump(),
        }
        try:
            raw_text, trace = await client.agenerate_text_with_trace(messages)
            result["api_trace"] = trace.model_dump()
            if not raw_text:
                result["parse_method"] = "empty"
                result["note"] = "empty_raw_response"
                return result

            result["raw_text"] = raw_text[:raw_truncate_chars] if store_raw_samples else ""
            explicit_answer = (
                _extract_answer_explicit(raw_text)
                if question.answer_format == "multiple_choice"
                else None
            )
            has_conf = _has_explicit_confidence(raw_text)
            result["original_has_explicit_answer"] = explicit_answer is not None
            result["original_has_confidence"] = has_conf
            result["missing_answer_in_original"] = explicit_answer is None
            result["missing_confidence_in_original"] = not has_conf

            parsed_resp = (
                parse_mc_response(raw_text, model_name=model_id)
                if question.answer_format == "multiple_choice"
                else parse_static_response(raw_text, model_name=model_id)
            )
            if parsed_resp:
                result["original_has_explicit_answer"] = True
                result["missing_answer_in_original"] = False
                result.update(
                    {
                        "answer": parsed_resp.answer,
                        "confidence": parsed_resp.confidence,
                        "strategy": parsed_resp.strategy or _extract_strategy(raw_text, parsed_resp.reasoning),
                        "reasoning": parsed_resp.reasoning or raw_text,
                        "parse_method": "text",
                    }
                )
                return result

            if explicit_answer:
                result.update(
                    {
                        "answer": explicit_answer,
                        "confidence": _extract_confidence_loose(raw_text),
                        "strategy": _extract_strategy(raw_text, raw_text),
                        "reasoning": raw_text,
                        "parse_method": "fallback",
                    }
                )
                return result

            if extract_client is not None:
                valid_labels = ", ".join(
                    sorted({c.split(")")[0].strip() for c in question.choices if ")" in c})
                )
                extract_raw, extract_trace = await extract_client.agenerate_text_with_trace(
                    _extractor_messages(raw_text, valid_labels or "A, B, C, D")
                )
                result["extract_attempted"] = True
                result["extract_api_trace"] = extract_trace.model_dump()
                result["extract_raw"] = (
                    extract_raw[:raw_truncate_chars] if (store_raw_samples and extract_raw) else ""
                )
                extracted = None
                if extract_raw:
                    extracted = (
                        parse_mc_response(
                            extract_raw,
                            model_name=f"{model_id}::extractor",
                        )
                        if question.answer_format == "multiple_choice"
                        else parse_static_response(
                            extract_raw,
                            model_name=f"{model_id}::extractor",
                        )
                    )
                if extracted:
                    result.update(
                        {
                            "answer": extracted.answer,
                            "confidence": extracted.confidence,
                            "strategy": _extract_strategy(raw_text, extracted.reasoning or raw_text),
                            "reasoning": raw_text,
                            "parse_method": "llm_extract",
                            "extract_answer": extracted.answer,
                            "extract_confidence": extracted.confidence,
                            "note": (
                                "extracted_missing_answer"
                                if result["missing_answer_in_original"]
                                else (
                                    "extracted_missing_confidence"
                                    if result["missing_confidence_in_original"]
                                    else "extracted_after_parse_failure"
                                )
                            ),
                        }
                    )
                    return result
                result["parse_method"] = "unparsed"
                result["note"] = "extract_attempted_but_unresolved"
                return result

            result["parse_method"] = "unparsed"
            result["note"] = "no_extract_client"
            return result
        except Exception as e:
            logger.debug(f"  Sample {i + 1} failed: {e}")
            result["note"] = f"request_exception:{type(e).__name__}"
            return result

    # Run K samples concurrently
    tasks = [fetch_one(i) for i in range(k)]
    results = await asyncio.gather(*tasks)

    for r in results:
        if r.get("answer") is not None and r.get("confidence") is not None:
            samples.append(r)
            if r.get("parse_method") == "llm_extract":
                n_extract_success += 1
        else:
            unresolved_samples += 1
        if r.get("extract_attempted"):
            n_extract_attempts += 1
        sample_traces.append(
            {
                "sample_index": int(r.get("sample_index", -1)),
                "parse_method": str(r.get("parse_method", "unknown")),
                "answer": r.get("answer"),
                "confidence": (
                    round(float(r["confidence"]), 4)
                    if isinstance(r.get("confidence"), (int, float))
                    else None
                ),
                "strategy": str(r.get("strategy", "")),
                "raw_text": r.get("raw_text", "") if store_raw_samples else "",
                "reasoning": str(r.get("reasoning", "")),
                "extract_attempted": bool(r.get("extract_attempted", False)),
                "extract_model": str(r.get("extract_model", "")),
                "extract_raw": r.get("extract_raw", "") if store_raw_samples else "",
                "extract_answer": r.get("extract_answer"),
                "extract_confidence": (
                    round(float(r["extract_confidence"]), 4)
                    if isinstance(r.get("extract_confidence"), (int, float))
                    else None
                ),
                "original_has_explicit_answer": bool(r.get("original_has_explicit_answer", False)),
                "original_has_confidence": bool(r.get("original_has_confidence", False)),
                "missing_answer_in_original": bool(r.get("missing_answer_in_original", True)),
                "missing_confidence_in_original": bool(
                    r.get("missing_confidence_in_original", True)
                ),
                "note": str(r.get("note", "")),
                "api_trace": r.get("api_trace", {}),
                "extract_api_trace": r.get("extract_api_trace", {}),
            }
        )

    if not samples:
        return {
            "question_id": question.id,
            "question": question.question,
            "choices": question.choices,
            "correct_answer": question.correct_answer,
            "reference_answer": question.reference_answer,
            "category": question.category,
            "source": question.source,
            "answer_format": question.answer_format,
            "evaluator_name": question.evaluator_name,
            "task_type": question.task_type,
            "environment_context": question.environment_context,
            "metadata": question.metadata,
            "external_difficulty": question.external_difficulty,
            "model_name": model_id,
            "k_samples": 0,
            "k_target": k,
            "majority_answer": "",
            "is_correct": False,
            "c_beh": 0.0,
            "c_rep": 0.0,
            "gap": 0.0,
            "strategy": "",
            "answer": "",
            "confidence": 0,
            "static_trace": StaticTrace().model_dump(),
            "primary_reasoning": "",
            "answer_distribution": {},
            "sample_confidences": [],
            "parse_structured": 0,
            "parse_fallback": 0,
            "parse_llm_extract": 0,
            "parse_unresolved": len(results),
            "extract_attempts": n_extract_attempts,
            "extract_success": n_extract_success,
            "all_samples_failed": True,
            "evaluation": {"score": 0.0, "evaluator_name": question.evaluator_name, "metadata": {}},
            "sample_traces": sample_traces,
        }

    # Compute behavioral confidence
    answers = [str(s["answer"]).strip() for s in samples]
    counter = collections.Counter(answers)
    majority_answer, majority_count = counter.most_common(1)[0]
    c_beh = majority_count / len(samples)

    # Average reported confidence (0-1)
    c_rep = sum(s["confidence"] for s in samples) / len(samples)

    eval_result = build_static_evaluator(question)(majority_answer, question)
    is_correct = eval_result.is_correct

    # Primary reasoning from majority-answer sample
    primary_reasoning = next(
        (s["reasoning"] for s in samples if s["answer"] == majority_answer),
        "",
    )
    primary_strategy = next(
        (str(s.get("strategy", "")) for s in samples if s["answer"] == majority_answer),
        "",
    )
    primary_trace = StaticTrace(
        strategy=primary_strategy,
        reasoning=primary_reasoning,
        answer=majority_answer,
        confidence=int(round(c_rep * 100)),
    )

    n_structured = sum(1 for s in samples if s.get("parse_method") == "structured")
    n_fallback = sum(1 for s in samples if s.get("parse_method") == "fallback")
    n_llm_extract = sum(1 for s in samples if s.get("parse_method") == "llm_extract")

    return {
        "question_id": question.id,
        "question": question.question,
        "choices": question.choices,
        "correct_answer": question.correct_answer,
        "reference_answer": question.reference_answer,
        "category": question.category,
        "source": question.source,
        "answer_format": question.answer_format,
        "evaluator_name": question.evaluator_name,
        "task_type": question.task_type,
        "environment_context": question.environment_context,
        "metadata": question.metadata,
        "external_difficulty": question.external_difficulty,
        "model_name": model_id,
        "k_samples": len(samples),
        "k_target": k,
        "majority_answer": majority_answer,
        "is_correct": is_correct,
        "c_beh": round(c_beh, 4),
        "c_rep": round(c_rep, 4),
        "gap": round(c_rep - c_beh, 4),
        "strategy": primary_strategy,
        "answer": majority_answer,
        "confidence": int(round(c_rep * 100)),
        "static_trace": primary_trace.model_dump(),
        "primary_reasoning": primary_reasoning,
        "evaluation": {
            "score": eval_result.score,
            "evaluator_name": eval_result.evaluator_name,
            "metadata": eval_result.metadata,
        },
        "answer_distribution": dict(counter),
        "sample_confidences": [round(s["confidence"], 4) for s in samples],
        "parse_structured": n_structured,
        "parse_fallback": n_fallback,
        "parse_llm_extract": n_llm_extract,
        "parse_unresolved": unresolved_samples,
        "extract_attempts": n_extract_attempts,
        "extract_success": n_extract_success,
        "all_samples_failed": False,
        "sample_traces": sample_traces,
    }


# ---- Per-model worker ----


async def run_model(
    model_id: str,
    model_label: str,
    questions: list[StaticTask],
    gen_cfg: Any,
    extract_cfg: Any,
    checkpoint: CheckpointManager,
    semaphore: asyncio.Semaphore,
) -> None:
    """Run all questions for a single model (async, with concurrency limit)."""
    logger.info(f"🚀 [{model_label}] Starting ({model_id})")

    loaded = checkpoint.load_checkpoint(model_label)
    if loaded > 0:
        logger.info(f"  [{model_label}] Resumed: {loaded} already done")

    reset_parse_stats()

    client = LLMClient(
        model=model_id,
        temperature=gen_cfg.temperature,
        max_tokens=gen_cfg.max_tokens,
    )
    extract_client: LLMClient | None = None
    if bool(extract_cfg.get("enabled", False)):
        extract_client = LLMClient(
            model=str(extract_cfg.get("model", "google/gemini-3.1-flash-lite-preview")),
            temperature=float(extract_cfg.get("temperature", 0.0)),
            max_tokens=int(extract_cfg.get("max_tokens", 512)),
        )

    total = len(questions)
    start_time = time.time()

    async def process_one(q: StaticTask) -> None:
        if checkpoint.is_processed(model_label, q.id):
            return

        async with semaphore:
            result = await solve_question_k_times(
                client=client,
                model_id=model_id,
                question=q,
                k=gen_cfg.k_samples,
                extract_client=extract_client,
                store_raw_samples=bool(extract_cfg.get("store_raw_samples", True)),
                raw_truncate_chars=int(extract_cfg.get("raw_truncate_chars", 1200)),
            )

            if result:
                await checkpoint.save_result(model_label, result)
                done = checkpoint.get_total_done(model_label)
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                logger.info(
                    f"  [{model_label}] "
                    f"[{done}/{total}] "
                    f"{q.id} | "
                    f"✅={result['is_correct']} | "
                    f"c_beh={result['c_beh']:.2f} | "
                    f"c_rep={result['c_rep']:.2f} | "
                    f"xtr={result.get('parse_llm_extract', 0)}/{result.get('extract_attempts', 0)} | "
                    f"gap={result['gap']:+.2f} | "
                    f"ETA {eta:.0f}s"
                )
            else:
                await checkpoint.record_failure(model_label)
                logger.warning(f"  [{model_label}] ❌ FAILED: {q.id}")

    # Launch all questions concurrently (semaphore controls parallelism)
    tasks = [process_one(q) for q in questions]
    await asyncio.gather(*tasks)

    # Final stats
    elapsed = time.time() - start_time
    counts = checkpoint.get_counts(model_label)
    stats = get_parse_stats()
    parse_info = stats.get(model_id, {"success": 0, "failure": 0})

    logger.info(
        f"✅ [{model_label}] Complete in {elapsed:.0f}s | "
        f"success={counts.get('success', 0)} | "
        f"failed={counts.get('failed', 0)} | "
        f"skipped={counts.get('skipped', 0)} | "
        f"parse_fail={parse_info['failure']}"
    )


# ---- Main entry ----


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="scale_single",
)
def main(cfg: DictConfig) -> None:
    asyncio.run(amain(cfg))


async def amain(cfg: DictConfig) -> None:
    # Load dataset once (shared across all models)
    dataset_counts = None
    if "counts" in cfg.dataset:
        dataset_counts = {
            "mmlu_pro": int(cfg.dataset.counts.get("mmlu_pro", 0)),
            "gpqa": int(cfg.dataset.counts.get("gpqa", 0)),
            "math_l5": int(cfg.dataset.counts.get("math_l5", 0)),
            "mmlu_hard": int(cfg.dataset.counts.get("mmlu_hard", 0)),
            "hle_mc": int(cfg.dataset.counts.get("hle_mc", 0)),
            "supergpqa": int(cfg.dataset.counts.get("supergpqa", 0)),
            "simplebench": int(cfg.dataset.counts.get("simplebench", 0)),
            "truthfulqa_mc": int(cfg.dataset.counts.get("truthfulqa_mc", 0)),
            "harp_mcq": int(cfg.dataset.counts.get("harp_mcq", 0)),
            "musr": int(cfg.dataset.counts.get("musr", 0)),
            "olympiadbench": int(cfg.dataset.counts.get("olympiadbench", 0)),
            "livebench": int(cfg.dataset.counts.get("livebench", 0)),
        }

    questions = load_scale_experiment_dataset(
        num_per_source=int(cfg.dataset.num_per_source),
        counts=dataset_counts,
    )
    logger.info(f"📦 Loaded {len(questions)} questions")

    extract_cfg = cfg.get("extractor", {})

    # Checkpoint manager (thread-safe per-model saving)
    checkpoint = CheckpointManager(Path(cfg.output_dir))

    # Concurrency: semaphore per model
    max_per_model = cfg.concurrency.get("max_per_model", 8)

    selected_models = list(cfg.scale_models)
    exclude_labels = {str(x) for x in cfg.get("exclude_model_labels", [])}
    if exclude_labels:
        selected_models = [m for m in selected_models if str(m.label) not in exclude_labels]
        logger.info(f"⚙️  Excluding models: {', '.join(sorted(exclude_labels))}")

    logger.info(
        f"⚙️  Config: "
        f"{len(selected_models)} models × "
        f"{len(questions)} questions × "
        f"K={cfg.generator.k_samples} = "
        f"{len(selected_models) * len(questions) * cfg.generator.k_samples} "
        f"API calls"
    )
    logger.info(f"⚙️  Concurrency: {max_per_model} per model")

    # Graceful shutdown
    shutdown_event = asyncio.Event()

    def handle_signal(sig: int, frame: Any) -> None:
        logger.warning("\n⚠️  Ctrl+C detected! Finishing current tasks and saving checkpoints...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    sequential_models = bool(cfg.concurrency.get("sequential_models", False))

    # ---- Launch models ----
    if sequential_models:
        logger.info("⚙️  Model scheduling: sequential")
        for model_cfg in selected_models:
            sem = asyncio.Semaphore(max_per_model)
            await run_model(
                model_id=model_cfg.id,
                model_label=model_cfg.label,
                questions=questions,
                gen_cfg=cfg.generator,
                extract_cfg=extract_cfg,
                checkpoint=checkpoint,
                semaphore=sem,
            )
    else:
        logger.info("⚙️  Model scheduling: parallel")
        model_tasks = []
        for model_cfg in selected_models:
            sem = asyncio.Semaphore(max_per_model)
            task = asyncio.create_task(
                run_model(
                    model_id=model_cfg.id,
                    model_label=model_cfg.label,
                    questions=questions,
                    gen_cfg=cfg.generator,
                    extract_cfg=extract_cfg,
                    checkpoint=checkpoint,
                    semaphore=sem,
                )
            )
            model_tasks.append(task)

        # Wait for all models or shutdown
        try:
            await asyncio.gather(*model_tasks)
        except asyncio.CancelledError:
            logger.warning("Tasks cancelled, checkpoints already saved.")

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    for model_cfg in selected_models:
        label = model_cfg.label
        counts = checkpoint.get_counts(label)
        total_done = checkpoint.get_total_done(label)
        logger.info(
            f"  {label:>16}: "
            f"{total_done}/{len(questions)} done | "
            f"new={counts.get('success', 0)} | "
            f"failed={counts.get('failed', 0)} | "
            f"resumed={counts.get('skipped', 0)}"
        )

    logger.info(f"\n✅ Results saved to: {cfg.output_dir}/")

    if shutdown_event.is_set():
        logger.info("💾 Partial results saved. Re-run to resume from checkpoint.")
        sys.exit(0)


if __name__ == "__main__":
    main()
