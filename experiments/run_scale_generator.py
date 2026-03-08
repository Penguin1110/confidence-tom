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
import signal
import sys
import time
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from confidence_tom.client import LLMClient
from confidence_tom.dataset_models import MCQuestion
from confidence_tom.parsing import get_parse_stats, parse_mc_response, reset_parse_stats
from confidence_tom.scale_dataset import load_scale_experiment_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


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
    question: MCQuestion,
    k: int = 10,
) -> dict[str, Any] | None:
    """Run a model on a single MC question K times.

    Returns a result dict with behavioral confidence, reported confidence,
    majority answer, etc. — all in the unified format.
    """
    choices_str = "\n".join(question.choices)

    system_prompt = (
        "You are answering a multiple-choice question.\n"
        "Read the question carefully, think step by step, "
        "and choose the best answer.\n\n"
        "You MUST respond in this EXACT JSON format:\n"
        '{"answer": "B", "confidence": 45, "reasoning": "..."}\n\n'
        "Rules:\n"
        "- answer: MUST be exactly one letter: A, B, C, or D\n"
        "- confidence: integer from 0 to 100. Your confidence should reflect the probability "
        "that your answer is correct. For example, confidence 70 means you expect to be "
        "correct about 70% of the time on similar questions.\n"
        "  Use this scale:\n"
        "  0-20: almost pure guess\n"
        "  21-40: weakly supported, likely wrong\n"
        "  41-60: uncertain / mixed evidence\n"
        "  61-80: probably correct but not fully sure\n"
        "  81-100: very likely correct with strong evidence\n"
        "  Do NOT default to 95. Use the full range when appropriate.\n"
        "- reasoning: brief step-by-step reasoning"
    )

    user_prompt = f"Question: {question.question}\n\n{choices_str}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    samples: list[dict[str, Any]] = []

    async def fetch_one(i: int) -> dict[str, Any] | None:
        """Fetch a single sample via text generation + robust parsing.

        All models use the same pipeline (text → parse) for fairness.
        Structured output was removed because Gemma 12B couldn't
        use it, causing systematic bias.
        """
        try:
            raw_text = await asyncio.to_thread(client.generate_text, messages)
            if raw_text:
                mc_resp = parse_mc_response(raw_text, model_name=model_id)
                if mc_resp:
                    return {
                        "answer": mc_resp.answer,
                        "confidence": mc_resp.confidence,
                        "reasoning": mc_resp.reasoning,
                        "parse_method": "text",
                    }
        except Exception as e:
            logger.debug(f"  Sample {i + 1} failed: {e}")

        return None

    # Run K samples concurrently
    tasks = [fetch_one(i) for i in range(k)]
    results = await asyncio.gather(*tasks)

    for r in results:
        if r is not None:
            samples.append(r)

    if not samples:
        return None

    # Compute behavioral confidence
    answers = [s["answer"] for s in samples]
    counter = collections.Counter(answers)
    majority_answer, majority_count = counter.most_common(1)[0]
    c_beh = majority_count / len(samples)

    # Average reported confidence (0-1)
    c_rep = sum(s["confidence"] for s in samples) / len(samples)

    # Is majority answer correct?
    is_correct = majority_answer == question.correct_answer

    # Primary reasoning from majority-answer sample
    primary_reasoning = next(
        (s["reasoning"] for s in samples if s["answer"] == majority_answer),
        "",
    )

    n_structured = sum(1 for s in samples if s.get("parse_method") == "structured")
    n_fallback = sum(1 for s in samples if s.get("parse_method") == "fallback")

    return {
        "question_id": question.id,
        "question": question.question,
        "choices": question.choices,
        "correct_answer": question.correct_answer,
        "category": question.category,
        "source": question.source,
        "external_difficulty": question.external_difficulty,
        "model_name": model_id,
        "k_samples": len(samples),
        "k_target": k,
        "majority_answer": majority_answer,
        "is_correct": is_correct,
        "c_beh": round(c_beh, 4),
        "c_rep": round(c_rep, 4),
        "gap": round(c_rep - c_beh, 4),
        "primary_reasoning": primary_reasoning,
        "answer_distribution": dict(counter),
        "sample_confidences": [round(s["confidence"], 4) for s in samples],
        "parse_structured": n_structured,
        "parse_fallback": n_fallback,
    }


# ---- Per-model worker ----


async def run_model(
    model_id: str,
    model_label: str,
    questions: list[MCQuestion],
    gen_cfg: Any,
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

    total = len(questions)
    start_time = time.time()

    async def process_one(q: MCQuestion) -> None:
        if checkpoint.is_processed(model_label, q.id):
            return

        async with semaphore:
            result = await solve_question_k_times(
                client=client,
                model_id=model_id,
                question=q,
                k=gen_cfg.k_samples,
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
    config_name="scale_experiment",
)
def main(cfg: DictConfig) -> None:
    asyncio.run(amain(cfg))


async def amain(cfg: DictConfig) -> None:
    # Load dataset once (shared across all models)
    questions = load_scale_experiment_dataset(num_per_source=cfg.dataset.num_per_source)
    logger.info(f"📦 Loaded {len(questions)} questions")

    # Checkpoint manager (thread-safe per-model saving)
    checkpoint = CheckpointManager(Path(cfg.output_dir))

    # Concurrency: semaphore per model
    max_per_model = cfg.concurrency.get("max_per_model", 8)

    logger.info(
        f"⚙️  Config: "
        f"{len(cfg.scale_models)} models × "
        f"{len(questions)} questions × "
        f"K={cfg.generator.k_samples} = "
        f"{len(cfg.scale_models) * len(questions) * cfg.generator.k_samples} "
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

    # ---- Launch ALL models in parallel ----
    model_tasks = []
    for model_cfg in cfg.scale_models:
        sem = asyncio.Semaphore(max_per_model)
        task = asyncio.create_task(
            run_model(
                model_id=model_cfg.id,
                model_label=model_cfg.label,
                questions=questions,
                gen_cfg=cfg.generator,
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
    for model_cfg in cfg.scale_models:
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
