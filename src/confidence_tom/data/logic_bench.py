"""Logic-benchmark loaders used for commitment-mismatch probes."""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any

from datasets import load_dataset

from confidence_tom.data.dataset_models import StaticTask


def _normalize_prontoqa_query(query: str) -> str:
    text = query.strip()
    if text.lower().startswith("prove:"):
        text = text.split(":", 1)[1].strip()
    return text.rstrip(".")


def _render_prontoqa_question(context: str, query: str) -> str:
    goal = _normalize_prontoqa_query(query)
    return f"Facts and rules:\n{context.strip()}\n\nGoal:\nProve that {goal}."


def _ensure_minimum_logic_trace_length(chain_of_thought: list[str], goal: str) -> list[str]:
    steps = [step.strip() for step in chain_of_thought if step.strip()]
    if len(steps) >= 3:
        return steps
    steps.append(f"Therefore, the goal is established: {goal}.")
    return steps


def _iter_prontoqa_examples(records: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for record in records:
        for key in sorted(record):
            value = record[key]
            if not key.startswith("example") or not isinstance(value, dict):
                continue
            question = str(value.get("question", "")).strip()
            query = str(value.get("query", "")).strip()
            chain = value.get("chain_of_thought", [])
            if not question or not query or not isinstance(chain, list):
                continue
            yield {
                "example_key": key,
                "question": question,
                "query": query,
                "chain_of_thought": [str(step).strip() for step in chain if str(step).strip()],
            }


def load_prontoqa_tasks(
    num_samples: int = 20,
    *,
    seed: int = 42,
    max_records: int = 32,
) -> list[StaticTask]:
    """Load a small ProntoQA sample as open-ended proof tasks.

    The public HF dataset is easiest to consume in streaming mode because the
    non-streaming builder may fail partway through generation. We therefore
    flatten the nested `exampleN` payloads on the fly and then draw a
    deterministic sample.
    """

    dataset = load_dataset("tasksource/prontoqa", split="train", streaming=True)
    records = []
    for idx, record in enumerate(dataset):
        if idx >= max_records:
            break
        records.append(record)

    examples = list(_iter_prontoqa_examples(records))
    rng = random.Random(seed)
    rng.shuffle(examples)
    selected = examples[:num_samples]

    tasks: list[StaticTask] = []
    for idx, example in enumerate(selected):
        query = str(example["query"])
        question = str(example["question"])
        goal = _normalize_prontoqa_query(query)
        cot = _ensure_minimum_logic_trace_length(list(example["chain_of_thought"]), goal)
        tasks.append(
            StaticTask(
                id=f"prontoqa_{idx:04d}",
                question=_render_prontoqa_question(question, query),
                correct_answer="",
                reference_answer=goal,
                category="logic_commitment",
                source="prontoqa",
                answer_format="open_ended",
                evaluator_name="logic_goal",
                task_type="proof",
                metadata={
                    "raw_context": question,
                    "query": query,
                    "goal": goal,
                    "chain_of_thought": cot,
                    "example_key": str(example["example_key"]),
                },
                external_difficulty="synthetic_logic",
            )
        )
    return tasks


def _render_proofwriter_question(theory: str, claim: str) -> str:
    return (
        f"Facts and rules:\n{theory.strip()}\n\n"
        f"Claim:\n{claim.strip()}\n\n"
        "Determine whether the claim is True, False, or Unknown."
    )


def load_proofwriter_tasks(
    num_samples: int = 20,
    *,
    seed: int = 42,
    split: str = "train",
    min_depth: int = 2,
    max_records: int = 2048,
    include_unknown: bool = False,
    dataset_name: str = "tasksource/proofwriter",
) -> list[StaticTask]:
    """Load ProofWriter tasks with a minimum question proof depth."""

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    candidates: list[dict[str, Any]] = []
    for idx, record in enumerate(dataset):
        if idx >= max_records:
            break
        answer = str(record.get("answer", "")).strip()
        if answer == "Uncertain":
            answer = "Unknown"
        if answer not in {"True", "False", "Unknown"}:
            continue
        if answer == "Unknown" and not include_unknown:
            continue
        qdep = int(record.get("QDep", -1))
        if qdep < min_depth:
            continue
        theory = str(record.get("theory", "")).strip()
        question = str(record.get("question", "")).strip()
        if not theory or not question:
            continue
        normalized = dict(record)
        normalized["answer"] = answer
        candidates.append(normalized)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:num_samples]

    tasks: list[StaticTask] = []
    for idx, record in enumerate(selected):
        theory = str(record.get("theory", "")).strip()
        claim = str(record.get("question", "")).strip().rstrip(".")
        answer = str(record.get("answer", "")).strip()
        task_id = str(record.get("id", f"proofwriter_{idx:04d}"))
        tasks.append(
            StaticTask(
                id=f"proofwriter_{idx:04d}_{task_id}",
                question=_render_proofwriter_question(theory, claim),
                correct_answer=answer,
                reference_answer=answer,
                category="logic_commitment",
                source="proofwriter",
                answer_format="open_ended",
                evaluator_name="proofwriter_label",
                task_type="proof",
                metadata={
                    "raw_id": task_id,
                    "theory": theory,
                    "claim": claim,
                    "answer": answer,
                    "QDep": int(record.get("QDep", -1)),
                    "maxD": int(record.get("maxD", -1)),
                    "NFact": int(record.get("NFact", -1)),
                    "NRule": int(record.get("NRule", -1)),
                    "allProofs": str(record.get("allProofs", "")),
                    "config": str(record.get("config", "")),
                    "dataset_name": dataset_name,
                },
                external_difficulty=f"QDep={int(record.get('QDep', -1))}",
            )
        )
    return tasks
