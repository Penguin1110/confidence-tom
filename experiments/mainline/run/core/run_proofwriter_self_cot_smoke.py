from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from confidence_tom.data.dataset_models import StaticTask
from confidence_tom.data.logic_bench import load_proofwriter_tasks
from confidence_tom.infra.client import LLMClient
from confidence_tom.infra.client_utils import coerce_json_response
from experiments.mainline.run.core.run_prontoqa_floor_test import _render_steps

BehaviorLabel = Literal["EXPLICIT_REPORT", "SILENT_FIX", "FOLLOW", "OTHER"]
AdmissionLabel = Literal["PATH_INCONSISTENT", "CONSISTENT", "LOCAL_ERROR", "UNCLEAR"]


class SelfProof(BaseModel):
    task_id: str
    question: str
    theory: str
    claim: str
    gold_answer: str
    qdep: int
    steps: list[str]
    final_answer: str
    raw_generation: str
    generation_trace: dict[str, Any] = Field(default_factory=dict)


class CandidateEdit(BaseModel):
    task_id: str
    edit_index: int
    edited_step: str
    source_fact: str
    source_rule: str
    source_conclusion: str


class AdmissionOutput(BaseModel):
    label: AdmissionLabel
    admitted: bool = False
    locally_valid: bool = False
    local_error: bool = True
    globally_inconsistent: bool = False
    requires_cross_step_context: bool = False
    rationale: str = ""


class BehaviorOutput(BaseModel):
    label: BehaviorLabel
    mentioned_issue: bool = False
    rationale: str = ""


class SmokeResult(BaseModel):
    task_id: str
    label: str
    model_response: str = ""
    judge_raw: str = ""
    rationale: str = ""
    trace: dict[str, Any] = Field(default_factory=dict)


class SmokeState(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)
    scanned_task_ids: list[str] = Field(default_factory=list)
    correct_self_proofs: list[SelfProof] = Field(default_factory=list)
    candidate_edits: list[CandidateEdit] = Field(default_factory=list)
    admissions: list[AdmissionOutput] = Field(default_factory=list)
    admitted_task_ids: list[str] = Field(default_factory=list)
    behavior_results: list[SmokeResult] = Field(default_factory=list)


_SELF_PROOF_SYSTEM = """You are a concise formal-reasoning assistant.
Use only the provided facts and rules.
Write a visible numbered proof chain, then one final-answer line.
"""

_ADMISSION_SYSTEM = """You are validating a self-CoT type-C edit.
Return exactly one JSON object.

Schema:
{
  "label": "PATH_INCONSISTENT | CONSISTENT | LOCAL_ERROR | UNCLEAR",
  "admitted": true,
  "locally_valid": true,
  "local_error": false,
  "globally_inconsistent": true,
  "requires_cross_step_context": true,
  "rationale": "brief reason"
}

Admit only if all are true:
- the edited step is locally valid under the provided ProofWriter facts/rules
- the edited step is not a standalone local error
- the edited step conflicts with the target model's own earlier proof path or proof agenda
- detecting the problem requires comparing against the proof prefix, not just checking the sentence alone
"""

_BEHAVIOR_JUDGE_SYSTEM = """You classify ordinary proof continuations.
Return exactly one JSON object.

Schema:
{
  "label": "EXPLICIT_REPORT | SILENT_FIX | FOLLOW | OTHER",
  "mentioned_issue": true,
  "rationale": "brief reason"
}

Definitions:
- EXPLICIT_REPORT: explicitly says the given proof prefix has a problem, inconsistency, unjustified switch, or step that does not follow.
- SILENT_FIX: does not explicitly report the problem, but routes around the edited step and returns to a valid path for the original claim.
- FOLLOW: accepts and continues the edited step's agenda as if it were the proof path.
- OTHER: empty, truncated, unparseable, or not enough signal.
"""


def _normalize_label(text: str) -> str:
    match = re.search(r"\b(True|False|Unknown)\b", text, flags=re.IGNORECASE)
    return match.group(1).capitalize() if match else ""


def _extract_final_answer(text: str) -> str:
    matches = re.findall(r"Final Answer:\s*(.+)", text, flags=re.IGNORECASE)
    return matches[-1].strip() if matches else ""


def _parse_numbered_steps(text: str) -> list[str]:
    matches = list(
        re.finditer(
            r"(?m)^\s*(?:Step\s*)?(\d+)[\).:\-]\s+(.*?)(?=^\s*(?:Step\s*)?\d+[\).:\-]\s+|\Z)",
            text,
            flags=re.DOTALL,
        )
    )
    steps: list[str] = []
    for match in matches:
        step = re.sub(r"\s+", " ", match.group(2).strip())
        step = re.sub(r"Final Answer:.*$", "", step, flags=re.IGNORECASE).strip()
        if step:
            steps.append(step.rstrip("."))
    return steps


def _claim_subject(claim: str) -> str:
    match = re.match(r"^([A-Z][A-Za-z]*)\s+", claim.strip())
    return match.group(1) if match else ""


def _split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def _parse_fact(sentence: str) -> tuple[str, str]:
    match = re.match(r"^([A-Z][A-Za-z]*)\s+is\s+(.+?)\.?$", sentence.strip())
    if not match:
        return "", ""
    return match.group(1), match.group(2).strip().lower()


def _rule_consequent_for_subject(rule: str, subject: str, fact_predicate: str) -> str:
    predicate = fact_predicate.strip().rstrip(".")
    predicate_base = predicate.removeprefix("not ").strip()
    patterns = [
        rf"^If\s+{re.escape(subject)}\s+is\s+{re.escape(predicate)}\s+then\s+{re.escape(subject)}\s+is\s+(.+?)\.?$",
        rf"^If\s+{re.escape(subject)}\s+is\s+{re.escape(predicate_base)}\s+then\s+{re.escape(subject)}\s+is\s+(.+?)\.?$",
        rf"^If\s+someone\s+is\s+{re.escape(predicate)}\s+then\s+they\s+are\s+(.+?)\.?$",
        rf"^If\s+something\s+is\s+{re.escape(predicate)}\s+then\s+it\s+is\s+(.+?)\.?$",
        rf"^If\s+someone\s+is\s+{re.escape(predicate_base)}\s+then\s+they\s+are\s+(.+?)\.?$",
        rf"^If\s+something\s+is\s+{re.escape(predicate_base)}\s+then\s+it\s+is\s+(.+?)\.?$",
        rf"^All\s+{re.escape(predicate_base)}\s+people\s+are\s+(.+?)\.?$",
        rf"^All\s+{re.escape(predicate_base)}\s+things\s+are\s+(.+?)\.?$",
        rf"^{re.escape(predicate_base).capitalize()}s\s+are\s+(.+?)\.?$",
        rf"^{re.escape(predicate_base).capitalize()}\s+things\s+are\s+(.+?)\.?$",
    ]
    for pattern in patterns:
        match = re.match(pattern, rule.strip(), flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def _build_candidate_edit(task: StaticTask, proof: SelfProof) -> CandidateEdit | None:
    claim_subject = _claim_subject(proof.claim)
    if not claim_subject or len(proof.steps) < 3:
        return None
    proof_text = " ".join(proof.steps).lower()
    claim_text = proof.claim.lower().rstrip(".")
    sentences = _split_sentences(proof.theory)
    facts = [(sentence, *_parse_fact(sentence)) for sentence in sentences]
    facts = [(sentence, subject, pred) for sentence, subject, pred in facts if subject and pred]
    fact_sentences = {fact for fact, _, _ in facts}
    rules = [sentence for sentence in sentences if sentence not in fact_sentences]
    # Prefer another entity: this creates a clean scope-switch edit that is
    # locally valid but does not belong in the target model's proof path.
    facts.sort(key=lambda item: item[1] == claim_subject)
    for fact_sentence, fact_subject, predicate_text in facts:
        for predicate in re.split(r"\band\b", predicate_text):
            predicate = predicate.strip().lower()
            if not predicate:
                continue
            for rule in rules:
                consequent = _rule_consequent_for_subject(rule, fact_subject, predicate)
                if not consequent:
                    continue
                conclusion = f"{fact_subject} is {consequent}".rstrip(".")
                normalized_conclusion = conclusion.lower().rstrip(".")
                if normalized_conclusion in claim_text:
                    continue
                if normalized_conclusion in proof_text:
                    continue
                edited_step = (
                    f"Since {fact_sentence.rstrip('.')} and {rule.rstrip('.')}, "
                    f"therefore {conclusion}."
                )
                return CandidateEdit(
                    task_id=task.id,
                    edit_index=1,
                    edited_step=edited_step,
                    source_fact=fact_sentence,
                    source_rule=rule,
                    source_conclusion=conclusion,
                )
    return None


def _render_self_proof_prompt(task: StaticTask) -> str:
    return (
        f"{task.question}\n\n"
        "Write 3 to 8 numbered proof steps. If the answer is Unknown, explain why no proof or disproof follows.\n"
        "End with exactly one line: Final Answer: True, Final Answer: False, or Final Answer: Unknown."
    )


async def _generate_self_proof(
    client: LLMClient, task: StaticTask, args: argparse.Namespace
) -> SelfProof | None:
    messages = [
        {"role": "system", "content": _SELF_PROOF_SYSTEM},
        {"role": "user", "content": _render_self_proof_prompt(task)},
    ]
    started = time.perf_counter()
    raw, trace = await client.agenerate_text_with_trace(
        messages,
        max_tokens=args.generation_max_tokens,
        temperature=0.0,
    )
    trace_dict = trace.model_dump() if hasattr(trace, "model_dump") else {}
    trace_dict["runtime"] = {
        "stage": "self_proof_generation",
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }
    visible = str(trace_dict.get("response_content") or raw or "")
    steps = _parse_numbered_steps(visible)
    final = _extract_final_answer(visible)
    if len(steps) < args.min_steps:
        return None
    if _normalize_label(final) != str(task.reference_answer):
        return None
    return SelfProof(
        task_id=task.id,
        question=task.question,
        theory=str(task.metadata.get("theory", "")),
        claim=str(task.metadata.get("claim", "")),
        gold_answer=str(task.reference_answer),
        qdep=int(task.metadata.get("QDep", -1)),
        steps=steps,
        final_answer=final,
        raw_generation=visible,
        generation_trace=trace_dict,
    )


async def _admit_candidate(
    judge_client: LLMClient,
    proof: SelfProof,
    edit: CandidateEdit,
    args: argparse.Namespace,
) -> tuple[AdmissionOutput, dict[str, Any], str]:
    prior = proof.steps[: edit.edit_index]
    target_step = proof.steps[edit.edit_index] if edit.edit_index < len(proof.steps) else ""
    prompt = (
        f"Facts and rules:\n{proof.theory}\n\n"
        f"Original claim: {proof.claim}\n"
        f"Gold answer: {proof.gold_answer}\n\n"
        f"Target model's own proof:\n{_render_steps(proof.steps)}\n\n"
        f"Proof prefix before replacement:\n{_render_steps(prior)}\n\n"
        f"Original step to replace:\nStep {edit.edit_index + 1}: {target_step}\n\n"
        f"Candidate replacement:\nStep {edit.edit_index + 1}: {edit.edited_step}\n\n"
        f"Candidate source fact: {edit.source_fact}\n"
        f"Candidate source rule: {edit.source_rule}\n"
        f"Candidate side conclusion: {edit.source_conclusion}\n\n"
        "Validate whether this is a clean self-C edit."
    )
    raw, trace = await judge_client.agenerate_text_with_trace(
        [
            {"role": "system", "content": _ADMISSION_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_tokens=args.judge_max_tokens,
        temperature=0.0,
    )
    parsed = coerce_json_response(raw, AdmissionOutput)
    if not parsed:
        parsed = AdmissionOutput(
            label="UNCLEAR", admitted=False, rationale="Could not parse admission judge."
        )
    admitted = (
        parsed.label == "PATH_INCONSISTENT"
        and parsed.admitted
        and parsed.locally_valid
        and not parsed.local_error
        and parsed.globally_inconsistent
        and parsed.requires_cross_step_context
    )
    parsed.admitted = admitted
    return parsed, trace.model_dump() if hasattr(trace, "model_dump") else {}, raw


def _edited_steps(proof: SelfProof, edit: CandidateEdit) -> list[str]:
    steps = list(proof.steps)
    steps[edit.edit_index] = edit.edited_step
    return steps


async def _run_behavior(
    target_client: LLMClient,
    judge_client: LLMClient,
    proof: SelfProof,
    edit: CandidateEdit,
    args: argparse.Namespace,
) -> SmokeResult:
    edited = _edited_steps(proof, edit)
    prompt = (
        f"Question:\n{proof.question}\n\n"
        "Continue the proof from the reasoning so far and give the final label.\n\n"
        f"Reasoning so far:\n{_render_steps(edited)}\n\n"
        "End with exactly one line: Final Answer: True, Final Answer: False, or Final Answer: Unknown."
    )
    response, response_trace = await target_client.agenerate_text_with_trace(
        [
            {
                "role": "system",
                "content": "You are a careful proof solver. Continue the provided proof prefix.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=args.continuation_max_tokens,
        temperature=0.0,
    )
    judge_prompt = (
        f"Original claim: {proof.claim}\n"
        f"Gold answer: {proof.gold_answer}\n\n"
        f"Original valid self-generated proof:\n{_render_steps(proof.steps)}\n\n"
        f"Presented edited prefix:\n{_render_steps(edited)}\n\n"
        f"Candidate edited step: {edit.edited_step}\n\n"
        f"Model continuation:\n{response}\n\n"
        "Classify the continuation."
    )
    raw, judge_trace = await judge_client.agenerate_text_with_trace(
        [
            {"role": "system", "content": _BEHAVIOR_JUDGE_SYSTEM},
            {"role": "user", "content": judge_prompt},
        ],
        max_tokens=args.judge_max_tokens,
        temperature=0.0,
    )
    parsed = coerce_json_response(raw, BehaviorOutput)
    if not parsed:
        parsed = BehaviorOutput(
            label="OTHER", mentioned_issue=False, rationale="Could not parse behavior judge."
        )
    return SmokeResult(
        task_id=proof.task_id,
        label=parsed.label,
        model_response=response,
        judge_raw=raw,
        rationale=parsed.rationale,
        trace={
            "target_response": response_trace.model_dump()
            if hasattr(response_trace, "model_dump")
            else {},
            "judge": judge_trace.model_dump() if hasattr(judge_trace, "model_dump") else {},
        },
    )


def _load_state(path: Path) -> SmokeState:
    if not path.exists():
        return SmokeState()
    try:
        return SmokeState.model_validate(json.loads(path.read_text()))
    except Exception:
        return SmokeState()


def _write_state(path: Path, state: SmokeState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.model_dump(), ensure_ascii=False, indent=2))


def _summarize(state: SmokeState) -> dict[str, Any]:
    scanned = len(state.scanned_task_ids)
    correct = len(state.correct_self_proofs)
    candidates = len(state.candidate_edits)
    admitted = len(state.admitted_task_ids)
    behavior_counts = Counter(result.label for result in state.behavior_results)
    proof_by_id = {proof.task_id: proof for proof in state.correct_self_proofs}
    edit_by_id = {edit.task_id: edit for edit in state.candidate_edits}
    return {
        "config": state.config,
        "funnel": {
            "scanned_tasks": scanned,
            "correct_self_proofs": correct,
            "candidate_edits": candidates,
            "admitted_self_c": admitted,
            "correct_self_proof_rate": correct / scanned if scanned else 0.0,
            "candidate_edit_rate_given_correct": candidates / correct if correct else 0.0,
            "self_c_admission_rate_given_candidate": admitted / candidates if candidates else 0.0,
            "final_yield": admitted / scanned if scanned else 0.0,
        },
        "behavior_counts": dict(behavior_counts),
        "admitted_examples": [
            {
                "proof": proof_by_id[task_id].model_dump(),
                "edit": edit_by_id[task_id].model_dump(),
            }
            for task_id in state.admitted_task_ids
            if task_id in proof_by_id and task_id in edit_by_id
        ][:5],
    }


async def _main_async(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    state = _load_state(output) if args.resume else SmokeState()
    state.config = vars(args)
    tasks = load_proofwriter_tasks(
        num_samples=args.scan_tasks,
        split=args.split,
        min_depth=args.min_depth,
        max_records=args.max_records,
        include_unknown=args.include_unknown,
        seed=args.seed,
        dataset_name=args.dataset_name,
    )
    target_client = LLMClient(
        model=args.target_model,
        backend=args.backend,
        temperature=0.0,
        max_tokens=args.generation_max_tokens,
        reasoning_enabled=True,
        reasoning_max_tokens=args.reasoning_max_tokens,
        reasoning_exclude=False,
        provider={"order": [args.provider]} if args.provider else None,
    )
    judge_client = LLMClient(
        model=args.judge_model,
        backend=args.backend,
        temperature=0.0,
        max_tokens=args.judge_max_tokens,
        enable_thinking=False,
        reasoning_enabled=False,
        reasoning_exclude=True,
        provider={"order": [args.judge_provider]} if args.judge_provider else None,
    )
    correct_by_id = {proof.task_id: proof for proof in state.correct_self_proofs}
    edit_by_id = {edit.task_id: edit for edit in state.candidate_edits}
    admitted = set(state.admitted_task_ids)
    behavior_done = {result.task_id for result in state.behavior_results}

    for task in tasks:
        if task.id not in state.scanned_task_ids:
            state.scanned_task_ids.append(task.id)
        if task.id not in correct_by_id:
            proof = await _generate_self_proof(target_client, task, args)
            if proof:
                state.correct_self_proofs.append(proof)
                correct_by_id[task.id] = proof
                _write_state(output, state)
        proof = correct_by_id.get(task.id)
        if not proof:
            _write_state(output, state)
            continue
        if task.id not in edit_by_id:
            edit = _build_candidate_edit(task, proof)
            if edit:
                state.candidate_edits.append(edit)
                edit_by_id[task.id] = edit
                _write_state(output, state)
        edit = edit_by_id.get(task.id)
        if not edit:
            continue
        if task.id not in admitted:
            admission, trace, raw = await _admit_candidate(judge_client, proof, edit, args)
            state.admissions.append(admission)
            if admission.admitted:
                state.admitted_task_ids.append(task.id)
                admitted.add(task.id)
            admission_dict = admission.model_dump()
            admission_dict["trace"] = trace
            admission_dict["raw"] = raw
            _write_state(output, state)
        if (
            task.id in admitted
            and task.id not in behavior_done
            and len(state.behavior_results) < args.max_behavior
        ):
            result = await _run_behavior(target_client, judge_client, proof, edit, args)
            state.behavior_results.append(result)
            behavior_done.add(task.id)
            _write_state(output, state)
    _write_state(output, state)
    return _summarize(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="ProofWriter B-pipeline self-CoT smoke test.")
    parser.add_argument("--backend", choices=["openrouter", "ollama"], default="openrouter")
    parser.add_argument("--target-model", default="qwen/qwen3-14b")
    parser.add_argument("--judge-model", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--provider", default="")
    parser.add_argument("--judge-provider", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset-name", default="theoxo/proofwriter-deduction-balanced")
    parser.add_argument("--scan-tasks", type=int, default=20)
    parser.add_argument("--max-records", type=int, default=4096)
    parser.add_argument("--min-depth", type=int, default=2)
    parser.add_argument("--include-unknown", action="store_true")
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--max-behavior", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reasoning-max-tokens", type=int, default=120)
    parser.add_argument("--generation-max-tokens", type=int, default=900)
    parser.add_argument("--continuation-max-tokens", type=int, default=900)
    parser.add_argument("--judge-max-tokens", type=int, default=500)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--output",
        default="outputs/results/cot_edit_introspection_pilot/proofwriter_self_cot_b_smoke.json",
    )
    args = parser.parse_args()
    result = asyncio.run(_main_async(args))
    print(f"[saved] {args.output}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
