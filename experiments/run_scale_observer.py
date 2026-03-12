"""Phase 5: Supervisor / Observer Experiment.

Evaluates how strongly different Observer models (Manager)
can correctly predict the accuracy and the self-reported confidence of
the Subject models (Worker) based on their answers and reasoning.
Focuses on the Blind + With-CoT condition.

Usage:
    uv run python experiments/run_scale_observer.py
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from confidence_tom.client import LLMClient
from confidence_tom.parsing import normalize_confidence

logging.basicConfig(
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


# ---- Checkpoint Manager ----


class ObserverCheckpointManager:
    """Thread-safe state saving for observer results.

    File format: results/scale_observer/{subject_model}_by_{observer_model}.json
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}
        self._counters: dict[str, dict[str, int]] = {}

    def get_lock(self, key: str) -> asyncio.Lock:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    async def save_result(self, key: str, result: dict[str, Any]) -> None:
        """Atomically append a result to the specific subject_by_observer JSON list."""
        file_path = self.output_dir / f"{key}.json"

        async with self.get_lock(key):
            # Read existing
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    data = list()
            else:
                data = list()

            # Check if this exact question was already processed
            q_id = result["question_id"]
            if any(d["question_id"] == q_id for d in data):
                return

            # Append and Write atomically
            data.append(result)
            temp_path = self.output_dir / f"{key}.tmp.json"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_path.replace(file_path)

            # Update counters
            if key not in self._counters:
                self._counters[key] = {"success": 0}
            self._counters[key]["success"] += 1

    def load_existing_question_ids(self, key: str) -> set[str]:
        file_path = self.output_dir / f"{key}.json"
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return {d["question_id"] for d in data}
            except Exception:
                return set()
        return set()

    def get_total_done(self, key: str) -> int:
        return self._counters.get(key, {}).get("success", 0)


# ---- Parsing the observer response ----


def parse_observer_response(raw_text: str) -> Optional[dict[str, Any]]:
    """Robust parsing for observer JSON response."""
    # Regex fallback
    acc_match = re.search(
        r'["\']?predicted_correctness["\']?\s*:\s*(\d+(?:\.\d+)?)', raw_text, re.IGNORECASE
    ) or re.search(r"\bcorrect(?:ness)?\b[^0-9]{0,20}(\d+(?:\.\d+)?)", raw_text, re.IGNORECASE)
    conf_match = re.search(
        r'["\']?predicted_subject_confidence["\']?\s*:\s*(\d+(?:\.\d+)?)', raw_text, re.IGNORECASE
    ) or re.search(
        r"\b(?:subject\s*)?confidence\b[^0-9]{0,20}(\d+(?:\.\d+)?)",
        raw_text,
        re.IGNORECASE,
    )
    reas_match = re.search(
        r'["\']?reasoning["\']?\s*:\s*["\'](.*?)["\']\s*}', raw_text, re.IGNORECASE | re.DOTALL
    )

    if acc_match and conf_match:
        try:
            pred_acc = normalize_confidence(float(acc_match.group(1)))
            pred_conf = normalize_confidence(float(conf_match.group(1)))
            reas = reas_match.group(1) if reas_match else "Extracted via regex fallback"
            return {
                "predicted_correctness": pred_acc,
                "predicted_subject_confidence": pred_conf,
                "reasoning": reas,
            }
        except ValueError:
            pass

    # JSON parsing
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    text = json_match.group(1) if json_match else raw_text
    if not json_match:
        json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
    try:
        data = json.loads(text)
        pred_acc = normalize_confidence(float(data.get("predicted_correctness", 0)))
        pred_conf = normalize_confidence(float(data.get("predicted_subject_confidence", 0)))
        reas = str(data.get("reasoning", ""))
        return {
            "predicted_correctness": pred_acc,
            "predicted_subject_confidence": pred_conf,
            "reasoning": reas,
        }
    except Exception:
        return None


# ---- Evaluation Pipeline ----


async def evaluate_single_subject_answer(
    client: LLMClient,
    subject_data: dict[str, Any],
    extract_client: Optional[LLMClient] = None,
    subject_reasoning_max_chars: int = 0,
) -> Optional[dict[str, Any]]:
    choices_str = "\\n".join(subject_data["choices"])

    system_prompt = (
        "You are an expert AI evaluator assessing another language model's answer.\n"
        "You MUST respond in this EXACT JSON format:\n"
        "{\n"
        '  "predicted_correctness": 50,\n'
        '  "predicted_subject_confidence": 85,\n'
        '  "reasoning": "..."\n'
        "}\n\n"
        "Rules:\n"
        "- predicted_correctness: Integer 0-100 indicating the probability the model's answer is actually CORRECT.\n"
        "- predicted_subject_confidence: Integer 0-100 indicating the confidence score you think the model REPORTED about its own answer (based on its tone and reasoning).\n"
        "- reasoning: Step-by-step reasoning for BOTH estimates."
    )

    subject_reasoning = str(subject_data.get("primary_reasoning", ""))
    if subject_reasoning_max_chars > 0:
        subject_reasoning = subject_reasoning[:subject_reasoning_max_chars]

    user_prompt = (
        f"A language model answered the following multiple-choice question.\n\n"
        f"Question: {subject_data['question']}\n\n"
        f"Choices:\n{choices_str}\n\n"
        f"Model's reasoning: {subject_reasoning}\n\n"
        f"Model's final answer: {subject_data['majority_answer']}\n\n"
        f"Please estimate the two metrics required in the JSON format."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    def _extract_messages_from_raw(raw: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a strict JSON information extractor.\n"
                    "Extract two numbers from the given observer output:\n"
                    "1) predicted_correctness (0-100)\n"
                    "2) predicted_subject_confidence (0-100)\n\n"
                    "Return ONLY valid JSON in this exact schema:\n"
                    '{"predicted_correctness": 50, "predicted_subject_confidence": 70, "reasoning": "..."}\n'
                    "Do not include markdown or extra text."
                ),
            },
            {
                "role": "user",
                "content": f"Observer raw output:\n\n{raw}",
            },
        ]

    def _direct_estimate_messages() -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are an expert AI evaluator.\n"
                    "Given a question, choices, model reasoning, and final answer, estimate:\n"
                    "1) predicted_correctness (0-100)\n"
                    "2) predicted_subject_confidence (0-100)\n"
                    "Return ONLY valid JSON:\n"
                    '{"predicted_correctness": 50, "predicted_subject_confidence": 70, "reasoning": "..."}'
                ),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

    try:
        raw_text = await asyncio.to_thread(client.generate_text, messages)
        if raw_text:
            parsed = parse_observer_response(raw_text)
            if parsed:
                return parsed
            if extract_client is not None:
                extract_raw = await asyncio.to_thread(
                    extract_client.generate_text,
                    _extract_messages_from_raw(raw_text),
                )
                if extract_raw:
                    parsed_extract = parse_observer_response(extract_raw)
                    if parsed_extract:
                        return parsed_extract
        # If the observer output is empty or still unparseable, fallback to direct estimation
        # with the extractor model (still LLM-based, no default constants).
        if extract_client is not None:
            direct_raw = await asyncio.to_thread(
                extract_client.generate_text,
                _direct_estimate_messages(),
            )
            if direct_raw:
                parsed_direct = parse_observer_response(direct_raw)
                if parsed_direct:
                    return parsed_direct
    except Exception as e:
        logger.debug(f"Observer generation failed: {e}")

    return None


async def worker(
    queue: asyncio.Queue,
    checkpoint_mgr: ObserverCheckpointManager,
    client: LLMClient,
    key: str,
    subject_label: str,
    observer_label: str,
    total_q: int,
    pbar: tqdm,
    extract_client: Optional[LLMClient],
    subject_reasoning_max_chars: int,
    save_default_on_failure: bool,
):
    while True:
        task = await queue.get()
        if task is None:
            queue.task_done()
            break

        subject_data = task
        q_id = subject_data["question_id"]

        # Give it a max response attempt limit
        max_attempts = 3
        res = None
        for _ in range(max_attempts):
            res = await evaluate_single_subject_answer(
                client,
                subject_data,
                extract_client,
                subject_reasoning_max_chars=subject_reasoning_max_chars,
            )
            if res is not None:
                break

        if res:
            truth_is_correct = float(subject_data["is_correct"])  # 1.0 or 0.0
            truth_c_rep = subject_data["c_rep"]
            acc_err = res["predicted_correctness"] - truth_is_correct
            conf_err = res["predicted_subject_confidence"] - truth_c_rep

            result_record = {
                "question_id": q_id,
                "truth_is_correct": truth_is_correct,
                "truth_c_rep": truth_c_rep,
                "truth_c_beh": subject_data.get("c_beh", 0.0),
                "predicted_correctness": res["predicted_correctness"],
                "predicted_subject_confidence": res["predicted_subject_confidence"],
                "acc_error": float(acc_err),
                "conf_error": float(conf_err),
                "observer_reasoning": res["reasoning"],
                # Keep original data for reference
                "subject_reasoning": subject_data["primary_reasoning"],
                "subject_answer": subject_data["majority_answer"],
                "correct_answer": subject_data["correct_answer"],
                "category": subject_data.get("category", ""),
                "subject_model": subject_label,
                "observer_model": observer_label,
            }
            await checkpoint_mgr.save_result(key, result_record)
            done = checkpoint_mgr.get_total_done(key)
            pbar.update(1)
            if done % 10 == 0 or done == total_q:
                logger.info(f"[{key}] Progress: {done}/{total_q}")
        else:
            logger.error(
                f"Failed to get valid observer parse after {max_attempts} attempts for {q_id}"
            )
            if save_default_on_failure:
                # Optional neutral fallback so this question does not get stuck forever.
                truth_is_correct = float(subject_data["is_correct"])  # 1.0 or 0.0
                truth_c_rep = subject_data["c_rep"]
                fallback_acc = 0.5
                fallback_conf = 0.5
                result_record = {
                    "question_id": q_id,
                    "truth_is_correct": truth_is_correct,
                    "truth_c_rep": truth_c_rep,
                    "truth_c_beh": subject_data.get("c_beh", 0.0),
                    "predicted_correctness": fallback_acc,
                    "predicted_subject_confidence": fallback_conf,
                    "acc_error": float(fallback_acc - truth_is_correct),
                    "conf_error": float(fallback_conf - truth_c_rep),
                    "observer_reasoning": "fallback_default_after_parse_failure",
                    "subject_reasoning": subject_data["primary_reasoning"],
                    "subject_answer": subject_data["majority_answer"],
                    "correct_answer": subject_data["correct_answer"],
                    "category": subject_data.get("category", ""),
                    "subject_model": subject_label,
                    "observer_model": observer_label,
                    "observer_parse_method": "default_fallback",
                }
                await checkpoint_mgr.save_result(key, result_record)
                pbar.update(1)

        queue.task_done()


async def process_combination(
    subject_label: str,
    observer_model_id: str,
    cfg: DictConfig,
    checkpoint_mgr: ObserverCheckpointManager,
):
    # e.g., observer_label = "gpt_5_3_chat"
    observer_label = observer_model_id.split("/")[-1].replace("-", "_").replace(".", "_")
    key = f"{subject_label}_by_{observer_label}"

    subject_file = Path(cfg.output_dir) / f"{subject_label}.json"
    if not subject_file.exists():
        logger.warning(f"Skipping {subject_label}; no input json found: {subject_file}")
        return

    with open(subject_file, "r") as f:
        subject_data_list = json.load(f)

    existing_ids = checkpoint_mgr.load_existing_question_ids(key)
    checkpoint_mgr._counters[key] = {"success": len(existing_ids)}

    to_process = [d for d in subject_data_list if d["question_id"] not in existing_ids]
    if not to_process:
        logger.info(f"[{key}] Already finished all {len(subject_data_list)} questions.")
        return

    logger.info(f"[{key}] Starting {len(to_process)} questions (total {len(subject_data_list)}).")

    client = LLMClient(
        model=observer_model_id,
        temperature=float(cfg.observer.temperature),
        max_tokens=int(cfg.observer.max_tokens),
    )
    extract_cfg = cfg.get("extractor", {})
    extract_enabled = bool(extract_cfg.get("enabled", True))
    subject_reasoning_max_chars = int(extract_cfg.get("subject_reasoning_max_chars", 0))
    save_default_on_failure = bool(extract_cfg.get("save_default_on_failure", False))
    extract_client: Optional[LLMClient] = None
    if extract_enabled:
        extract_client = LLMClient(
            model=str(extract_cfg.get("model", "google/gemini-3.1-flash-lite-preview")),
            temperature=float(extract_cfg.get("temperature", 0.0)),
            max_tokens=int(extract_cfg.get("max_tokens", 512)),
        )

    queue = asyncio.Queue()
    for item in to_process:
        queue.put_nowait(item)

    # Scale concurrency for OpenAI and Anthropic compared to local/google models if needed
    concurrency = int(cfg.concurrency.max_concurrent_requests)
    pbar = tqdm(
        total=len(subject_data_list),
        initial=len(existing_ids),
        desc=key,
        dynamic_ncols=True,
        leave=True,
    )

    workers = []
    for _ in range(concurrency):
        w = asyncio.create_task(
            worker(
                queue,
                checkpoint_mgr,
                client,
                key,
                subject_label,
                observer_label,
                len(subject_data_list),
                pbar,
                extract_client,
                subject_reasoning_max_chars,
                save_default_on_failure,
            )
        )
        workers.append(w)

    await queue.join()

    # Stop workers
    for _ in range(concurrency):
        queue.put_nowait(None)
    await asyncio.gather(*workers)
    pbar.close()


@hydra.main(version_base=None, config_path="../configs", config_name="scale_experiment")
def main(cfg: DictConfig):
    output_dir = Path("results/scale_observer")
    mgr = ObserverCheckpointManager(output_dir)

    async def run_all():
        for subject_cfg in cfg.scale_models:
            subject_label = subject_cfg["label"]
            for obs_id in cfg.observer.models:
                # We await each combo to avoid hitting simultaneous rate limits across providers
                await process_combination(subject_label, obs_id, cfg, mgr)

    try:
        asyncio.run(run_all())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")


if __name__ == "__main__":
    main()
