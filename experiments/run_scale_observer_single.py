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
from confidence_tom.observer.models import ErrorType
from confidence_tom.parsing import normalize_confidence
from confidence_tom.task_models import ApiTrace

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
        r'["\']?predicted_worker_confidence["\']?\s*:\s*(\d+(?:\.\d+)?)', raw_text, re.IGNORECASE
    ) or re.search(
        r'["\']?predicted_subject_confidence["\']?\s*:\s*(\d+(?:\.\d+)?)', raw_text, re.IGNORECASE
    ) or re.search(
        r"\b(?:worker|subject)\s*confidence\b[^0-9]{0,20}(\d+(?:\.\d+)?)",
        raw_text,
        re.IGNORECASE,
    )
    mgr_conf_match = re.search(
        r'["\']?manager_self_confidence["\']?\s*:\s*(\d+(?:\.\d+)?)', raw_text, re.IGNORECASE
    ) or re.search(
        r"\bmanager\s*self\s*confidence\b[^0-9]{0,20}(\d+(?:\.\d+)?)",
        raw_text,
        re.IGNORECASE,
    )
    reas_match = re.search(
        r'["\']?judge_reasoning["\']?\s*:\s*["\'](.*?)["\']\s*[,}]',
        raw_text,
        re.IGNORECASE | re.DOTALL,
    ) or re.search(
        r'["\']?reasoning["\']?\s*:\s*["\'](.*?)["\']\s*}', raw_text, re.IGNORECASE | re.DOTALL
    )
    err_match = re.search(
        r'["\']?predicted_error_type["\']?\s*:\s*["\']?([A-Za-z_]+)["\']?',
        raw_text,
        re.IGNORECASE,
    )

    if acc_match and conf_match:
        try:
            pred_acc = normalize_confidence(float(acc_match.group(1)))
            pred_conf = normalize_confidence(float(conf_match.group(1)))
            mgr_conf = normalize_confidence(float(mgr_conf_match.group(1))) if mgr_conf_match else 0.5
            reas = reas_match.group(1) if reas_match else "Extracted via regex fallback"
            error_type = err_match.group(1) if err_match else ErrorType.None_.value
            return {
                "predicted_correctness": pred_acc,
                "predicted_worker_confidence": pred_conf,
                "predicted_error_type": error_type,
                "manager_self_confidence": mgr_conf,
                "judge_reasoning": reas,
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
        pred_conf = normalize_confidence(
            float(
                data.get(
                    "predicted_worker_confidence",
                    data.get("predicted_subject_confidence", 0),
                )
            )
        )
        mgr_conf = normalize_confidence(float(data.get("manager_self_confidence", 50)))
        reas = str(data.get("judge_reasoning", data.get("reasoning", "")))
        error_type = str(data.get("predicted_error_type", ErrorType.None_.value))
        return {
            "predicted_correctness": pred_acc,
            "predicted_worker_confidence": pred_conf,
            "predicted_error_type": error_type,
            "manager_self_confidence": mgr_conf,
            "judge_reasoning": reas,
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
        '  "judge_reasoning": "...",\n'
        '  "predicted_correctness": 0.50,\n'
        '  "predicted_worker_confidence": 0.85,\n'
        '  "predicted_error_type": "Logic_Error",\n'
        '  "manager_self_confidence": 0.80\n'
        "}\n\n"
        "Rules:\n"
        "- judge_reasoning: Step-by-step reasoning for your judgment.\n"
        "- predicted_correctness: Float 0.0-1.0 indicating the probability the model's answer is correct.\n"
        "- predicted_worker_confidence: Float 0.0-1.0 indicating the worker's implied true confidence.\n"
        "- predicted_error_type: One of Logic_Error, Hallucination, Tool_Argument_Error, Observation_Ignored, None.\n"
        "- manager_self_confidence: Float 0.0-1.0 indicating your confidence in this evaluation."
    )

    subject_reasoning = str(subject_data.get("primary_reasoning", ""))
    subject_strategy = str(
        subject_data.get("strategy", subject_data.get("static_trace", {}).get("strategy", ""))
    )
    if subject_reasoning_max_chars > 0:
        subject_reasoning = subject_reasoning[:subject_reasoning_max_chars]

    user_prompt = (
        f"A language model answered the following multiple-choice question.\n\n"
        f"Question: {subject_data['question']}\n\n"
        f"Choices:\n{choices_str}\n\n"
        f"Model's strategy: {subject_strategy}\n\n"
        f"Model's reasoning: {subject_reasoning}\n\n"
        f"Model's final answer: {subject_data['majority_answer']}\n\n"
        f"Please estimate the required metrics in the JSON format."
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
                    "Extract the unified manager judgment from the given observer output.\n\n"
                    "Return ONLY valid JSON in this exact schema:\n"
                    '{"judge_reasoning":"...","predicted_correctness":0.5,"predicted_worker_confidence":0.7,"predicted_error_type":"Logic_Error","manager_self_confidence":0.6}\n'
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
                    "Given a question, choices, model strategy, model reasoning, and final answer, "
                    "estimate the unified manager judgment.\n"
                    "Return ONLY valid JSON:\n"
                    '{"judge_reasoning":"...","predicted_correctness":0.5,"predicted_worker_confidence":0.7,"predicted_error_type":"Logic_Error","manager_self_confidence":0.6}'
                ),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

    try:
        raw_text, observer_trace = await client.agenerate_text_with_trace(messages)
        if raw_text:
            parsed = parse_observer_response(raw_text)
            if parsed:
                parsed["api_trace"] = observer_trace.model_dump()
                return parsed
            if extract_client is not None:
                extract_raw, extract_trace = await extract_client.agenerate_text_with_trace(
                    _extract_messages_from_raw(raw_text)
                )
                if extract_raw:
                    parsed_extract = parse_observer_response(extract_raw)
                    if parsed_extract:
                        parsed_extract["api_trace"] = observer_trace.model_dump()
                        parsed_extract["extract_api_trace"] = extract_trace.model_dump()
                        return parsed_extract
        # If the observer output is empty or still unparseable, fallback to direct estimation
        # with the extractor model (still LLM-based, no default constants).
        if extract_client is not None:
            direct_raw, direct_trace = await extract_client.agenerate_text_with_trace(
                _direct_estimate_messages()
            )
            if direct_raw:
                parsed_direct = parse_observer_response(direct_raw)
                if parsed_direct:
                    parsed_direct["api_trace"] = direct_trace.model_dump()
                    parsed_direct["extract_api_trace"] = direct_trace.model_dump()
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
            conf_err = res["predicted_worker_confidence"] - truth_c_rep

            result_record = {
                "question_id": q_id,
                "truth_is_correct": truth_is_correct,
                "truth_c_rep": truth_c_rep,
                "truth_c_beh": subject_data.get("c_beh", 0.0),
                "predicted_correctness": res["predicted_correctness"],
                "predicted_worker_confidence": res["predicted_worker_confidence"],
                "predicted_subject_confidence": res["predicted_worker_confidence"],
                "predicted_error_type": res["predicted_error_type"],
                "manager_self_confidence": res["manager_self_confidence"],
                "acc_error": float(acc_err),
                "conf_error": float(conf_err),
                "judge_reasoning": res["judge_reasoning"],
                "observer_reasoning": res["judge_reasoning"],
                # Keep original data for reference
                "task_type": subject_data.get("task_type", "QA"),
                "environment_context": subject_data.get("environment_context", []),
                "subject_strategy": subject_data.get(
                    "strategy", subject_data.get("static_trace", {}).get("strategy", "")
                ),
                "subject_reasoning": subject_data["primary_reasoning"],
                "subject_answer": subject_data["majority_answer"],
                "correct_answer": subject_data["correct_answer"],
                "category": subject_data.get("category", ""),
                "subject_model": subject_label,
                "observer_model": observer_label,
                "observer_api_trace": res.get("api_trace", ApiTrace().model_dump()),
                "extractor_api_trace": res.get("extract_api_trace", ApiTrace().model_dump()),
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
                    "predicted_worker_confidence": fallback_conf,
                    "predicted_subject_confidence": fallback_conf,
                    "predicted_error_type": ErrorType.None_.value,
                    "manager_self_confidence": 0.0,
                    "acc_error": float(fallback_acc - truth_is_correct),
                    "conf_error": float(fallback_conf - truth_c_rep),
                    "judge_reasoning": "fallback_default_after_parse_failure",
                    "observer_reasoning": "fallback_default_after_parse_failure",
                    "task_type": subject_data.get("task_type", "QA"),
                    "environment_context": subject_data.get("environment_context", []),
                    "subject_strategy": subject_data.get(
                        "strategy", subject_data.get("static_trace", {}).get("strategy", "")
                    ),
                    "subject_reasoning": subject_data["primary_reasoning"],
                    "subject_answer": subject_data["majority_answer"],
                    "correct_answer": subject_data["correct_answer"],
                    "category": subject_data.get("category", ""),
                    "subject_model": subject_label,
                    "observer_model": observer_label,
                    "observer_parse_method": "default_fallback",
                    "observer_api_trace": ApiTrace().model_dump(),
                    "extractor_api_trace": ApiTrace().model_dump(),
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


@hydra.main(version_base=None, config_path="../configs", config_name="scale_single")
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
