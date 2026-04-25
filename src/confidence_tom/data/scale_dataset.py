"""Dataset loaders for the scale experiment.

All benchmarks are standardized to multiple-choice format.
This ensures consistent parsing and fair confidence comparison across tasks.

Supported benchmarks:
- MMLU: general knowledge QA (4-choice MC)
- ARC-Challenge: science reasoning (4-choice MC)
- TruthfulQA: truthfulness / misconceptions (MC version)
- GSM8K: math reasoning (converted to MC with distractors)
- MuSR: official multiple-choice reasoning benchmark
- OlympiadBench: open-ended olympiad math benchmark
- LiveBench reasoning: open-ended reasoning tasks with task-specific scorers
- AIME 2024: open-ended competition math benchmark
- MATH-500: open-ended math benchmark subset
- GPQA Diamond: hard multiple-choice science benchmark
"""

import ast
import io
import json
import logging
import random
import re
import zipfile
from typing import Callable, Optional, cast

import requests
from datasets import load_dataset

from confidence_tom.data.dataset_models import MCQuestion

logger = logging.getLogger(__name__)

# Reproducibility
_SEED = 42

HARD_MMLU_SUBJECTS = [
    "college_mathematics",
    "college_physics",
    "college_chemistry",
    "econometrics",
    "formal_logic",
    "professional_accounting",
    "abstract_algebra",
]


def load_mmlu(
    subjects: Optional[list[str]] = None,
    num_samples: int = 100,
    split: str = "test",
) -> list[MCQuestion]:
    """Load MMLU questions as standardized MC format.

    Args:
        subjects: Optional list of MMLU subjects to include.
                  If None, samples across all subjects.
        num_samples: Target number of questions.
        split: Dataset split to use.

    Returns:
        List of MCQuestion in standardized format.
    """
    logger.info(f"Loading MMLU ({num_samples} questions)...")

    dataset = load_dataset("cais/mmlu", "all", split=split)
    dataset = dataset.shuffle(seed=_SEED)

    # Filter by subjects if specified
    if subjects:
        subjects_lower = [s.lower() for s in subjects]
        dataset = dataset.filter(lambda x: x["subject"].lower() in subjects_lower)

    choice_labels = ["A", "B", "C", "D"]
    questions: list[MCQuestion] = []

    for i, item in enumerate(dataset):
        if len(questions) >= num_samples:
            break

        raw_choices = item["choices"]
        correct_idx = int(item["answer"])

        formatted_choices = [
            f"{choice_labels[j]}) {raw_choices[j]}" for j in range(len(raw_choices))
        ]

        questions.append(
            MCQuestion(
                id=f"mmlu_{item['subject']}_{i:04d}",
                question=str(item["question"]),
                choices=formatted_choices,
                correct_answer=choice_labels[correct_idx],
                category="knowledge",
                source="mmlu",
                external_difficulty=str(item["subject"]),
            )
        )

    logger.info(f"  Loaded {len(questions)} MMLU questions")
    return questions


def load_mmlu_pro(
    num_samples: int = 100,
    split: str = "test",
) -> list[MCQuestion]:
    """Load MMLU-Pro questions (10-choice MC, much harder than MMLU)."""
    logger.info(f"Loading MMLU-Pro ({num_samples} questions)...")

    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    dataset = dataset.shuffle(seed=_SEED)

    choice_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    questions: list[MCQuestion] = []

    for i, item in enumerate(dataset):
        if len(questions) >= num_samples:
            break

        raw_choices = item["options"]
        correct_idx = int(item["answer_index"])

        # Construct options up to mapping (some may have fewer than 10, but we standardize)
        formatted_choices = []
        for j in range(len(raw_choices)):
            if j < len(choice_labels):
                formatted_choices.append(f"{choice_labels[j]}) {raw_choices[j]}")

        questions.append(
            MCQuestion(
                id=f"mmlu_pro_{item['category']}_{i:04d}",
                question=str(item["question"]),
                choices=formatted_choices,
                correct_answer=choice_labels[correct_idx]
                if correct_idx < len(choice_labels)
                else "A",
                category="knowledge_pro",
                source="mmlu_pro",
                external_difficulty=item["category"],
            )
        )

    logger.info(f"  Loaded {len(questions)} MMLU-Pro questions")
    return questions


def load_arc_challenge(
    num_samples: int = 100,
    split: str = "test",
) -> list[MCQuestion]:
    """Load ARC-Challenge questions as standardized MC format.

    ARC-Challenge is already in MC format with 4 choices (mostly).
    Some questions have 3 or 5 choices — we filter for exactly 4.

    Returns:
        List of MCQuestion in standardized format.
    """
    logger.info(f"Loading ARC-Challenge ({num_samples} questions)...")

    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    dataset = dataset.shuffle(seed=_SEED)

    choice_labels = ["A", "B", "C", "D"]
    questions: list[MCQuestion] = []

    for i, item in enumerate(dataset):
        if len(questions) >= num_samples:
            break

        raw_choices = item["choices"]["text"]
        raw_labels = item["choices"]["label"]
        answer_key = str(item["answerKey"])

        # Skip questions that don't have exactly 4 choices
        if len(raw_choices) != 4:
            continue

        # Map original labels (could be "A","B","C","D" or "1","2","3","4")
        # to our standardized A/B/C/D
        try:
            correct_original_idx = raw_labels.index(answer_key)
        except ValueError:
            continue

        formatted_choices = [
            f"{choice_labels[j]}) {raw_choices[j]}" for j in range(len(raw_choices))
        ]

        questions.append(
            MCQuestion(
                id=f"arc_{i:04d}",
                question=str(item["question"]),
                choices=formatted_choices,
                correct_answer=choice_labels[correct_original_idx],
                category="science",
                source="arc_challenge",
                external_difficulty="challenge",
            )
        )

    logger.info(f"  Loaded {len(questions)} ARC-Challenge questions")
    return questions


def load_truthfulqa_mc(
    num_samples: int = 100,
) -> list[MCQuestion]:
    """Load TruthfulQA in MC format (not generative).

    Uses the multiple_choice configuration which provides pre-built MC options.
    Each question has multiple correct and incorrect options — we construct
    a 4-choice MC by picking 1 correct + 3 incorrect (or as many as available).

    Returns:
        List of MCQuestion in standardized format.
    """
    logger.info(f"Loading TruthfulQA MC ({num_samples} questions)...")

    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    dataset = dataset.shuffle(seed=_SEED)

    rng = random.Random(_SEED)
    choice_labels = ["A", "B", "C", "D"]
    questions: list[MCQuestion] = []

    for i, item in enumerate(dataset):
        if len(questions) >= num_samples:
            break

        correct_answers = cast(list[str], item["correct_answers"])
        incorrect_answers = cast(list[str], item["incorrect_answers"])

        if not correct_answers or not incorrect_answers:
            continue

        # Pick 1 correct answer
        correct_ans = rng.choice(correct_answers)

        # Pick up to 3 incorrect answers
        n_distractors = min(3, len(incorrect_answers))
        distractors = rng.sample(incorrect_answers, n_distractors)

        # If we don't have 3 distractors, pad with more correct answers (rare)
        while len(distractors) < 3 and len(correct_answers) > 1:
            extra = [a for a in correct_answers if a != correct_ans and a not in distractors]
            if extra:
                distractors.append(extra[0])
            else:
                break

        if len(distractors) < 3:
            continue  # Skip if we can't build 4 choices

        # Shuffle correct answer position
        all_options = [correct_ans] + distractors
        rng.shuffle(all_options)
        correct_idx = all_options.index(correct_ans)

        formatted_choices = [f"{choice_labels[j]}) {all_options[j]}" for j in range(4)]

        # Map TruthfulQA categories to our taxonomy
        cat = str(item.get("category", "")).lower()
        if any(
            c in cat
            for c in [
                "fiction",
                "myth",
                "superstition",
                "paranormal",
                "conspiracies",
                "misconceptions",
                "mandela",
            ]
        ):
            ext_diff = "misconception_trap"
        elif any(c in cat for c in ["science", "health", "history"]):
            ext_diff = "factual"
        else:
            ext_diff = "ambiguous"

        questions.append(
            MCQuestion(
                id=f"tqa_{i:04d}",
                question=str(item["question"]),
                choices=formatted_choices,
                correct_answer=choice_labels[correct_idx],
                category="truthfulness",
                source="truthfulqa",
                external_difficulty=ext_diff,
            )
        )

    logger.info(f"  Loaded {len(questions)} TruthfulQA MC questions")
    return questions


def _generate_gsm8k_distractors(
    correct: str,
    rng: random.Random,
) -> list[str]:
    """Generate plausible numeric distractors for a GSM8K answer.

    Strategy: apply small perturbations to the correct numeric answer
    to create plausible-looking wrong answers.
    """
    try:
        correct_num = float(correct.replace(",", ""))
    except ValueError:
        # Non-numeric answer, use simple string distractors
        return [f"{correct}0", f"{correct}1", f"{correct}2"]

    distractors_set: set[str] = set()
    perturbations: list[Callable[[float], float]] = [
        lambda x: x + rng.randint(1, 5),
        lambda x: x - rng.randint(1, 5),
        lambda x: x * 2,
        lambda x: x * 10,
        lambda x: x / 2,
        lambda x: x + rng.randint(10, 50),
        lambda x: x - rng.randint(10, 50),
        lambda x: abs(x - rng.randint(1, 10)),
    ]

    rng.shuffle(perturbations)

    for perturb in perturbations:
        if len(distractors_set) >= 3:
            break
        try:
            val = perturb(correct_num)
            # Format consistently with the original
            if correct_num == int(correct_num):
                val_str = str(int(val))
            else:
                val_str = f"{val:.2f}"

            if val_str != correct and val_str not in distractors_set:
                distractors_set.add(val_str)
        except (ZeroDivisionError, OverflowError):
            continue

    # Fallback: ensure we have exactly 3
    fallback_counter = 1
    while len(distractors_set) < 3:
        distractors_set.add(str(int(correct_num) + fallback_counter * 7))
        fallback_counter += 1

    return list(distractors_set)[:3]


def load_gsm8k_mc(
    num_samples: int = 100,
    split: str = "train",
) -> list[MCQuestion]:
    """Load GSM8K and convert to MC format by generating numeric distractors.

    Each question gets the correct numeric answer + 3 plausible distractors.
    External difficulty proxy: number of reasoning steps in the solution.

    Returns:
        List of MCQuestion in standardized format.
    """
    logger.info(f"Loading GSM8K as MC ({num_samples} questions)...")

    dataset = load_dataset("openai/gsm8k", "main", split=split)
    dataset = dataset.shuffle(seed=_SEED)

    rng = random.Random(_SEED)
    choice_labels = ["A", "B", "C", "D"]
    questions: list[MCQuestion] = []

    for i, item in enumerate(dataset):
        if len(questions) >= num_samples:
            break

        answer_text = str(item["answer"])
        gt_answer = answer_text.split("####")[-1].strip()

        # Count reasoning steps (lines before ####) as difficulty proxy
        reasoning_lines = answer_text.split("####")[0].strip().split("\n")
        n_steps = len([line for line in reasoning_lines if line.strip()])

        if n_steps <= 2:
            step_difficulty = "easy_1-2_steps"
        elif n_steps <= 4:
            step_difficulty = "medium_3-4_steps"
        else:
            step_difficulty = "hard_5+_steps"

        # Generate distractors
        distractors = _generate_gsm8k_distractors(gt_answer, rng)

        # Build MC
        all_options = [gt_answer] + distractors
        rng.shuffle(all_options)
        correct_idx = all_options.index(gt_answer)

        formatted_choices = [f"{choice_labels[j]}) {all_options[j]}" for j in range(4)]

        questions.append(
            MCQuestion(
                id=f"gsm8k_{i:04d}",
                question=str(item["question"]),
                choices=formatted_choices,
                correct_answer=choice_labels[correct_idx],
                category="math",
                source="gsm8k",
                external_difficulty=step_difficulty,
            )
        )

    logger.info(f"  Loaded {len(questions)} GSM8K MC questions")
    return questions


def load_math_level5(
    num_samples: int = 100,
) -> list[MCQuestion]:
    """Load MATH Level 5 questions (competition grade) as MC.

    Converts open-ended math problems to MC using numeric distractors.
    """
    logger.info(f"Loading MATH Level 5 ({num_samples} questions)...")

    dataset = load_dataset("HuggingFaceH4/MATH", split="test")
    dataset = dataset.shuffle(seed=_SEED)
    dataset = dataset.filter(lambda x: x["level"] == "Level 5")

    rng = random.Random(_SEED)
    choice_labels = ["A", "B", "C", "D"]
    questions: list[MCQuestion] = []

    for i, item in enumerate(dataset):
        if len(questions) >= num_samples:
            break

        correct_ans = str(item["solution"]).split("####")[-1].strip()
        # Clean up LaTeX or complex strings if possible, but keep numeric core

        distractors = _generate_gsm8k_distractors(correct_ans, rng)

        all_options = [correct_ans] + distractors
        rng.shuffle(all_options)
        correct_idx = all_options.index(correct_ans)

        formatted_choices = [f"{choice_labels[j]}) {all_options[j]}" for j in range(4)]

        questions.append(
            MCQuestion(
                id=f"math_l5_{i:04d}",
                question=str(item["problem"]),
                choices=formatted_choices,
                correct_answer=choice_labels[correct_idx],
                category="math_hard",
                source="math_level5",
                external_difficulty="competition_grade",
            )
        )

    logger.info(f"  Loaded {len(questions)} MATH L5 questions")
    return questions


def load_gpqa_mc(
    num_samples: int = 100,
) -> list[MCQuestion]:
    """Load GPQA (Diamond split) to intentionally challenge frontier models.

    GPQA is specifically designed to be extremely difficult even for PhDs.
    """
    logger.info(f"Loading GPQA Diamond ({num_samples} questions)...")

    # Using the standard huggingface gpqa dataset
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    dataset = dataset.shuffle(seed=_SEED)

    rng = random.Random(_SEED)
    choice_labels = ["A", "B", "C", "D"]
    questions: list[MCQuestion] = []

    for i, item in enumerate(dataset):
        if len(questions) >= num_samples:
            break

        correct_ans = str(item["Correct Answer"])
        distractors = [
            str(item["Incorrect Answer 1"]),
            str(item["Incorrect Answer 2"]),
            str(item["Incorrect Answer 3"]),
        ]

        # Build 4-choice list and shuffle
        all_options = [correct_ans] + distractors
        rng.shuffle(all_options)
        correct_idx = all_options.index(correct_ans)

        formatted_choices = [f"{choice_labels[j]}) {all_options[j]}" for j in range(4)]

        questions.append(
            MCQuestion(
                id=f"gpqa_{item['Record ID']}_{i:04d}",
                question=str(item["Question"]),
                choices=formatted_choices,
                correct_answer=choice_labels[correct_idx],
                category="science",
                source="gpqa_diamond",
                external_difficulty="phd_level",
            )
        )

    logger.info(f"  Loaded {len(questions)} GPQA questions")
    return questions


def _parse_mc_choices_from_prompt(prompt: str) -> list[str]:
    """Extract A-J style options from a prompt into standardized 'A) ...' format."""
    # Match labels like "A)", "B.", "C:" and slice text until next label.
    matches = list(re.finditer(r"(?<![A-Za-z0-9])([A-J])[\)\.\:]\s*", prompt))
    if len(matches) < 2:
        return []

    choices: list[str] = []
    for i, m in enumerate(matches):
        label = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt)
        text = " ".join(prompt[start:end].strip().split())
        if text:
            choices.append(f"{label}) {text}")
    return choices


def load_hle_mc_text_only(
    num_samples: int = 10,
    split: str = "test",
) -> list[MCQuestion]:
    """Load text-only multiple-choice subset from CAIS HLE."""
    logger.info(f"Loading HLE text-only MC ({num_samples} questions)...")

    dataset = load_dataset("cais/hle", split=split).shuffle(seed=_SEED)
    questions: list[MCQuestion] = []

    for i, item in enumerate(dataset):
        if len(questions) >= num_samples:
            break

        # Only keep text-only + MC questions
        img = str(item.get("image", ""))
        if img.startswith("data:image"):
            continue
        if str(item.get("answer_type")) != "multipleChoice":
            continue

        question_text = str(item["question"])
        choices = _parse_mc_choices_from_prompt(question_text)
        if len(choices) < 4:
            continue

        ans = str(item.get("answer", "")).strip().upper()
        if not ans or ans[0] not in "ABCDEFGHIJ":
            continue

        questions.append(
            MCQuestion(
                id=f"hle_mc_{item['id']}_{i:04d}",
                question=question_text,
                choices=choices,
                correct_answer=ans[0],
                category="hle_mc",
                source="hle_mc",
                external_difficulty=str(item.get("category", "unknown")),
            )
        )

    logger.info(f"  Loaded {len(questions)} HLE text-only MC questions")
    return questions


def load_supergpqa_mc(
    num_samples: int = 10,
    split: str = "train",
) -> list[MCQuestion]:
    """Load SuperGPQA as standardized MC format."""
    logger.info(f"Loading SuperGPQA ({num_samples} questions)...")

    dataset = load_dataset("m-a-p/SuperGPQA", split=split).shuffle(seed=_SEED)
    questions: list[MCQuestion] = []

    for i, item in enumerate(dataset):
        if len(questions) >= num_samples:
            break

        options = cast(list[str], item.get("options", []))
        if len(options) < 4:
            continue
        if len(options) > 10:
            options = options[:10]

        labels = [chr(ord("A") + j) for j in range(len(options))]
        formatted = [f"{labels[j]}) {options[j]}" for j in range(len(options))]

        ans = str(item.get("answer_letter", "")).strip().upper()
        if ans not in labels:
            continue

        questions.append(
            MCQuestion(
                id=f"supergpqa_{item['uuid']}_{i:04d}",
                question=str(item["question"]),
                choices=formatted,
                correct_answer=ans,
                category="supergpqa",
                source="supergpqa",
                external_difficulty=str(item.get("difficulty", "unknown")),
            )
        )

    logger.info(f"  Loaded {len(questions)} SuperGPQA questions")
    return questions


def load_simplebench_mc(
    num_samples: int = 10,
    url: str = "https://raw.githubusercontent.com/simple-bench/SimpleBench/main/simple_bench_public.json",
) -> list[MCQuestion]:
    """Load SimpleBench public set and parse MC options from prompt text."""
    logger.info(f"Loading SimpleBench public ({num_samples} questions)...")

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    eval_data = payload.get("eval_data", [])
    rng = random.Random(_SEED)
    rng.shuffle(eval_data)

    questions: list[MCQuestion] = []
    for i, item in enumerate(eval_data):
        if len(questions) >= num_samples:
            break

        prompt = str(item.get("prompt", ""))
        choices = _parse_mc_choices_from_prompt(prompt)
        if len(choices) < 4:
            continue

        ans = str(item.get("answer", "")).strip().upper()
        if not ans or ans[0] not in "ABCDEFGHIJ":
            continue

        questions.append(
            MCQuestion(
                id=f"simplebench_{item.get('question_id', i)}_{i:04d}",
                question=prompt,
                choices=choices,
                correct_answer=ans[0],
                category="simplebench",
                source="simplebench",
                external_difficulty="public",
            )
        )

    logger.info(f"  Loaded {len(questions)} SimpleBench questions")
    return questions


def load_harp_mcq(
    num_samples: int = 10,
    url: str = "https://github.com/aadityasingh/HARP/raw/refs/heads/main/HARP_mcq.jsonl.zip",
) -> list[MCQuestion]:
    """Load HARP MCQ benchmark from official zip and normalize to MCQuestion."""
    logger.info(f"Loading HARP_mcq ({num_samples} questions)...")

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    name = "HARP_mcq.jsonl" if "HARP_mcq.jsonl" in zf.namelist() else zf.namelist()[0]

    rows: list[dict[str, object]] = []
    with zf.open(name) as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))

    rng = random.Random(_SEED)
    rng.shuffle(rows)

    questions: list[MCQuestion] = []
    for i, item in enumerate(rows):
        if len(questions) >= num_samples:
            break

        problem = str(item.get("problem", "")).strip()
        choices_raw = cast(dict[str, str], item.get("choices", {}))
        answer_choice = str(item.get("answer_choice", "")).strip().upper()
        if not problem or not choices_raw or answer_choice not in choices_raw:
            continue

        labels = sorted([k for k in choices_raw.keys() if len(k) == 1 and k.isalpha()])
        if len(labels) < 4:
            continue

        formatted = [f"{lab}) {str(choices_raw[lab]).strip()}" for lab in labels]
        difficulty = f"{item.get('contest', 'unknown')}_{item.get('level', 'unknown')}"

        questions.append(
            MCQuestion(
                id=(
                    f"harp_mcq_{item.get('year', 'na')}_{item.get('contest', 'na')}_"
                    f"{item.get('number', i)}_{i:04d}"
                ),
                question=problem,
                choices=formatted,
                correct_answer=answer_choice,
                category="math_hard",
                source="harp_mcq",
                external_difficulty=difficulty,
            )
        )

    logger.info(f"  Loaded {len(questions)} HARP_mcq questions")
    return questions


def load_musr(
    num_samples: int = 30,
    splits: Optional[list[str]] = None,
) -> list[MCQuestion]:
    """Load MuSR multiple-choice tasks across its three official splits."""
    logger.info(f"Loading MuSR ({num_samples} questions)...")

    ds = load_dataset("TAUR-Lab/MuSR")
    selected_splits = splits or ["murder_mysteries", "object_placements", "team_allocation"]
    choice_labels = ["A", "B", "C", "D", "E", "F"]
    rows: list[tuple[str, dict[str, object]]] = []
    for split in selected_splits:
        if split not in ds:
            continue
        for row in ds[split]:
            rows.append((split, row))
    random.Random(_SEED).shuffle(rows)

    questions: list[MCQuestion] = []
    for i, (split, item) in enumerate(rows):
        if len(questions) >= num_samples:
            break
        try:
            options = ast.literal_eval(str(item["choices"]))
        except Exception:
            continue
        if not isinstance(options, list) or len(options) < 2:
            continue
        labels = choice_labels[: len(options)]
        formatted = [f"{labels[j]}) {str(options[j]).strip()}" for j in range(len(options))]
        answer_idx = int(cast(int | str, item["answer_index"]))
        if answer_idx < 0 or answer_idx >= len(options):
            continue
        questions.append(
            MCQuestion(
                id=f"musr_{split}_{i:04d}",
                question=f"{str(item['narrative']).strip()}\n\n{str(item['question']).strip()}",
                choices=formatted,
                correct_answer=labels[answer_idx],
                reference_answer=str(item.get("answer_choice", "")),
                category="reasoning",
                source="musr",
                answer_format="multiple_choice",
                evaluator_name="musr",
                external_difficulty=split,
                metadata={"split": split},
            )
        )

    logger.info(f"  Loaded {len(questions)} MuSR questions")
    return questions


def load_olympiadbench(
    num_samples: int = 30,
    config_name: str = "OE_TO_maths_en_COMP",
    split: str = "train",
) -> list[MCQuestion]:
    """Load OlympiadBench text-only open-ended math questions."""
    logger.info(f"Loading OlympiadBench ({num_samples} questions, config={config_name})...")

    dataset = load_dataset("Hothan/OlympiadBench", config_name, split=split).shuffle(seed=_SEED)
    questions: list[MCQuestion] = []

    for i, item in enumerate(dataset):
        if len(questions) >= num_samples:
            break
        if str(item.get("modality", "")).lower() != "text-only":
            continue
        if str(item.get("question_type", "")).lower() != "open-ended":
            continue
        final_answers = item.get("final_answer", [])
        if not final_answers:
            continue
        reference_answer = str(final_answers[0]).strip()
        prompt = str(item["question"]).strip()
        context = item.get("context")
        if context:
            prompt = f"{prompt}\n\nContext:\n{str(context).strip()}"
        questions.append(
            MCQuestion(
                id=f"olympiadbench_{item['id']}_{i:04d}",
                question=prompt,
                choices=[],
                correct_answer="",
                reference_answer=reference_answer,
                category="math_hard",
                source="olympiadbench",
                answer_format="open_ended",
                evaluator_name="olympiadbench",
                external_difficulty=str(item.get("difficulty", "unknown")),
                metadata={
                    "answer_type": item.get("answer_type"),
                    "is_multiple_answer": item.get("is_multiple_answer"),
                    "unit": item.get("unit"),
                    "subject": item.get("subject"),
                    "subfield": item.get("subfield"),
                    "language": item.get("language"),
                },
            )
        )

    logger.info(f"  Loaded {len(questions)} OlympiadBench questions")
    return questions


def load_livebench_reasoning(
    num_samples: int = 30,
    split: str = "test",
) -> list[MCQuestion]:
    """Load LiveBench reasoning tasks as open-ended static questions."""
    logger.info(f"Loading LiveBench reasoning ({num_samples} questions)...")

    dataset = load_dataset("livebench/reasoning", split=split).shuffle(seed=_SEED)
    questions: list[MCQuestion] = []
    for i, item in enumerate(dataset):
        if len(questions) >= num_samples:
            break
        turns = cast(list[str], item.get("turns", []))
        if not turns:
            continue
        questions.append(
            MCQuestion(
                id=f"livebench_reasoning_{item['question_id']}_{i:04d}",
                question=str(turns[0]).strip(),
                choices=[],
                correct_answer="",
                reference_answer=str(item.get("ground_truth", "")).strip(),
                category="reasoning",
                source="livebench_reasoning",
                answer_format="open_ended",
                evaluator_name="livebench_reasoning",
                external_difficulty=str(item.get("level", "unknown")),
                metadata={
                    "task": item.get("task"),
                    "turns": turns,
                    "question_id": item.get("question_id"),
                    "category": item.get("category"),
                    "livebench_release_date": str(item.get("livebench_release_date", ""))[:10],
                },
            )
        )

    logger.info(f"  Loaded {len(questions)} LiveBench reasoning questions")
    return questions


def load_aime_2024(
    num_samples: int = 30,
    split: str = "train",
) -> list[MCQuestion]:
    """Load AIME 2024 as open-ended integer-answer tasks."""
    logger.info(f"Loading AIME 2024 ({num_samples} questions)...")

    dataset = load_dataset("HuggingFaceH4/aime_2024", split=split).shuffle(seed=_SEED)
    questions: list[MCQuestion] = []
    for i, item in enumerate(dataset):
        if len(questions) >= num_samples:
            break
        problem = str(
            item.get("problem") or item.get("question") or item.get("prompt") or ""
        ).strip()
        answer = str(item.get("answer") or item.get("final_answer") or "").strip()
        if not problem or not answer:
            continue
        questions.append(
            MCQuestion(
                id=f"aime_2024_{i:04d}",
                question=problem,
                choices=[],
                correct_answer="",
                reference_answer=answer,
                category="math_competition",
                source="aime_2024",
                answer_format="open_ended",
                evaluator_name="math_exact",
                external_difficulty="aime_2024",
                metadata={"split": split},
            )
        )

    logger.info(f"  Loaded {len(questions)} AIME 2024 questions")
    return questions


def load_math500(
    num_samples: int = 100,
    split: str = "test",
) -> list[MCQuestion]:
    """Load MATH-500 as open-ended math tasks."""
    logger.info(f"Loading MATH-500 ({num_samples} questions)...")

    dataset = load_dataset("math-ai/math500", split=split).shuffle(seed=_SEED)
    questions: list[MCQuestion] = []
    for i, item in enumerate(dataset):
        if len(questions) >= num_samples:
            break
        problem = str(item.get("problem") or item.get("question") or "").strip()
        answer = str(item.get("answer") or item.get("final_answer") or "").strip()
        if not problem or not answer:
            continue
        questions.append(
            MCQuestion(
                id=f"math500_{item.get('unique_id', i)}_{i:04d}",
                question=problem,
                choices=[],
                correct_answer="",
                reference_answer=answer,
                category="math_hard",
                source="math500",
                answer_format="open_ended",
                evaluator_name="math_exact",
                external_difficulty=str(item.get("level", "unknown")),
                metadata={
                    "split": split,
                    "subject": item.get("subject"),
                    "level": item.get("level"),
                    "unique_id": item.get("unique_id"),
                },
            )
        )

    logger.info(f"  Loaded {len(questions)} MATH-500 questions")
    return questions


def load_gpqa_diamond(
    num_samples: int = 100,
    split: str = "train",
) -> list[MCQuestion]:
    """Load GPQA Diamond as standardized 4-choice MC tasks."""
    logger.info(f"Loading GPQA Diamond ({num_samples} questions)...")

    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split=split).shuffle(seed=_SEED)
    rng = random.Random(_SEED)
    choice_labels = ["A", "B", "C", "D"]
    questions: list[MCQuestion] = []

    for i, item in enumerate(dataset):
        if len(questions) >= num_samples:
            break
        correct_ans = str(item.get("Correct Answer", "")).strip()
        distractors = [
            str(item.get("Incorrect Answer 1", "")).strip(),
            str(item.get("Incorrect Answer 2", "")).strip(),
            str(item.get("Incorrect Answer 3", "")).strip(),
        ]
        question = str(item.get("Question", "")).strip()
        if not question or not correct_ans or any(not d for d in distractors):
            continue

        all_options = [correct_ans] + distractors
        rng.shuffle(all_options)
        correct_idx = all_options.index(correct_ans)
        formatted_choices = [f"{choice_labels[j]}) {all_options[j]}" for j in range(4)]

        questions.append(
            MCQuestion(
                id=f"gpqa_diamond_{item.get('Record ID', i)}_{i:04d}",
                question=question,
                choices=formatted_choices,
                correct_answer=choice_labels[correct_idx],
                reference_answer=correct_ans,
                category="science_hard",
                source="gpqa_diamond",
                answer_format="multiple_choice",
                evaluator_name="mc_letter",
                external_difficulty="gpqa_diamond",
                metadata={
                    "split": split,
                    "record_id": item.get("Record ID"),
                    "subcategory": item.get("Subdomain"),
                },
            )
        )

    logger.info(f"  Loaded {len(questions)} GPQA Diamond questions")
    return questions


def load_scale_experiment_dataset(
    num_per_source: int = 100,
    counts: Optional[dict[str, int]] = None,
) -> list[MCQuestion]:
    """Load the complete scale experiment dataset focusing on ULTRA-HARD tasks.

    Composition:
    - MMLU-Pro: Hard reasoning general knowledge
    - GPQA Diamond: Science questions where non-experts score 0%
    - MATH Level 5: Competition-grade mathematical reasoning
    - Hard MMLU: Specialized STEM subjects
    """
    logger.info("Loading ULTRA-HARD scale experiment dataset (2026 Frontier Edition)...")

    # Allow explicit per-dataset counts for pilot/final runs.
    # If not provided, preserve legacy behavior.
    target = {
        "mmlu_pro": num_per_source,
        "gpqa": num_per_source,
        "math_l5": num_per_source // 2,
        "mmlu_hard": num_per_source // 2,
        "hle_mc": 0,
        "supergpqa": 0,
        "simplebench": 0,
        "truthfulqa_mc": 0,
        "harp_mcq": 0,
        "musr": 0,
        "olympiadbench": 0,
        "livebench": 0,
    }
    if counts:
        target.update({k: max(0, int(v)) for k, v in counts.items() if k in target})

    # If new frontier text-MC sources are requested, prioritize that composition.
    if (
        target["hle_mc"] > 0
        or target["supergpqa"] > 0
        or target["simplebench"] > 0
        or target["truthfulqa_mc"] > 0
        or target["harp_mcq"] > 0
        or target["musr"] > 0
        or target["olympiadbench"] > 0
        or target["livebench"] > 0
    ):
        hle_qs = load_hle_mc_text_only(num_samples=target["hle_mc"])
        super_qs = load_supergpqa_mc(num_samples=target["supergpqa"])
        simple_qs = load_simplebench_mc(num_samples=target["simplebench"])
        truthful_qs = load_truthfulqa_mc(num_samples=target["truthfulqa_mc"])
        harp_qs = load_harp_mcq(num_samples=target["harp_mcq"])
        musr_qs = load_musr(num_samples=target["musr"])
        olympiad_qs = load_olympiadbench(num_samples=target["olympiadbench"])
        livebench_qs = load_livebench_reasoning(num_samples=target["livebench"])
        combined = (
            hle_qs
            + super_qs
            + simple_qs
            + truthful_qs
            + harp_qs
            + musr_qs
            + olympiad_qs
            + livebench_qs
        )
        random.Random(_SEED).shuffle(combined)
        logger.info(
            f"Frontier text-MC dataset ready: {len(combined)} total questions "
            f"(HLE-MC={len(hle_qs)}, SuperGPQA={len(super_qs)}, "
            f"SimpleBench={len(simple_qs)}, TruthfulQA-MC={len(truthful_qs)}, "
            f"HARP-MCQ={len(harp_qs)}, MuSR={len(musr_qs)}, "
            f"OlympiadBench={len(olympiad_qs)}, LiveBench={len(livebench_qs)})"
        )
        return combined

    # 1. Load MMLU-Pro
    pro_qs = load_mmlu_pro(num_samples=target["mmlu_pro"])

    # 2. Load GPQA Diamond (Filtered for 0% NEV Accuracy only)
    logger.info("Filtering GPQA for 0% Non-Expert Accuracy...")
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    dataset = dataset.shuffle(seed=_SEED)
    dataset = dataset.filter(lambda x: x["Non-Expert Validator Accuracy"] == 0.0)

    rng = random.Random(_SEED)
    choice_labels = ["A", "B", "C", "D"]
    gpqa_qs: list[MCQuestion] = []

    for i, item in enumerate(dataset):
        if len(gpqa_qs) >= target["gpqa"]:
            break

        correct_ans = str(item["Correct Answer"])
        distractors = [
            str(item["Incorrect Answer 1"]),
            str(item["Incorrect Answer 2"]),
            str(item["Incorrect Answer 3"]),
        ]

        all_options = [correct_ans] + distractors
        rng.shuffle(all_options)
        correct_idx = all_options.index(correct_ans)

        formatted_choices = [f"{choice_labels[j]}) {all_options[j]}" for j in range(4)]

        gpqa_qs.append(
            MCQuestion(
                id=f"gpqa_hard_{item['Record ID']}_{i:04d}",
                question=str(item["Question"]),
                choices=formatted_choices,
                correct_answer=choice_labels[correct_idx],
                category="science",
                source="gpqa_diamond",
                external_difficulty="high_blind_spot",
            )
        )

    # 3. Load MATH Level 5
    math_qs = load_math_level5(num_samples=target["math_l5"])

    # 4. Load Hard MMLU
    mmlu_qs = load_mmlu(subjects=HARD_MMLU_SUBJECTS, num_samples=target["mmlu_hard"])

    combined = pro_qs + gpqa_qs + math_qs + mmlu_qs
    random.Random(_SEED).shuffle(combined)

    logger.info(
        f"Ultra-Hard dataset ready: {len(combined)} total questions "
        f"(MMLU-Pro={len(pro_qs)}, GPQA-ZeroAcc={len(gpqa_qs)}, "
        f"MATH-L5={len(math_qs)}, MMLU-Hard={len(mmlu_qs)})"
    )
    return combined
