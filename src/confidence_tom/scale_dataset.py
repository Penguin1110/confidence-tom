"""Dataset loaders for the scale experiment.

All benchmarks are standardized to multiple-choice format.
This ensures consistent parsing and fair confidence comparison across tasks.

Supported benchmarks:
- MMLU: general knowledge QA (4-choice MC)
- ARC-Challenge: science reasoning (4-choice MC)
- TruthfulQA: truthfulness / misconceptions (MC version)
- GSM8K: math reasoning (converted to MC with distractors)
"""

import logging
import random
from typing import Optional, cast

from datasets import load_dataset

from confidence_tom.dataset_models import MCQuestion

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
    perturbations = [
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


def load_scale_experiment_dataset(
    num_per_source: int = 100,
) -> list[MCQuestion]:
    """Load the complete scale experiment dataset focusing on HARD tasks.

    Returns a balanced dataset from hard sources (e.g., GPQA, hard MMLU),
    all in standardized MC format. This guarantees that model accuracy gets crushed,
    preventing ceiling effects for models like Qwen 397B.
    """
    logger.info(f"Loading HARD scale experiment dataset (target {num_per_source} per hard source)...")

    # Load hard MMLU
    mmlu_qs = load_mmlu(subjects=HARD_MMLU_SUBJECTS, num_samples=num_per_source)
    # Load GPQA Diamond
    gpqa_qs = load_gpqa_mc(num_samples=num_per_source)
    # We can still add some ARC/GSM but they might be too easy, keeping a small batch
    arc_qs = load_arc_challenge(num_samples=num_per_source // 2)
    tqa_qs = load_truthfulqa_mc(num_samples=num_per_source // 2)

    combined = mmlu_qs + gpqa_qs + arc_qs + tqa_qs
    random.Random(_SEED).shuffle(combined)

    logger.info(
        f"Hard datasets ready: {len(combined)} total questions "
        f"(GPQA={len(gpqa_qs)}, hard_MMLU={len(mmlu_qs)}, ARC={len(arc_qs)}, TQA={len(tqa_qs)})"
    )
    return combined
