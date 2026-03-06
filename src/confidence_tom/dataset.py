import logging
import random
from typing import Dict, List, cast

from datasets import load_dataset

logger = logging.getLogger(__name__)


# ============================================================================
# BBH (BIG-Bench Hard) Dataset Loaders
# ============================================================================

def load_bbh_navigate(num_samples: int = 100) -> List[Dict[str, str]]:
    """
    Loads BBH Navigate task - spatial reasoning with directional instructions.
    
    This task tests if a model can follow a sequence of navigation instructions
    and determine the final position/orientation.
    """
    dataset = load_dataset("lukaemon/bbh", "navigate", split="test")
    samples = list(dataset.shuffle(seed=42))[:min(num_samples, len(dataset))]

    formatted = []
    for i, item in enumerate(samples):
        formatted.append(
            {
                "id": f"bbh_navigate_{i}",
                "question": str(item["input"]),
                "ground_truth": str(item["target"]).strip().lower(),
                "ambiguity_level": "L1 (Deterministic reasoning)",
                "task_type": "navigate",
            }
        )
    logger.info(f"Loaded {len(formatted)} BBH Navigate questions")
    return formatted


def load_bbh_formal_fallacies(num_samples: int = 100) -> List[Dict[str, str]]:
    """
    Loads BBH Formal Fallacies task - logical deduction with potential traps.
    
    This task tests if a model can identify whether an argument is valid or
    contains a logical fallacy. High potential for "correct answer, wrong reasoning".
    """
    dataset = load_dataset("lukaemon/bbh", "formal_fallacies", split="test")
    samples = list(dataset.shuffle(seed=42))[:min(num_samples, len(dataset))]

    formatted = []
    for i, item in enumerate(samples):
        formatted.append(
            {
                "id": f"bbh_formal_fallacies_{i}",
                "question": str(item["input"]),
                "ground_truth": str(item["target"]).strip().lower(),
                "ambiguity_level": "L2 (Logical trap-prone)",
                "task_type": "formal_fallacies",
            }
        )
    logger.info(f"Loaded {len(formatted)} BBH Formal Fallacies questions")
    return formatted


def load_bbh_mixed(num_per_task: int = 50) -> List[Dict[str, str]]:
    """Loads a balanced mix of BBH Navigate and Formal Fallacies tasks."""
    logger.info(f"Loading {num_per_task} questions per BBH task...")
    navigate_qs = load_bbh_navigate(num_samples=num_per_task)
    fallacies_qs = load_bbh_formal_fallacies(num_samples=num_per_task)

    combined = navigate_qs + fallacies_qs
    random.shuffle(combined)
    return combined


# ============================================================================
# Hard Dataset Loaders (GPQA, LogiQA, ReClor, MATH)
# ============================================================================


def load_gpqa_diamond(num_samples: int = 30) -> List[Dict[str, str]]:
    """
    Loads GPQA Diamond - Expert-level science reasoning.
    
    This dataset contains extremely difficult questions requiring PhD-level
    knowledge in physics, chemistry, and biology. Small models often generate
    confident but completely wrong answers with sophisticated-sounding jargon.
    
    Perfect for testing if Observer can see through professional terminology fallacies.
    
    Note: If GPQA is not accessible, falls back to MMLU professional subjects.
    """
    try:
        # Try GPQA first (requires HF authentication for gated dataset)
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    except Exception as e:
        logger.warning(f"GPQA not accessible (gated dataset): {e}")
        logger.info("Falling back to MMLU professional subjects (similar difficulty)")
        return load_mmlu_hard(num_samples=num_samples)
    
    samples = list(dataset.shuffle(seed=42))[:min(num_samples, len(dataset))]

    formatted = []
    for i, item in enumerate(samples):
        # GPQA format: Question with multiple choice options
        question_text = str(item.get("Question", item.get("question", "")))
        
        # Build multiple choice format
        choices = []
        for key in ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]:
            if key in item and item[key]:
                choices.append(str(item[key]))
        
        # Shuffle choices to avoid position bias
        correct_answer = str(item.get("Correct Answer", ""))
        random.shuffle(choices)
        correct_idx = choices.index(correct_answer) if correct_answer in choices else 0
        
        # Format as A, B, C, D
        choice_labels = ["A", "B", "C", "D"]
        formatted_choices = "\n".join([f"{choice_labels[j]}. {c}" for j, c in enumerate(choices[:4])])
        full_question = f"{question_text}\n\n{formatted_choices}"
        
        formatted.append(
            {
                "id": f"gpqa_diamond_{i}",
                "question": full_question,
                "ground_truth": choice_labels[correct_idx],
                "ambiguity_level": "L5 (Expert-level reasoning)",
                "task_type": "gpqa_diamond",
                "domain": str(item.get("Subdomain", item.get("subdomain", "science"))),
            }
        )
    logger.info(f"Loaded {len(formatted)} GPQA Diamond questions")
    return formatted


def load_mmlu_hard(num_samples: int = 30) -> List[Dict[str, str]]:
    """
    Loads MMLU professional/graduate-level subjects as GPQA fallback.
    
    Selects from the hardest MMLU categories:
    - college_physics, college_chemistry, college_biology
    - professional_medicine, professional_law
    - abstract_algebra, formal_logic
    """
    hard_subjects = [
        "college_physics",
        "college_chemistry",
        "college_biology",
        "professional_medicine", 
        "abstract_algebra",
        "formal_logic",
        "professional_law",
    ]
    
    all_questions = []
    samples_per_subject = max(1, num_samples // len(hard_subjects))
    
    for subject in hard_subjects:
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test")
            samples = list(dataset.shuffle(seed=42))[:samples_per_subject]
            
            for i, item in enumerate(samples):
                question = str(item.get("question", ""))
                choices = item.get("choices", [])
                answer_idx = int(item.get("answer", 0))
                
                choice_labels = ["A", "B", "C", "D"]
                formatted_choices = "\n".join([f"{choice_labels[j]}. {c}" for j, c in enumerate(choices[:4])])
                full_question = f"{question}\n\n{formatted_choices}"
                
                all_questions.append({
                    "id": f"mmlu_{subject}_{i}",
                    "question": full_question,
                    "ground_truth": choice_labels[answer_idx] if answer_idx < 4 else "A",
                    "ambiguity_level": "L5 (Expert-level reasoning)",
                    "task_type": f"mmlu_{subject}",
                    "domain": subject,
                })
        except Exception as e:
            logger.warning(f"Failed to load MMLU {subject}: {e}")
            continue
    
    random.shuffle(all_questions)
    logger.info(f"Loaded {len(all_questions)} MMLU hard questions (GPQA fallback)")
    return all_questions[:num_samples]


def load_logiqa(num_samples: int = 30) -> List[Dict[str, str]]:
    """
    Loads LogiQA-style logical reasoning questions.
    
    Falls back to ARC-Challenge (complex science reasoning) if LogiQA is unavailable.
    ARC-Challenge contains questions that require multi-step reasoning and 
    are designed to be difficult for AI systems.
    """
    # Use ARC-Challenge as a reliable alternative for complex reasoning
    try:
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    except Exception as e:
        logger.warning(f"Failed to load ARC-Challenge: {e}")
        # Fallback to MMLU formal_logic
        return load_mmlu_logic(num_samples=num_samples)
    
    samples = list(dataset.shuffle(seed=42))[:min(num_samples, len(dataset))]

    formatted = []
    for i, item in enumerate(samples):
        question = str(item.get("question", ""))
        choices = item.get("choices", {})
        answer_key = str(item.get("answerKey", "A"))
        
        # ARC format: choices is a dict with "text" and "label" lists
        choice_texts = choices.get("text", [])
        choice_labels = choices.get("label", ["A", "B", "C", "D"])
        
        formatted_choices = "\n".join([f"{label}. {text}" for label, text in zip(choice_labels, choice_texts)])
        full_question = f"{question}\n\n{formatted_choices}"
        
        formatted.append(
            {
                "id": f"arc_challenge_{i}",
                "question": full_question,
                "ground_truth": answer_key,
                "ambiguity_level": "L4 (Complex reasoning trap)",
                "task_type": "arc_challenge",
            }
        )
    logger.info(f"Loaded {len(formatted)} ARC-Challenge questions (LogiQA alternative)")
    return formatted


def load_mmlu_logic(num_samples: int = 30) -> List[Dict[str, str]]:
    """Loads MMLU logical_fallacies and formal_logic as fallback."""
    subjects = ["logical_fallacies", "formal_logic"]
    all_questions = []
    
    for subject in subjects:
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test")
            samples = list(dataset.shuffle(seed=42))[:num_samples // 2]
            
            for i, item in enumerate(samples):
                question = str(item.get("question", ""))
                choices = item.get("choices", [])
                answer_idx = int(item.get("answer", 0))
                
                choice_labels = ["A", "B", "C", "D"]
                formatted_choices = "\n".join([f"{choice_labels[j]}. {c}" for j, c in enumerate(choices[:4])])
                full_question = f"{question}\n\n{formatted_choices}"
                
                all_questions.append({
                    "id": f"mmlu_{subject}_{i}",
                    "question": full_question,
                    "ground_truth": choice_labels[answer_idx] if answer_idx < 4 else "A",
                    "ambiguity_level": "L4 (Complex logical trap)",
                    "task_type": f"mmlu_{subject}",
                })
        except Exception as e:
            logger.warning(f"Failed to load MMLU {subject}: {e}")
    
    logger.info(f"Loaded {len(all_questions)} MMLU logic questions")
    return all_questions[:num_samples]


def load_reclor(num_samples: int = 30) -> List[Dict[str, str]]:
    """
    Loads ReClor - Reading Comprehension with Logical Reasoning.
    
    Graduate-level logical reasoning dataset from standardized tests like LSAT/GMAT.
    Features extremely tricky distractor options that require deep logical analysis.
    """
    try:
        dataset = load_dataset("metaeval/reclor", split="validation")
    except Exception:
        # Fallback: ReClor may need authentication or alternative source
        logger.warning("ReClor dataset requires special access. Using LogiQA as fallback.")
        return load_logiqa(num_samples)
    
    samples = list(dataset.shuffle(seed=42))[:min(num_samples, len(dataset))]

    formatted = []
    for i, item in enumerate(samples):
        context = str(item.get("context", ""))
        question = str(item.get("question", ""))
        answers = item.get("answers", [])
        label = int(item.get("label", 0))
        
        choice_labels = ["A", "B", "C", "D"]
        formatted_choices = "\n".join([f"{choice_labels[j]}. {ans}" for j, ans in enumerate(answers[:4])])
        full_question = f"{context}\n\nQuestion: {question}\n\n{formatted_choices}"
        
        formatted.append(
            {
                "id": f"reclor_{i}",
                "question": full_question,
                "ground_truth": choice_labels[label] if label < len(choice_labels) else "A",
                "ambiguity_level": "L4 (Graduate-level logical trap)",
                "task_type": "reclor",
            }
        )
    logger.info(f"Loaded {len(formatted)} ReClor questions")
    return formatted


def load_math_hard(num_samples: int = 30, min_level: int = 4) -> List[Dict[str, str]]:
    """
    Loads MATH dataset (Level 4-5) - Multi-step mathematical reasoning.
    
    These are competition-level math problems requiring long chains of reasoning.
    Small models often experience "computational fatigue" and make subtle errors
    in the middle of long derivations - perfect for testing Observer's ability
    to spot "the moment of collapse" in extended reasoning chains.
    
    Falls back to MMLU college math if MATH dataset is unavailable.
    """
    import re
    
    try:
        dataset = load_dataset("lighteval/MATH", "all", split="test")
        
        # Filter to only Level 4 and Level 5 (hardest problems)
        hard_problems = [item for item in dataset if int(str(item.get("level", "0")).replace("Level ", "")) >= min_level]
        samples = list(random.Random(42).sample(hard_problems, min(num_samples, len(hard_problems))))
    except Exception as e:
        logger.warning(f"MATH dataset not available: {e}")
        logger.info("Falling back to MMLU college mathematics")
        return load_mmlu_math(num_samples=num_samples)

    formatted = []
    for i, item in enumerate(samples):
        problem = str(item.get("problem", ""))
        solution = str(item.get("solution", ""))
        level = str(item.get("level", "Level 5"))
        subject = str(item.get("type", "math"))
        
        # Extract final answer from solution (usually in \boxed{})
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', solution)
        if boxed_match:
            ground_truth = boxed_match.group(1)
        else:
            # Fallback: use last line or full solution
            ground_truth = solution.split('\n')[-1].strip()
        
        formatted.append(
            {
                "id": f"math_{level.replace(' ', '').lower()}_{i}",
                "question": problem,
                "ground_truth": ground_truth,
                "ambiguity_level": f"L5 ({level} - Multi-step math)",
                "task_type": f"math_{subject}",
                "full_solution": solution,  # Keep for reference
            }
        )
    logger.info(f"Loaded {len(formatted)} MATH (Level {min_level}+) questions")
    return formatted


def load_mmlu_math(num_samples: int = 30) -> List[Dict[str, str]]:
    """Loads MMLU college-level math subjects as MATH fallback."""
    subjects = ["college_mathematics", "high_school_mathematics", "abstract_algebra"]
    all_questions = []
    samples_per_subject = max(1, num_samples // len(subjects))
    
    for subject in subjects:
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test")
            samples = list(dataset.shuffle(seed=42))[:samples_per_subject]
            
            for i, item in enumerate(samples):
                question = str(item.get("question", ""))
                choices = item.get("choices", [])
                answer_idx = int(item.get("answer", 0))
                
                choice_labels = ["A", "B", "C", "D"]
                formatted_choices = "\n".join([f"{choice_labels[j]}. {c}" for j, c in enumerate(choices[:4])])
                full_question = f"{question}\n\n{formatted_choices}"
                
                all_questions.append({
                    "id": f"mmlu_{subject}_{i}",
                    "question": full_question,
                    "ground_truth": choice_labels[answer_idx] if answer_idx < 4 else "A",
                    "ambiguity_level": "L5 (College-level math)",
                    "task_type": f"mmlu_{subject}",
                })
        except Exception as e:
            logger.warning(f"Failed to load MMLU {subject}: {e}")
    
    logger.info(f"Loaded {len(all_questions)} MMLU math questions")
    return all_questions[:num_samples]


def load_hard_dataset_mix(num_per_task: int = 30) -> List[Dict[str, str]]:
    """
    Loads a balanced mix of hard datasets for challenging small models.
    
    Includes:
    - GPQA Diamond: Expert-level science (tests jargon-filled nonsense detection)
    - LogiQA: Complex logical traps (tests reasoning flaw detection)
    - MATH Level 4-5: Long-chain math (tests computational fatigue detection)
    """
    logger.info(f"Loading hard dataset mix with {num_per_task} questions per task...")
    
    gpqa_qs = load_gpqa_diamond(num_samples=num_per_task)
    logiqa_qs = load_logiqa(num_samples=num_per_task)
    math_qs = load_math_hard(num_samples=num_per_task)
    
    combined = gpqa_qs + logiqa_qs + math_qs
    random.shuffle(combined)
    
    logger.info(f"Total hard questions loaded: {len(combined)}")
    return combined


def load_gpqa_math_mix(num_per_task: int = 30) -> List[Dict[str, str]]:
    """
    Loads a mix of MMLU Hard (science) + MMLU Math for multi-round experiments.
    
    All questions are multiple-choice (A/B/C/D) format for consistent answer comparison.
    
    This combination is designed to test:
    - Expert-level scientific reasoning (MMLU Hard) - small models produce jargon-filled nonsense
    - College-level mathematical reasoning (MMLU Math) - computational errors
    
    Total: num_per_task * 2 questions (half science, half math)
    """
    logger.info(f"Loading MMLU Hard + MMLU Math mix with {num_per_task} questions per task...")
    
    # Use MMLU variants (all multiple-choice ABCD format)
    science_qs = load_mmlu_hard(num_samples=num_per_task)
    math_qs = load_mmlu_math(num_samples=num_per_task)
    
    combined = science_qs + math_qs
    random.shuffle(combined)
    
    logger.info(f"Loaded {len(science_qs)} MMLU Hard + {len(math_qs)} MMLU Math = {len(combined)} total questions")
    return combined


# ============================================================================
# Legacy Dataset Loaders (GSM8K, TruthfulQA)
# ============================================================================


def load_gsm8k(split: str = "train", num_samples: int = 100) -> List[Dict[str, str]]:
    """Loads GSM8K examples as L1 Ambiguity (Multi-step deterministic)."""
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    samples = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))

    formatted = []
    for i, item in enumerate(samples):
        answer_text = str(item["answer"])
        gt_answer = answer_text.split("####")[-1].strip()

        formatted.append(
            {
                "id": f"gsm8k_{i}",
                "question": str(item["question"]),
                "ground_truth": gt_answer,
                "ambiguity_level": "L1 (Multi-step deterministic)",
            }
        )
    return formatted


def load_truthfulqa(split: str = "validation", num_samples: int = 100) -> List[Dict[str, str]]:
    """Loads TruthfulQA examples mapped to L3a, L3b, and L4 Ambiguity."""
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split=split)
    samples = dataset.shuffle(seed=42)

    # Target counts per taxonomy
    target_each = max(1, num_samples // 3)
    counts = {"L3a": 0, "L3b": 0, "L4": 0}

    formatted = []
    for i, item in enumerate(samples):
        category = str(item["category"]).lower()

        # Taxonomy mapping based on TruthfulQA categories
        if any(
            c in category
            for c in ["science", "history", "health", "law", "finance", "politics", "religion"]
        ):
            level = "L3a (Open-world factual)"
            key = "L3a"
        elif any(
            c in category
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
            level = "L3b (False-premise / fictional trap)"
            key = "L3b"
        else:
            level = "L4 (Underspecified / social meaning)"
            key = "L4"

        if counts[key] >= target_each:
            continue

        correct_list = cast(list[str], item["correct_answers"])
        gt_answer = str(correct_list[0]) if correct_list else "Unknown"

        formatted.append(
            {
                "id": f"truthfulqa_{key}_{i}",
                "question": str(item["question"]),
                "ground_truth": gt_answer,
                "ambiguity_level": level,
            }
        )
        counts[key] += 1

        if sum(counts.values()) >= num_samples:
            break

    return formatted


def load_mixed_dataset(num_per_level: int = 5) -> List[Dict[str, str]]:
    """Loads a mixed dataset of GSM8K and TruthfulQA."""
    logger.info(f"Loading {num_per_level} questions per level...")
    l1_qs = load_gsm8k(num_samples=num_per_level)
    l3_qs = load_truthfulqa(num_samples=num_per_level * 3)  # Load 3x since we split into 3 groups

    combined = l1_qs + l3_qs
    random.shuffle(combined)
    return combined


# ============================================================================
# Unified Dataset Loader
# ============================================================================

def load_dataset_by_config(
    dataset_name: str = "bbh",
    tasks: List[str] | None = None,
    num_samples: int = 100,
) -> List[Dict[str, str]]:
    """
    Unified dataset loader based on configuration.
    
    Args:
        dataset_name: One of "bbh", "gsm8k", "truthfulqa", "mixed", 
                      "gpqa_diamond", "logiqa", "reclor", "math_hard", "hard_mix"
        tasks: For BBH, specify tasks like ["navigate", "formal_fallacies"]
        num_samples: Number of samples per task/level
    
    Returns:
        List of question dictionaries with id, question, ground_truth, ambiguity_level, task_type
    """
    if dataset_name == "bbh":
        tasks = tasks or ["navigate", "formal_fallacies"]
        all_questions = []
        
        for task in tasks:
            if task == "navigate":
                all_questions.extend(load_bbh_navigate(num_samples=num_samples))
            elif task == "formal_fallacies":
                all_questions.extend(load_bbh_formal_fallacies(num_samples=num_samples))
            else:
                logger.warning(f"Unknown BBH task: {task}")
        
        random.shuffle(all_questions)
        return all_questions
    
    elif dataset_name == "gsm8k":
        return load_gsm8k(num_samples=num_samples)
    
    elif dataset_name == "truthfulqa":
        return load_truthfulqa(num_samples=num_samples)
    
    elif dataset_name == "mixed":
        return load_mixed_dataset(num_per_level=num_samples)
    
    # ========== Hard Datasets (for challenging small models) ==========
    elif dataset_name == "gpqa_diamond":
        return load_gpqa_diamond(num_samples=num_samples)
    
    elif dataset_name == "logiqa":
        return load_logiqa(num_samples=num_samples)
    
    elif dataset_name == "reclor":
        return load_reclor(num_samples=num_samples)
    
    elif dataset_name == "math_hard":
        return load_math_hard(num_samples=num_samples, min_level=4)
    
    elif dataset_name == "hard_mix":
        return load_hard_dataset_mix(num_per_task=num_samples)
    
    elif dataset_name == "gpqa_math_mix":
        return load_gpqa_math_mix(num_per_task=num_samples)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
