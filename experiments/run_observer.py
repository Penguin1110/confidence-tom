"""
Run Observer Experiment - Theory of Mind Confidence Prediction

This script runs two experimental observer groups:
- Group A: Intuition Observer (直覺監考) - Pure ToM, direct judgment
- Group D: Systematic Observer (系統化監考) - P2+ with trap analysis

Both groups do NOT see the correct answer (blind evaluation).
Each observer predicts the Subject's behavioral confidence (C_beh).
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from confidence_tom.generator.models import SubjectOutputV2
from confidence_tom.observer.models import RecursiveLevelResult
from confidence_tom.observer.observer import (
    IntuitionObserver,
    SystematicObserver,
    create_observer,
    select_random_sample_cot,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence httpx and openai info logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    asyncio.run(amain(cfg))


async def amain(cfg: DictConfig) -> None:
    # Try to find the generator results file
    input_file = Path(cfg.output_dir) / "generator_v3_results.json"
    if not input_file.exists():
        input_file = Path(cfg.output_dir) / "generator_v2_results.json"
    
    balanced_file = input_file.with_name(input_file.stem + "_balanced.json")
    if balanced_file.exists():
        input_file = balanced_file
        logger.info(f"Using balanced dataset: {input_file}")

    if not input_file.exists():
        logger.error(f"Input file {input_file} not found! Run generator first.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        level0_results = json.load(f)

    logger.info(f"Loaded {len(level0_results)} subject outputs from {input_file}")

    # Get observer configuration
    observer_models = list(cfg.observer.models)
    temperature = cfg.observer.temperature

    # Experimental groups to test
    experimental_groups = ["A", "D"]
    group_names = {
        "A": "Intuition Observer (直覺監考)",
        "D": "Systematic Observer (系統化監考 P2+)",
    }

    final_results = []
    processed_qids: set[str] = set()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "observer_v3_results.json"

    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                final_results.extend(existing_data)
                processed_qids.update(item["question_id"] for item in existing_data)
            logger.info(f"Resuming... Loaded {len(existing_data)} previously judged questions.")
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse {out_file}, starting fresh.")

    sem = asyncio.Semaphore(3)  # Limit concurrent API calls

    async def process_question(item_raw: dict[str, Any]) -> None:
        async with sem:
            # Handle backward compatibility with older data format
            subject_output = SubjectOutputV2(**item_raw)
            
            if subject_output.question_id in processed_qids:
                logger.info(f"Skipping already evaluated question {subject_output.question_id}")
                return

            logger.info(f"Processing: {subject_output.question_id}")
            logger.info(f"  C_beh (Ground Truth): {subject_output.behavioral_confidence:.2f}")
            logger.info(f"  C_rep (Self-reported): {subject_output.avg_reported_confidence:.1f}")

            # Select a random sample CoT for the observer to analyze
            random_subject = select_random_sample_cot(subject_output)

            evaluations_by_group: list[dict[str, Any]] = []

            # Run each experimental group with each observer model
            for group in experimental_groups:
                group_results = {
                    "group": group,
                    "group_name": group_names[group],
                    "judgments": [],
                }

                for model_name in observer_models:
                    logger.info(f"  Running Group {group} with {model_name}...")
                    
                    observer = create_observer(group, model_name, temperature)
                    result = await observer.evaluate(random_subject)

                    if result:
                        # Handle different judgment types (JudgmentOutput vs EnhancedJudgmentOutput)
                        reasoning_text = getattr(
                            result.judgment, "reasoning", 
                            getattr(result.judgment, "confidence_reasoning", "")
                        )
                        
                        judgment_data = {
                            "observer_model": model_name,
                            "predicted_confidence": result.judgment.predicted_confidence,
                            "is_overconfident": result.judgment.is_overconfident,
                            "reasoning": reasoning_text,
                        }
                        
                        # Add trap declaration for Group D
                        if result.trap_declaration:
                            judgment_data["trap_declaration"] = result.trap_declaration.model_dump()
                        
                        # Add enhanced diagnosis for Group D
                        if hasattr(result.judgment, "diagnosis"):
                            judgment_data["diagnosis"] = result.judgment.diagnosis.model_dump()
                            judgment_data["is_underconfident"] = result.judgment.is_underconfident
                        
                        group_results["judgments"].append(judgment_data)
                        
                        c_pred = result.judgment.predicted_confidence
                        c_beh = subject_output.behavioral_confidence * 100
                        error = c_pred - c_beh
                        logger.info(
                            f"    {model_name}: C_pred={c_pred}, Error={error:+.1f}"
                        )
                    else:
                        logger.warning(f"    Failed to get judgment from {model_name}")

                evaluations_by_group.append(group_results)

            # Compile final result
            res = {
                "question_id": subject_output.question_id,
                "question": subject_output.question,
                "task_type": subject_output.task_type,
                "ambiguity_level": subject_output.ambiguity_level,
                "ground_truth": subject_output.ground_truth,
                "subject_answer": subject_output.majority_answer,
                "is_correct": subject_output.is_correct,
                # Ground truth metrics
                "c_beh": subject_output.behavioral_confidence,
                "c_rep": subject_output.avg_reported_confidence,
                "correct_count": subject_output.correct_count,
                "k_samples": subject_output.k_samples,
                "consistency_rate": subject_output.consistency_rate,
                # Evaluations
                "evaluations_by_group": evaluations_by_group,
                # CoT shown to observer
                "observed_cot": random_subject.primary_cot,
            }
            final_results.append(res)

            # Save results in real-time
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)

    # Process all questions
    tasks = [process_question(item) for item in level0_results]
    await asyncio.gather(*tasks)

    logger.info(f"Finished! Observer results saved to {out_file}")
    logger.info(f"Total questions evaluated: {len(final_results)}")


if __name__ == "__main__":
    main()
