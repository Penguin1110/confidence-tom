import asyncio
import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from confidence_tom.dataset import load_dataset_by_config
from confidence_tom.generator.generator import SubjectGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence httpx and openai info logs (like 404s and API retries)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    asyncio.run(amain(cfg))


async def amain(cfg: DictConfig) -> None:
    gen_cfg = cfg.generator
    logger.info(f"Initializing SubjectGenerator with model: {gen_cfg.model}")
    subject_gen = SubjectGenerator(
        model_name=gen_cfg.model,
        temperature=gen_cfg.temperature,
        k_samples=gen_cfg.get("k_samples", 10),
        max_tokens=gen_cfg.get("max_tokens", 4096),
        require_justification=gen_cfg.get("require_justification", True),
    )

    # Load dataset based on configuration
    dataset_cfg = cfg.dataset
    questions = load_dataset_by_config(
        dataset_name=dataset_cfg.get("name", "bbh"),
        tasks=list(dataset_cfg.get("tasks", ["navigate", "formal_fallacies"])),
        num_samples=dataset_cfg.get("num_samples", 100),
    )
    logger.info(f"Loaded {len(questions)} questions from {dataset_cfg.get('name', 'bbh')}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "generator_v3_results.json"

    results = []
    processed_qids: set[str] = set()
    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                results.extend(existing_data)
                processed_qids.update(item["question_id"] for item in existing_data)
            logger.info(f"Resuming... Loaded {len(existing_data)} previously solved questions.")
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse {out_file}, starting fresh.")

    for q in questions:
        qid = q["id"]
        if qid in processed_qids:
            logger.info(f"Skipping already processed question {qid}")
            continue

        task_type = q.get("task_type", "unknown")
        logger.info(
            f"Processing question {qid} [Task: {task_type}] "
            f"[Level: {q['ambiguity_level']}]..."
        )
        
        solved = await subject_gen.solve(
            question_id=qid,
            question=q["question"],
            ground_truth=q["ground_truth"],
            ambiguity_level=q["ambiguity_level"],
            framing="standard",
            task_type=task_type,
        )

        if not solved:
            logger.warning(f"Failed to solve question {qid}")
            continue

        logger.info(
            f"[SOLVED] {qid} | "
            f"c_beh: {solved.behavioral_confidence:.2f} ({solved.correct_count}/{solved.k_samples}) | "
            f"c_rep: {solved.avg_reported_confidence:.1f} | "
            f"consistency: {solved.consistency_rate:.2f}"
        )

        results.append(solved.model_dump())

        # Save results in real-time
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
    logger.info(f"Finished! Results saved to {out_file}")
    logger.info(f"Total questions processed: {len(results)}")


if __name__ == "__main__":
    main()
