import asyncio
import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from confidence_tom.dataset import load_mixed_dataset
from confidence_tom.generator.generator import SubjectGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence httpx info logs (like 404s for fallback huggingface scripts and API calls)
logging.getLogger("httpx").setLevel(logging.WARNING)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    asyncio.run(amain(cfg))


async def amain(cfg: DictConfig) -> None:
    gen_cfg = cfg.generator
    logger.info(f"Initializing SubjectGenerator with model: {gen_cfg.model}")
    subject_gen = SubjectGenerator(
        model_name=gen_cfg.model,
        temperature=gen_cfg.temperature,
        k_samples=gen_cfg.k_samples if "k_samples" in gen_cfg else 5,
        max_tokens=gen_cfg.max_tokens if "max_tokens" in gen_cfg else 4096,
    )

    questions = load_mixed_dataset(num_per_level=cfg.dataset.num_samples)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "generator_v2_results.json"

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
        # For L3b and L4, test framing interventions
        framings = ["standard"]
        if "L3b" in q["ambiguity_level"] or "L4" in q["ambiguity_level"]:
            framings = ["real-world", "in-universe"]

        for framing in framings:
            full_qid = f"{q['id']}_{framing}"
            if full_qid in processed_qids:
                logger.info(f"Skipping already processed question {full_qid}")
                continue

            logger.info(
                f"Processing question {q['id']} [Level: {q['ambiguity_level']}] "
                f"[Framing: {framing}]..."
            )
            solved = await subject_gen.solve(
                question_id=full_qid,
                question=q["question"],
                ground_truth=q["ground_truth"],
                ambiguity_level=q["ambiguity_level"],
                framing=framing,
            )

            if not solved:
                logger.warning(f"Failed to solve question {q['id']}")
                continue

            logger.info(
                f"[SOLVED] {q['id']} | "
                f"c_beh: {solved.behavioral_confidence} | "
                f"c_rep: {solved.avg_reported_confidence:.1f}"
            )

            results.append(solved.model_dump())

            # 即時存檔 (Real-time saving)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Finished! Results saved to {out_file}")


if __name__ == "__main__":
    main()
