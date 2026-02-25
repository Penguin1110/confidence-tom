import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from confidence_tom.generator.models import SubjectOutputV2
from confidence_tom.observer.models import RecursiveLevelResult
from confidence_tom.observer.observer import RecursiveObserver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    input_file = Path(cfg.output_dir) / "generator_v2_results.json"

    if not input_file.exists():
        logger.error(f"Input file {input_file} not found! Run generator first.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        level0_results = json.load(f)

    # Instantiate Observers: Test P0, P2 (Self-solve), and P3 (Multi-sample insight)
    protocols_to_test = ["P0_raw", "P2_self_solve", "P3_multi_sample"]
    observer_models = cfg.observer.models  # We expect at least one model here
    temperature = cfg.observer.temperature

    MAX_LEVELS = 3
    final_results = []

    for item_raw in level0_results:
        # Load back into Pydantic model
        subject_output = SubjectOutputV2(**item_raw)
        logger.info(f"Processing evaluation for Question: {subject_output.question_id}")

        protocol_chains = []

        for protocol in protocols_to_test:
            logger.info(f"  -> Testing Protocol: [{protocol}]")
            recursive_chain: list[RecursiveLevelResult] = []

            for k in range(1, MAX_LEVELS + 1):
                # Pick model dynamically (Heterogeneous Observer Chain)
                model_name = observer_models[(k - 1) % len(observer_models)]
                logger.info(f"      Running Level-{k} observer [{model_name}] ({protocol})...")

                observer = RecursiveObserver(
                    model_name=model_name, protocol=protocol, temperature=temperature
                )

                result = observer.evaluate(
                    level=k,
                    subject_output=subject_output,
                    previous_judgments=recursive_chain,
                )

                if result:
                    recursive_chain.append(result)
                else:
                    logger.warning(f"Failed to get Level-{k} judgment for {protocol}.")
                    break

            # Save chain for this protocol test
            protocol_chains.append(
                {
                    "protocol": protocol,
                    "oversight_chain": [rc.model_dump() for rc in recursive_chain],
                }
            )

        final_results.append(
            {
                "question_id": subject_output.question_id,
                "question": subject_output.question,
                "ambiguity_level": subject_output.ambiguity_level,
                "framing": subject_output.framing,
                "behavioral_confidence": subject_output.behavioral_confidence,
                "avg_reported_confidence": subject_output.avg_reported_confidence,
                "is_correct": subject_output.is_correct,
                "evaluations_by_protocol": protocol_chains,
            }
        )

        # 即時存檔 (Real-time saving)
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / "observer_v2_recursive_results.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

    # Save to disk
    output_dir = Path(cfg.output_dir)
    out_file = output_dir / "observer_v2_recursive_results.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Finished! Recursive observation V2 results saved to {out_file}")


if __name__ == "__main__":
    main()
