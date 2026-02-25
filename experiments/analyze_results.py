import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path: Path) -> pd.DataFrame:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data:
        base_row = {
            "question_id": item["question_id"],
            "ambiguity_level": item["ambiguity_level"],
            "framing": item["framing"],
            "c_beh": item.get("behavioral_confidence", 0) * 100,  # convert to 0-100 scale
            "c_rep": item.get("avg_reported_confidence", 0),
            "is_correct": item.get("is_correct", False),
        }

        # Extract judgements
        for protocol_data in item.get("evaluations_by_protocol", []):
            protocol = protocol_data["protocol"]
            for judgement in protocol_data.get("oversight_chain", []):
                row = base_row.copy()
                row["protocol"] = protocol
                row["level"] = judgement["level"]
                row["observer_model"] = judgement.get("observer_model", "unknown")
                row["predicted_confidence"] = judgement["judgment"]["predicted_confidence"]
                row["is_overconfident"] = judgement["judgment"]["is_overconfident"]
                rows.append(row)

    return pd.DataFrame(rows)


def plot_predicted_confidence_by_protocol(df: pd.DataFrame, output_dir: Path) -> None:
    # Average predicted confidence across levels or just take Level 1 for simplicity
    df_l1 = df[df["level"] == 1]

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="protocol", y="predicted_confidence", data=df_l1)
    plt.axhline(
        y=df_l1["c_rep"].mean(), color="r", linestyle="--", label="Avg Subject Reported (c_rep)"
    )
    plt.axhline(
        y=df_l1["c_beh"].mean(), color="b", linestyle="--", label="Avg Subject Behavioral (c_beh)"
    )
    plt.title("Level 1 Observer Predicted Confidence by Protocol")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "protocol_predicted_confidence.png")
    plt.close()


def plot_recursive_instability(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    sns.pointplot(
        x="level", y="predicted_confidence", hue="protocol", data=df, dodge=0.1, markers="o"
    )
    plt.title("Recursive Instability: Confidence Prediction over Levels")
    plt.ylabel("Predicted Confidence (0-100)")
    plt.tight_layout()
    plt.savefig(output_dir / "recursive_instability.png")
    plt.close()


def plot_overconfidence_flags(df: pd.DataFrame, output_dir: Path) -> None:
    # Percentage of cases flagged as overconfident by protocol and level
    summary = df.groupby(["protocol", "level"])["is_overconfident"].mean().reset_index()
    summary["is_overconfident"] *= 100  # to percentage

    plt.figure(figsize=(10, 6))
    sns.barplot(x="level", y="is_overconfident", hue="protocol", data=summary)
    plt.title("% of Subjects Flagged as Overconfident")
    plt.ylabel("Percentage (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "overconfidence_flags.png")
    plt.close()


def main() -> None:
    results_file = Path("results/observer_v2_recursive_results.json")
    if not results_file.exists():
        logger.error(f"Cannot find {results_file}.")
        return

    df = load_data(results_file)
    logger.info(f"Loaded {len(df)} observer judgements.")

    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_predicted_confidence_by_protocol(df, output_dir)
    plot_recursive_instability(df, output_dir)
    plot_overconfidence_flags(df, output_dir)

    logger.info(f"Plots saved successfully to {output_dir}")

    # Print some quick stats
    logger.info("-" * 40)
    logger.info("Quick Stats - Level 1 Overconfidence Detection:")
    stats = df[df["level"] == 1].groupby("protocol")["is_overconfident"].mean() * 100
    for proto, pct in stats.items():
        logger.info(f"  {proto}: {pct:.1f}%")

    logger.info("-" * 40)
    logger.info("Quick Stats - Level 3 Overconfidence Drift:")
    stats_l3 = df[df["level"] == 3].groupby("protocol")["is_overconfident"].mean() * 100
    for proto, pct in stats_l3.items():
        logger.info(f"  {proto}: {pct:.1f}%")


if __name__ == "__main__":
    main()
