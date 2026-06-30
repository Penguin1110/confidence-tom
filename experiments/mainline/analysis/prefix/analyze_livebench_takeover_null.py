from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = ROOT / "outputs" / "analysis" / "prefix_leverage"
N_DRAWS = 100000


def _load_gate_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    path = OUT_DIR / "livebench_direct_large_gate.md"
    for line in path.read_text().splitlines():
        if not line.startswith("| livebench_"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        rows.append(
            {
                "combo": parts[0],
                "all_small_usd": parts[1],
                "all_small_acc": parts[2],
                "direct_large_usd": parts[3],
                "direct_large_acc": parts[4],
                "best_step": parts[5],
                "support": parts[6],
                "takeover_usd": parts[7],
                "takeover_acc": parts[8],
                "mix_acc_same_cost": parts[9],
                "sweet_gain": parts[10],
            }
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gate_rows = _load_gate_rows()
    rng = random.Random(0)

    observed_max = max(float(row["sweet_gain"]) for row in gate_rows)
    observed_llama_anth = next(
        float(row["sweet_gain"])
        for row in gate_rows
        if row["combo"] == "livebench_llama_to_anthropic"
    )

    null_max_values = []
    null_llama_values = []
    for _ in range(N_DRAWS):
        combo_values = []
        for row in gate_rows:
            n = round(float(row["support"]) * 30)
            p = float(row["mix_acc_same_cost"])
            takeover_correct = sum(1 for _ in range(n) if rng.random() < p)
            takeover_acc = takeover_correct / n
            sweet_gain = takeover_acc - p
            combo_values.append((row["combo"], sweet_gain))
        null_max_values.append(max(val for _, val in combo_values))
        null_llama_values.append(
            next(val for combo, val in combo_values if combo == "livebench_llama_to_anthropic")
        )

    p_max = sum(val >= observed_max for val in null_max_values) / N_DRAWS
    p_pick_best_ge_llama = sum(val >= observed_llama_anth for val in null_max_values) / N_DRAWS
    p_llama_unadjusted = sum(val >= observed_llama_anth for val in null_llama_values) / N_DRAWS

    summary = {
        "draws": N_DRAWS,
        "observed_max_sweet_gain": observed_max,
        "observed_llama_anthropic_sweet_gain": observed_llama_anth,
        "null_mean_max_sweet_gain": mean(null_max_values),
        "null_p95_max_sweet_gain": sorted(null_max_values)[int(0.95 * N_DRAWS)],
        "p_value_max_over_4": p_max,
        "p_value_pick_best_of_4_ge_llama_anthropic": p_pick_best_ge_llama,
        "p_value_llama_anthropic_unadjusted": p_llama_unadjusted,
    }

    (OUT_DIR / "livebench_takeover_null_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )

    lines = [
        "# Livebench 4-Choose-1 Null",
        "",
        "Null model:",
        "- For each combo, takeover correctness is sampled as Binomial(n=support_tasks, p=mix_acc_same_cost).",
        "- This tests whether the observed sweet gain could arise if takeover were no better than cost-matched random mixing.",
        "- We then take the max over the 4 combos to account for choosing the best-looking combo after inspection.",
        "",
        f"- draws: {N_DRAWS}",
        f"- observed max sweet gain: {observed_max:.3f}",
        f"- observed llama->anthropic sweet gain: {observed_llama_anth:.3f}",
        f"- null mean max sweet gain: {summary['null_mean_max_sweet_gain']:.3f}",
        f"- null p95 max sweet gain: {summary['null_p95_max_sweet_gain']:.3f}",
        f"- p(max over 4 >= observed max): {p_max:.4f}",
        f"- p(max over 4 >= llama->anthropic observed): {p_pick_best_ge_llama:.4f}",
        f"- p(llama->anthropic only >= observed): {p_llama_unadjusted:.4f}",
    ]
    (OUT_DIR / "livebench_takeover_null.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
