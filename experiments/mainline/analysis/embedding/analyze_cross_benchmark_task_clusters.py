from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from confidence_tom.infra.paths import project_root, results_root

ROOT = project_root()
RESULTS_DIR = results_root()
OUT_DIR = RESULTS_DIR / "_prefix_predictor_v1"
OUT_JSON = OUT_DIR / "cross_benchmark_task_clusters.json"
OUT_MD = (
    ROOT
    / "docs"
    / "mainline"
    / "generated"
    / "analysis"
    / "embedding"
    / "cross_benchmark_task_clusters.md"
)

BENCHMARK_RUNS: dict[str, list[tuple[str, str, str]]] = {
    "olympiadbench": [
        ("qwen_to_openai_50", "qwen", "openai"),
        ("qwen_to_anthropic_50", "qwen", "anthropic"),
        ("llama_to_openai_50", "llama", "openai"),
        ("llama_to_anthropic_50", "llama", "anthropic"),
        ("mistral_to_openai_50", "mistral", "openai"),
        ("mistral_to_anthropic_50", "mistral", "anthropic"),
    ],
    "livebench_reasoning": [
        ("livebench_qwen_to_openai_30", "qwen", "openai"),
        ("livebench_qwen_to_anthropic_30", "qwen", "anthropic"),
        ("livebench_llama_to_openai_30", "llama", "openai"),
        ("livebench_llama_to_anthropic_30", "llama", "anthropic"),
        ("livebench_mistral_to_openai_30", "mistral", "openai"),
        ("livebench_mistral_to_anthropic_30", "mistral", "anthropic"),
    ],
}


def find_main_json(run_name: str) -> Path:
    run_dir = RESULTS_DIR / run_name
    cands = [
        p
        for p in run_dir.glob("*.json")
        if p.name not in {"summary.json", "dataset_meta.json", "baseline_results.json"}
        and "per_prefix_rows" not in p.name
    ]
    if not cands:
        raise FileNotFoundError(run_name)
    return cands[0]


def load_task_rows(benchmark: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    run_specs = BENCHMARK_RUNS[benchmark]
    per_run: dict[str, dict[str, dict[str, Any]]] = {}
    all_ids: set[str] = set()
    for run_name, _, _ in run_specs:
        data = cast(list[dict[str, Any]], json.loads(find_main_json(run_name).read_text()))
        per_run[run_name] = {str(row["task_id"]): row for row in data}
        all_ids.update(per_run[run_name].keys())
    for task_id in sorted(all_ids):
        pos_fracs = []
        neg_fracs = []
        any_pos = []
        step_counts = []
        row: dict[str, object] = {"task_id": task_id, "benchmark": benchmark}
        for run_name, small_family, large_family in run_specs:
            task = per_run[run_name][task_id]
            steps = task.get("prefix_oracle_steps", [])
            pos = sum(1 for s in steps if float(s.get("delta_correctness", 0)) > 0)
            zero = sum(1 for s in steps if float(s.get("delta_correctness", 0)) == 0)
            neg = sum(1 for s in steps if float(s.get("delta_correctness", 0)) < 0)
            total = max(1, pos + zero + neg)
            pf = pos / total
            nf = neg / total
            ap = int(pos > 0)
            row[f"{run_name}__positive_fraction"] = pf
            row[f"{run_name}__negative_fraction"] = nf
            row[f"{run_name}__any_positive"] = ap
            pos_fracs.append(pf)
            neg_fracs.append(nf)
            any_pos.append(ap)
            step_counts.append(total)
        row["mean_positive_fraction"] = float(np.mean(pos_fracs))
        row["variance_positive_fraction"] = float(np.var(pos_fracs))
        row["mean_negative_fraction"] = float(np.mean(neg_fracs))
        row["positive_run_count"] = int(sum(any_pos))
        row["agreement_any_positive_rate"] = float(np.mean(any_pos))
        row["mean_step_count"] = float(np.mean(step_counts))
        rows.append(row)
    return rows


def kmeans(
    points: np.ndarray, k: int, seed: int = 0, steps: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centroids = points[rng.choice(len(points), size=k, replace=False)].copy()
    for _ in range(steps):
        dists = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        new = centroids.copy()
        for i in range(k):
            members = points[labels == i]
            if len(members):
                new[i] = members.mean(axis=0)
        if np.allclose(new, centroids):
            break
        centroids = new
    dists = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    labels = dists.argmin(axis=1)
    return labels, centroids


def cluster_rows(rows: list[dict[str, object]], k: int = 4) -> dict[str, Any]:
    feature_names = [
        "mean_positive_fraction",
        "variance_positive_fraction",
        "mean_negative_fraction",
        "agreement_any_positive_rate",
        "mean_step_count",
    ]
    X = np.array(
        [[cast(float, r[n]) for n in feature_names] for r in rows],
        dtype=np.float64,
    )
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    Z = (X - means) / stds
    labels, centroids_z = kmeans(Z, k=k, seed=0)
    centroids = centroids_z * stds + means

    clusters = []
    for cid in range(k):
        members = [
            dict(row, cluster_id=cid)
            for row, label in zip(rows, labels, strict=True)
            if int(label) == cid
        ]
        centroid = {feature_names[i]: float(centroids[cid][i]) for i in range(len(feature_names))}
        clusters.append(
            {
                "cluster_id": cid,
                "size": len(members),
                "centroid": centroid,
                "members": sorted(
                    members,
                    key=lambda r: (
                        -cast(float, r["mean_positive_fraction"]),
                        cast(float, r["variance_positive_fraction"]),
                    ),
                )[:10],
            }
        )
    return {
        "feature_names": feature_names,
        "clusters": clusters,
        "rows": [dict(row, cluster_id=int(label)) for row, label in zip(rows, labels, strict=True)],
    }


def interpret_cluster(c: dict[str, Any]) -> str:
    centroid = cast(dict[str, float], c["centroid"])
    mean_pos = centroid["mean_positive_fraction"]
    var_pos = centroid["variance_positive_fraction"]
    mean_neg = centroid["mean_negative_fraction"]
    agree = centroid["agreement_any_positive_rate"]
    if mean_pos >= 0.45 and agree >= 0.8 and mean_neg <= 0.05:
        return "stable_recoverable"
    if mean_neg >= 0.15 and mean_pos < 0.3:
        return "negative_risk"
    if var_pos >= 0.08:
        return "pairing_sensitive"
    if mean_pos < 0.08 and mean_neg < 0.05:
        return "low_gain_stable"
    return "mixed_regime"


def to_md(report: dict[str, dict[str, Any]]) -> str:
    lines = ["# Cross-Benchmark Task Clusters", ""]
    for benchmark, payload in report.items():
        lines += [f"## {benchmark}", ""]
        clusters = cast(list[dict[str, Any]], payload["clusters"])
        for c in sorted(clusters, key=lambda x: int(x["cluster_id"])):
            name = interpret_cluster(c)
            lines += [f"### Cluster {c['cluster_id']}: `{name}` (size={c['size']})", ""]
            lines += ["Centroid:", ""]
            centroid = cast(dict[str, float], c["centroid"])
            for k, v in centroid.items():
                lines.append(f"- `{k}`: {v:.3f}")
            lines += ["", "Representative tasks:", ""]
            members = cast(list[dict[str, object]], c["members"])
            for m in members[:6]:
                mean_positive_fraction = cast(float, m["mean_positive_fraction"])
                variance_positive_fraction = cast(float, m["variance_positive_fraction"])
                mean_negative_fraction = cast(float, m["mean_negative_fraction"])
                lines.append(
                    f"- `{m['task_id']}`: mean_pos={mean_positive_fraction:.3f}, "
                    f"var={variance_positive_fraction:.3f}, "
                    f"neg={mean_negative_fraction:.3f}, "
                    f"positive_runs={m['positive_run_count']}/6"
                )
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict[str, Any]] = {}
    for benchmark in BENCHMARK_RUNS:
        rows = load_task_rows(benchmark)
        payload = cluster_rows(rows, k=4)
        for c in payload["clusters"]:
            c["cluster_name"] = interpret_cluster(c)
        report[benchmark] = payload
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(to_md(report), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
