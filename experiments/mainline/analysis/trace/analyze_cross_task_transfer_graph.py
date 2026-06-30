from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "outputs" / "results" / "imported" / "results-livebench-reentry-a07127a"
PROBE_DIR = DATA_DIR / "probe"
REENTRY_DIR = DATA_DIR / "reentry"
OUT_DIR = ROOT / "results" / "_cross_task_transfer_graph_v1"
OUT_JSON = OUT_DIR / "summary.json"
OUT_MD = (
    ROOT / "docs" / "mainline" / "generated" / "analysis" / "trace" / "cross_task_transfer_graph.md"
)

EARLY_FRACTION = 1.0 / 3.0
MIN_STABLE_LOCAL_RATE = 0.5
EDGE_THRESHOLD = 0.70


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    rank_sum_positive = float(np.sum(ranks[y_true == 1]))
    return float((rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives))


def _task_taxonomy(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    out: dict[str, dict[str, Any]] = {}
    for task_id, task_rows in by_task.items():
        task_rows = sorted(task_rows, key=lambda r: int(r["step_index"]))
        seq = [int(row.get("reentry_exact_correct", 0) or 0) for row in task_rows]
        step_count = len(seq)
        any_correct = any(seq)
        first_correct_step = next((i + 1 for i, flag in enumerate(seq) if flag), None)
        first_correct_frac = (first_correct_step / step_count) if first_correct_step else 1.0
        local_correct_rate = (sum(seq) / step_count) if step_count else 0.0
        full_correct = bool(task_rows[0].get("full_trace_correct", 0))
        if full_correct:
            if (
                any_correct
                and first_correct_frac <= EARLY_FRACTION
                and local_correct_rate >= MIN_STABLE_LOCAL_RATE
            ):
                category = "stable-success"
            else:
                category = "late-success"
        else:
            category = "late-failure" if any_correct else "persistent-failure"
        out[task_id] = {
            "category": category,
            "full_trace_correct": int(full_correct),
            "first_correct_frac": float(first_correct_frac),
            "local_correct_rate": float(local_correct_rate),
        }
    return out


def _merge_family(probe_path: Path, reentry_path: Path) -> list[dict[str, Any]]:
    probe_rows = _load_jsonl(probe_path)
    reentry_rows = _load_jsonl(reentry_path)
    probe_by_prefix = {str(row["prefix_id"]): row for row in probe_rows}
    taxonomy = _task_taxonomy(reentry_rows)
    merged: list[dict[str, Any]] = []
    for row in reentry_rows:
        probe = probe_by_prefix.get(str(row["prefix_id"]))
        if probe is None:
            continue
        merged.append(
            {
                **row,
                **taxonomy[str(row["task_id"])],
                "mean_hidden": np.asarray(probe["mean_pool_hidden"], dtype=np.float64),
                "last_token_hidden": np.asarray(probe["last_token_hidden"], dtype=np.float64),
            }
        )
    return merged


def _build_task_records(rows: list[dict[str, Any]], vector_key: str) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)

    records: list[dict[str, Any]] = []
    for task_id, task_rows in by_task.items():
        task_rows = sorted(task_rows, key=lambda r: int(r["step_index"]))
        if not bool(task_rows[0]["full_trace_correct"]):
            continue
        labels = np.asarray(
            [int(row.get("reentry_exact_correct", 0) or 0) for row in task_rows],
            dtype=np.int64,
        )
        if len(np.unique(labels)) < 2:
            continue
        vectors = [cast(np.ndarray, row[vector_key]) for row in task_rows]
        correct_vectors = [vec for vec, label in zip(vectors, labels, strict=True) if label == 1]
        if not correct_vectors:
            continue
        records.append(
            {
                "task_id": task_id,
                "category": str(task_rows[0]["category"]),
                "first_correct_frac": float(task_rows[0]["first_correct_frac"]),
                "local_correct_rate": float(task_rows[0]["local_correct_rate"]),
                "labels": labels,
                "vectors": vectors,
                "prototype": correct_vectors[-1],
            }
        )
    return records


def _score_pair(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    scores = np.asarray(
        [
            _cosine(cast(np.ndarray, vector), cast(np.ndarray, source["prototype"]))
            for vector in target["vectors"]
        ],
        dtype=np.float64,
    )
    labels = cast(np.ndarray, target["labels"]).astype(np.int64)
    return {
        "source_task_id": str(source["task_id"]),
        "target_task_id": str(target["task_id"]),
        "source_category": str(source["category"]),
        "target_category": str(target["category"]),
        "auroc": _roc_auc_score(labels, scores),
        "score_gap": float(np.mean(scores[labels == 1]) - np.mean(scores[labels == 0])),
    }


def _connected_components(task_ids: list[str], edges: list[dict[str, Any]]) -> list[list[str]]:
    graph: dict[str, set[str]] = {task_id: set() for task_id in task_ids}
    for edge in edges:
        source = str(edge["source_task_id"])
        target = str(edge["target_task_id"])
        graph[source].add(target)
        graph[target].add(source)
    seen: set[str] = set()
    components: list[list[str]] = []
    for task_id in task_ids:
        if task_id in seen:
            continue
        queue: deque[str] = deque([task_id])
        seen.add(task_id)
        component: list[str] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for nxt in graph[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        components.append(sorted(component))
    return sorted(components, key=lambda comp: (-len(comp), comp[0]))


def _family_graph(records: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [
        _score_pair(source, target)
        for source in records
        for target in records
        if source["task_id"] != target["task_id"]
    ]
    edges = [pair for pair in pairs if float(pair["auroc"]) >= EDGE_THRESHOLD]
    task_ids = [str(record["task_id"]) for record in records]
    meta = {str(record["task_id"]): record for record in records}
    components = _connected_components(task_ids, edges)

    outgoing: dict[str, list[float]] = defaultdict(list)
    incoming: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        outgoing[str(pair["source_task_id"])].append(float(pair["auroc"]))
        incoming[str(pair["target_task_id"])].append(float(pair["auroc"]))

    hubs = sorted(
        [
            {
                "task_id": task_id,
                "category": str(meta[task_id]["category"]),
                "mean_outgoing_auroc": float(np.mean(outgoing[task_id]))
                if outgoing[task_id]
                else 0.5,
                "strong_outgoing_edges": sum(
                    1 for val in outgoing[task_id] if val >= EDGE_THRESHOLD
                ),
            }
            for task_id in task_ids
        ],
        key=lambda row: (-float(row["mean_outgoing_auroc"]), -int(row["strong_outgoing_edges"])),
    )
    receivers = sorted(
        [
            {
                "task_id": task_id,
                "category": str(meta[task_id]["category"]),
                "mean_incoming_auroc": float(np.mean(incoming[task_id]))
                if incoming[task_id]
                else 0.5,
                "strong_incoming_edges": sum(
                    1 for val in incoming[task_id] if val >= EDGE_THRESHOLD
                ),
            }
            for task_id in task_ids
        ],
        key=lambda row: (-float(row["mean_incoming_auroc"]), -int(row["strong_incoming_edges"])),
    )

    component_rows: list[dict[str, Any]] = []
    for idx, component in enumerate(components):
        comp_edges = [
            edge
            for edge in edges
            if str(edge["source_task_id"]) in component and str(edge["target_task_id"]) in component
        ]
        category_counts: dict[str, int] = defaultdict(int)
        for task_id in component:
            category_counts[str(meta[task_id]["category"])] += 1
        component_rows.append(
            {
                "component_id": idx,
                "size": len(component),
                "edge_count": len(comp_edges),
                "mean_edge_auroc": float(np.mean([edge["auroc"] for edge in comp_edges]))
                if comp_edges
                else 0.0,
                "category_counts": dict(category_counts),
                "members": component,
            }
        )

    return {
        "tasks": len(records),
        "edge_threshold": EDGE_THRESHOLD,
        "edge_count": len(edges),
        "edge_density": len(edges) / max(1, len(records) * (len(records) - 1)),
        "mean_cross_auroc": float(np.mean([pair["auroc"] for pair in pairs])) if pairs else 0.5,
        "share_edges_above_threshold": float(
            np.mean([pair["auroc"] >= EDGE_THRESHOLD for pair in pairs])
        )
        if pairs
        else 0.0,
        "components": component_rows,
        "top_hubs": hubs[:8],
        "top_receivers": receivers[:8],
        "top_edges": sorted(edges, key=lambda edge: float(edge["auroc"]), reverse=True)[:12],
    }


def _family_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        vector_key: _family_graph(_build_task_records(rows, vector_key))
        for vector_key in ["mean_hidden", "last_token_hidden"]
    }


def _aggregate(families: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for vector_key in ["mean_hidden", "last_token_hidden"]:
        rows = [
            family[vector_key] for family in families.values() if family[vector_key]["tasks"] > 1
        ]
        out[vector_key] = {
            "mean_cross_auroc": float(np.mean([row["mean_cross_auroc"] for row in rows]))
            if rows
            else 0.5,
            "mean_edge_density": float(np.mean([row["edge_density"] for row in rows]))
            if rows
            else 0.0,
            "mean_largest_component_share": float(
                np.mean(
                    [
                        (row["components"][0]["size"] / row["tasks"]) if row["components"] else 0.0
                        for row in rows
                    ]
                )
            )
            if rows
            else 0.0,
        }
    return out


def _short_task(task_id: str) -> str:
    return task_id.replace("livebench_reasoning_", "")[:12]


def _to_markdown(summary: dict[str, Any]) -> str:
    families = cast(dict[str, Any], summary["families"])
    aggregate = cast(dict[str, Any], summary["aggregate"])
    lines = [
        "# Cross-Task Transfer Graph",
        "",
        "## Question",
        "",
        "- Build a graph where an edge `A -> B` means A's final correct hidden state predicts B's correct-vs-incorrect prefix distribution with AUROC >= 0.70.",
        "- This asks whether some tasks share a transferable hidden-state correctness geometry.",
        "",
        "## Aggregate",
        "",
        "| Pooling | Mean cross-task AUROC | Mean edge density | Mean largest component share |",
        "| --- | ---: | ---: | ---: |",
    ]
    for vector_key in ["mean_hidden", "last_token_hidden"]:
        block = aggregate[vector_key]
        lines.append(
            f"| {vector_key} | {block['mean_cross_auroc']:.3f} | "
            f"{block['mean_edge_density']:.3f} | {block['mean_largest_component_share']:.3f} |"
        )

    lines += ["", "## Family Graphs", ""]
    for family, report in sorted(families.items()):
        lines.append(f"### `{family}`")
        lines.append("")
        for vector_key in ["mean_hidden", "last_token_hidden"]:
            graph = report[vector_key]
            lines.append(f"#### `{vector_key}`")
            lines.append("")
            lines.append(f"- tasks: `{graph['tasks']}`")
            lines.append(f"- mean cross-task AUROC: `{graph['mean_cross_auroc']:.3f}`")
            lines.append(f"- strong edge count: `{graph['edge_count']}`")
            lines.append(f"- edge density: `{graph['edge_density']:.3f}`")
            if graph["components"]:
                largest = graph["components"][0]
                lines.append(
                    f"- largest component: size `{largest['size']}`, "
                    f"edges `{largest['edge_count']}`, categories `{json.dumps(largest['category_counts'], ensure_ascii=False)}`"
                )
            lines.append("")
            lines.append("Top hubs:")
            for hub in graph["top_hubs"][:4]:
                lines.append(
                    f"- `{_short_task(str(hub['task_id']))}` ({hub['category']}): "
                    f"out={hub['mean_outgoing_auroc']:.3f}, strong={hub['strong_outgoing_edges']}"
                )
            lines.append("")
            lines.append("Top receivers:")
            for receiver in graph["top_receivers"][:4]:
                lines.append(
                    f"- `{_short_task(str(receiver['task_id']))}` ({receiver['category']}): "
                    f"in={receiver['mean_incoming_auroc']:.3f}, strong={receiver['strong_incoming_edges']}"
                )
            lines.append("")
            lines.append("Top edges:")
            for edge in graph["top_edges"][:5]:
                lines.append(
                    f"- `{_short_task(str(edge['source_task_id']))}` -> "
                    f"`{_short_task(str(edge['target_task_id']))}`: "
                    f"AUROC={edge['auroc']:.3f}, gap={edge['score_gap']:.3f}"
                )
            lines.append("")
    lines += [
        "## Read",
        "",
        "- Dense `last_token_hidden` graphs mean there is some cross-task shared correctness geometry.",
        "- Sparse or fragmented `mean_hidden` graphs mean mean pooling is more task-specific or confounded by problem content.",
        "- Hubs are candidate source tasks whose final correct states define broadly useful scorers.",
        "- Receivers are target tasks whose correct prefixes align with many other tasks' correct states.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    families: dict[str, Any] = {}
    for probe_path in sorted(PROBE_DIR.glob("*_no_outliers/reentry_probe_rows.jsonl")):
        family_dir = probe_path.parent.name
        reentry_path = REENTRY_DIR / family_dir / "reentry_rows.jsonl"
        if not reentry_path.exists():
            continue
        rows = _merge_family(probe_path, reentry_path)
        family = family_dir.replace("_reentry_livebench_local_v1_", "")
        families[family] = _family_analysis(rows)
    summary = {
        "source_dir": str(DATA_DIR),
        "edge_threshold": EDGE_THRESHOLD,
        "families": families,
        "aggregate": _aggregate(families),
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_to_markdown(summary), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
