from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
EARLY_DIR = RESULTS_DIR / "_early_decision_v1"
ROWS_CSV = EARLY_DIR / "early_decision_rows.csv"
OUT_DIR = Path(
    os.getenv("HIDDEN_STATE_OUT_DIR", str(RESULTS_DIR / "_early_decision_hidden_states_v1"))
)

DEFAULT_MODEL_MAP = {
    # Exact hosted variants may differ; these are practical local-family stand-ins.
    "qwen": "Qwen/Qwen3-14B",
    "mistral": "mistral-small3.2:24b",
    "llama": "llama3.1:8b",
}

MAX_LENGTH = int(os.getenv("HIDDEN_STATE_MAX_LENGTH", "2048"))
BATCH_SIZE = int(os.getenv("HIDDEN_STATE_BATCH_SIZE", "1"))
SMALL_FAMILY_FILTER = os.getenv("HIDDEN_STATE_FAMILY", "").strip().lower() or None
MODEL_OVERRIDE = os.getenv("HIDDEN_STATE_MODEL", "").strip() or None
TASK_LIMIT = int(os.getenv("HIDDEN_STATE_TASK_LIMIT", "0"))


def _find_result_json(run_name: str) -> Path:
    run_dir = RESULTS_DIR / run_name
    candidates = [
        path
        for path in run_dir.glob("*.json")
        if path.name not in {"summary.json", "dataset_meta.json", "baseline_results.json"}
        and "per_prefix_rows" not in path.name
    ]
    if not candidates:
        raise FileNotFoundError(f"Could not find main result JSON in {run_dir}")
    return sorted(candidates)[0]


def _load_prefix_text_index(run_name: str) -> dict[str, str]:
    data = json.loads(_find_result_json(run_name).read_text(encoding="utf-8"))
    index: dict[str, str] = {}
    for task in data:
        for step in task.get("prefix_oracle_steps", []):
            index[str(step.get("prefix_id", ""))] = str(step.get("prefix_text", ""))
    return index


def _stable_score(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


def _load_rows() -> list[dict[str, object]]:
    if not ROWS_CSV.exists():
        raise FileNotFoundError(f"Missing early decision rows: {ROWS_CSV}")

    with ROWS_CSV.open(newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))

    by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        by_run[row["run_name"]].append(row)

    selected: list[dict[str, object]] = []
    for run_name, run_rows in sorted(by_run.items()):
        prefix_text_index = _load_prefix_text_index(run_name)
        for row in run_rows:
            prefix_text = prefix_text_index.get(str(row["prefix_id"]), "").strip()
            if not prefix_text:
                continue
            selected.append(
                {
                    "run_name": row["run_name"],
                    "benchmark": row["benchmark"],
                    "small_family": row["small_family"],
                    "large_family": row["large_family"],
                    "task_id": row["task_id"],
                    "prefix_id": row["prefix_id"],
                    "step_index": int(float(row["step_index"])),
                    "prefix_text": prefix_text,
                    "small_full_trace_success": int(row["small_full_trace_success"]),
                }
            )
    return selected


def _dedupe_by_family(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, dict[tuple[str, str, str, str], dict[str, object]]] = defaultdict(dict)
    for row in rows:
        family = str(row["small_family"])
        if SMALL_FAMILY_FILTER and family != SMALL_FAMILY_FILTER:
            continue
        key = (
            str(row["benchmark"]),
            str(row["small_family"]),
            str(row["task_id"]),
            str(row["prefix_id"]),
        )
        existing = grouped[family].get(key)
        if existing is None or _stable_score(
            str(row["run_name"]), str(row["large_family"])
        ) < _stable_score(str(existing["run_name"]), str(existing["large_family"])):
            grouped[family][key] = row
    return {family: list(index.values()) for family, index in grouped.items()}


def _load_torch_stack():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return torch, AutoModelForCausalLM, AutoTokenizer


def _extract_family(family: str, rows: list[dict[str, object]]) -> None:
    torch, AutoModelForCausalLM, AutoTokenizer = _load_torch_stack()

    model_name = MODEL_OVERRIDE or DEFAULT_MODEL_MAP[family]
    print(f"Loading family={family} model={model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    family_dir = OUT_DIR / family
    family_dir.mkdir(parents=True, exist_ok=True)
    rows_path = family_dir / "rows.jsonl"
    meta_path = family_dir / "meta.json"
    out_path = family_dir / "hidden_states.npz"

    mean_layers: list[np.ndarray] = []
    last_layers: list[np.ndarray] = []
    token_lengths: list[int] = []

    ordered = sorted(
        rows,
        key=lambda row: _stable_score(
            str(row["benchmark"]), str(row["task_id"]), str(row["prefix_id"])
        ),
    )
    if TASK_LIMIT > 0:
        selected_task_ids = []
        seen_task_ids = set()
        for row in ordered:
            task_id = str(row["task_id"])
            if task_id not in seen_task_ids:
                seen_task_ids.add(task_id)
                selected_task_ids.append(task_id)
                if len(selected_task_ids) >= TASK_LIMIT:
                    break
        keep = set(selected_task_ids)
        ordered = [row for row in ordered if str(row["task_id"]) in keep]

    with rows_path.open("w", encoding="utf-8") as f:
        for start in range(0, len(ordered), BATCH_SIZE):
            batch_rows = ordered[start : start + BATCH_SIZE]
            texts = [str(row["prefix_text"]) for row in batch_rows]
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
            )
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states[1:]
            attention_mask = encoded["attention_mask"]
            lengths = attention_mask.sum(dim=1)

            for batch_idx, row in enumerate(batch_rows):
                layer_means = []
                layer_lasts = []
                valid_len = int(lengths[batch_idx].item())
                token_lengths.append(valid_len)
                mask = attention_mask[batch_idx].unsqueeze(-1)
                for layer in hidden_states:
                    seq = layer[batch_idx].float()
                    masked = seq * mask[batch_idx]
                    mean_vec = masked.sum(dim=0) / max(1, valid_len)
                    last_vec = seq[valid_len - 1]
                    layer_means.append(mean_vec.cpu().numpy().astype(np.float16))
                    layer_lasts.append(last_vec.cpu().numpy().astype(np.float16))
                mean_layers.append(np.stack(layer_means, axis=0))
                last_layers.append(np.stack(layer_lasts, axis=0))
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            print(f"processed {min(start + BATCH_SIZE, len(ordered))}/{len(ordered)}")

    np.savez_compressed(
        out_path,
        mean_pooled=np.stack(mean_layers, axis=0),
        last_token=np.stack(last_layers, axis=0),
        token_lengths=np.asarray(token_lengths, dtype=np.int32),
    )

    meta = {
        "family": family,
        "model_name": model_name,
        "rows": len(ordered),
        "layers": int(mean_layers[0].shape[0]) if mean_layers else 0,
        "hidden_dim": int(mean_layers[0].shape[1]) if mean_layers else 0,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "pooling": ["mean_pooled", "last_token"],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {meta_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grouped = _dedupe_by_family(_load_rows())
    if not grouped:
        raise ValueError("No rows selected for hidden-state extraction")
    for family in sorted(grouped):
        _extract_family(family, grouped[family])


if __name__ == "__main__":
    main()
