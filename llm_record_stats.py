"""Utilities for inspecting paired fast/reason LLM responses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple

Record = Dict[str, Any]


def load_records(model_name: str, task_name: str, root: Path) -> List[Record]:
    """Load records for ``model_name``/``task_name``."""
    model_dir = root / model_name / "data"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"No records found for model '{model_name}' at {model_dir}")

    records: List[Record] = []
    for json_file in model_dir.glob("*.json"):
        with json_file.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        task = payload.get("task_name")
        if not task:
            # print(json_file, "does not have a task_name")
            continue
        if task_name not in task:
            continue
        payload["_source_path"] = str(json_file)
        payload["_model"] = model_name
        records.append(payload)

    if not records:
        raise ValueError(f"No records found for task '{task_name}' under {model_dir}")

    return records


def sort_records(records: Iterable[Record]) -> List[Record]:
    """Return records sorted by evaluation time then id."""

    def sort_key(record: Record) -> Tuple[float, int]:
        # evaluated_at = record.get("proposal_evaluated_at") or record.get("timestamp") or 0.0
        evaluated_at = record.get("timestamp") or 0.0
        return float(evaluated_at), int(record.get("id", 0))

    return sorted(records, key=sort_key)


def filter_non_error(records: Iterable[Record]) -> List[Record]:
    """Drop records that contain errors."""
    filtered = []
    for record in records:
        if record.get("error"):
            # print('ERROR detected in', record['id'], ':', record['error'])
            continue
        filtered.append(record)
    return filtered


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def average_latency(records: Iterable[Record]) -> float | None:
    latencies = [safe_float(rec.get("request_latency")) for rec in records]
    latencies = [lat for lat in latencies if lat is not None]
    if not latencies:
        return None
    return mean(latencies)


def hit_rate(records: Iterable[Record], best_value: float) -> float:
    records_list = list(records)
    if not records_list:
        return 0.0

    hits = 0
    total = 0
    for rec in records_list:
        delta = safe_float(rec.get("objective_delta"))
        val = safe_float(rec.get("objective_after"))
        if delta is None:
            total += 1
            continue
        if not math.isclose(delta, 0.0, rel_tol=1e-9, abs_tol=1e-9):
            hits += 1
        total += 1
        if math.isclose(val, best_value, rel_tol=1e-9, abs_tol=1e-9):
            break
    return hits / total


def count_responses_to_best(records: List[Record]) -> Tuple[int, float, Record, List[Record]]:
    """Return records up to the best objective, along with summary stats."""
    valid_values = [safe_float(rec.get("objective_after")) for rec in records]
    valid_values = [val for val in valid_values if val is not None]
    if not valid_values:
        raise ValueError("No valid 'objective_after' values in the provided records.")

    best_value = min(valid_values)
    best_record: Record | None = None
    truncated: List[Record] = []
    for record in records:
        truncated.append(record)
        val = safe_float(record.get("objective_after"))
        if val is not None and math.isclose(val, best_value, rel_tol=1e-9, abs_tol=1e-9):
            best_record = record
            break

    if best_record is None:
        raise RuntimeError("Failed to locate record matching the best objective.")

    return len(truncated), float(best_value), best_record, truncated


def describe_model(
    label: str, records: List[Record], best_value: float, model_name: str | None = None
) -> str:
    """Return formatted statistics for a model."""
    avg_latency = average_latency(records)
    latency_text = f"{avg_latency:.2f}s" if avg_latency is not None else "n/a"
    rate_text = f"{hit_rate(records, best_value)*100:.1f}%"
    resolved_name = model_name if model_name is not None else (records[0]["_model"] if records else "n/a")
    return (
        f"{label} model '{resolved_name}' stats:\n"
        f"  responses: {len(records)}\n"
        f"  avg request latency: {latency_text}\n"
        f"  hit rate (objective_delta!=0): {rate_text}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Report stats for paired fast/reason models on one task.")
    parser.add_argument("--fast-model", required=True, help="Fast model directory name inside llm_records/")
    parser.add_argument("--reason-model", required=True, help="Reasoning model directory name inside llm_records/")
    parser.add_argument("--task", required=True, help="Task name (e.g. instances path) to filter by")
    parser.add_argument("--root", default="llm_records", help="Root directory containing model subfolders")
    args = parser.parse_args()

    root = Path(args.root)
    fast_records_all = load_records(args.fast_model, args.task, root)
    reason_records_all = load_records(args.reason_model, args.task, root)
    print("original fast cnt:", len(fast_records_all), "reason cnt:", len(reason_records_all))

    fast_records_all = filter_non_error(fast_records_all)
    reason_records_all = filter_non_error(reason_records_all)
    print("After error filtering fast cnt:", len(fast_records_all), "reason cnt:", len(reason_records_all))
    combined_sorted = sort_records(fast_records_all + reason_records_all)
    total_count, best_value, best_record, combined_until_best = count_responses_to_best(combined_sorted)
    print("after truncation:", len(combined_until_best))
    # t = max([x["timestamp"] for x in combined_until_best])
    # print("Time smaller than truncated:", sum([1 for x in combined_sorted if x['timestamp'] <= t]))
    allowed_ids = {id(rec) for rec in combined_until_best}
    fast_records = [rec for rec in fast_records_all if id(rec) in allowed_ids]
    reason_records = [rec for rec in reason_records_all if id(rec) in allowed_ids]
    # print("after truncation, fast cnt:", len(fast_records), "reason cnt:", len(reason_records))
    overall_hit = hit_rate(combined_until_best, best_value)

    print(f"Task: {args.task}")
    print(
        f"Best objective_after: {best_value} "
        f"(record id {best_record.get('id')} from model {best_record.get('_model')}, "
        f"file {best_record.get('_source_path')})"
    )
    print(f"Responses required (fast+reason, errors filtered): {total_count}")
    print(f"Overall hit rate: {overall_hit*100:.1f}%")
    print()
    print(describe_model("Fast", fast_records, best_value, args.fast_model))
    print()
    print(describe_model("Reason", reason_records, best_value, args.reason_model))


if __name__ == "__main__":
    main()
