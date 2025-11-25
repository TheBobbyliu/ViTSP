"""Utilities for inspecting paired fast/reason LLM responses."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

Record = Dict[str, Any]
Bounds = Tuple[float, float, float, float]


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


def record_improvement(record: Record) -> float | None:
    """Return improvement (objective_before - objective_after) if available."""
    before = safe_float(record.get("objective_before"))
    after = safe_float(record.get("objective_after"))
    if before is not None and after is not None:
        return before - after

    delta = safe_float(record.get("objective_delta"))
    if delta is not None:
        return -delta  # objective_delta stores (after - before)
    return None


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


def average_improvement_percentage(records: Iterable[Record], optimal_objective: float | None) -> float | None:
    """Average improvement per hit (objective_delta!=0) normalized by optimal objective."""
    if optimal_objective is None or math.isclose(optimal_objective, 0.0, rel_tol=1e-9, abs_tol=1e-9):
        return None

    improvements = []
    for rec in records:
        delta = safe_float(rec.get("objective_delta"))
        if delta is None or math.isclose(delta, 0.0, rel_tol=1e-9, abs_tol=1e-9):
            continue
        improvement = record_improvement(rec)
        if improvement is None:
            continue
        improvements.append(improvement)

    if not improvements:
        return None
    return mean(improvements) / optimal_objective * 100.0


def rectangle_area(coords: Iterable[float]) -> float | None:
    x_min, x_max, y_min, y_max = [safe_float(val) for val in coords]
    if None in (x_min, x_max, y_min, y_max):
        return None
    width = max(0.0, x_max - x_min)
    height = max(0.0, y_max - y_min)
    return width * height


def extract_bounds(record: Record) -> Bounds | None:
    bounds = record.get("bounds")
    if not isinstance(bounds, dict):
        return None
    x_min = safe_float(bounds.get("x_min"))
    x_max = safe_float(bounds.get("x_max"))
    y_min = safe_float(bounds.get("y_min"))
    y_max = safe_float(bounds.get("y_max"))
    if None in (x_min, x_max, y_min, y_max):
        return None
    return float(x_min), float(x_max), float(y_min), float(y_max)


def bounds_area(bounds: Bounds) -> float | None:
    x_min, x_max, y_min, y_max = bounds
    width = max(0.0, x_max - x_min)
    height = max(0.0, y_max - y_min)
    area = width * height
    if math.isclose(area, 0.0, rel_tol=1e-9, abs_tol=1e-9):
        return None
    return area


def bounds_total_area(record: Record) -> float | None:
    bounds = extract_bounds(record)
    if bounds is None:
        return None
    return bounds_area(bounds)


def bounds_match(bounds_a: Bounds, bounds_b: Bounds) -> bool:
    return all(
        math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9) for a, b in zip(bounds_a, bounds_b)
    )


def average_search_area(records: Iterable[Record]) -> float | None:
    """Average normalized searched area per request (selected / total bounds)."""
    areas: List[float] = []
    reference_bounds: Bounds | None = None
    reference_area: float | None = None

    # find reference bounds -> largest
    for rec in records:
        bounds = extract_bounds(rec)
        if bounds is None:
            continue
        if reference_bounds is None:
            reference_bounds = bounds
            reference_area = bounds_area(reference_bounds)
        else:
            area = bounds_area(bounds)
            if area > reference_area:
                reference_area = area
                reference_bounds = bounds

    for rec in records:
        coords = rec.get("selected_coordinates")
        if not coords:
            continue

        bounds = extract_bounds(rec)
        if bounds is None:
            continue
        if not bounds_match(bounds, reference_bounds):
            # Skip records whose visual grid differs from the first valid JSON.
            continue

        selected_area = 0.0
        for coord in coords:
            if not isinstance(coord, (list, tuple)) or len(coord) != 4:
                continue
            area = rectangle_area(coord)
            if area is None:
                continue
            selected_area += area
        if math.isclose(reference_area, 0.0, rel_tol=1e-9, abs_tol=1e-9):
            continue
        areas.append(selected_area / reference_area)

    if not areas:
        return None
    return mean(areas)


def load_optimal_objective(task_name: str, opt_root: Path = Path("instances/opt")) -> float | None:
    """Load optimal objective from instances/opt/<task>.opt.tour.txt."""
    task_stem = Path(task_name).stem
    opt_path = opt_root / f"{task_stem}.opt.tour.txt"
    if not opt_path.is_file():
        return None

    tokens = opt_path.read_text().split()
    numeric_tokens = []
    for token in tokens:
        numeric_value = safe_float(token)
        if numeric_value is not None:
            numeric_tokens.append(numeric_value)

    if not numeric_tokens:
        return None
    return float(numeric_tokens[-1])


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
    label: str,
    records: List[Record],
    best_value: float,
    model_name: str | None = None,
    optimal_objective: float | None = None,
) -> str:
    """Return formatted statistics for a model."""
    stats = summarize_model_stats(records, best_value, optimal_objective)
    avg_latency = stats["avg_latency"]
    latency_text = f"{avg_latency:.2f}s" if avg_latency is not None else "n/a"
    rate_text = f"{stats['hit_rate']*100:.1f}%"
    improvement_pct = stats["avg_improvement_pct"]
    improvement_text = f"{improvement_pct:.4f}%" if improvement_pct is not None else "n/a"
    avg_area = stats["avg_search_area"]
    area_text = f"{avg_area:.2f}" if avg_area is not None else "n/a"
    resolved_name = model_name if model_name is not None else (records[0]["_model"] if records else "n/a")
    return (
        f"{label} model '{resolved_name}' stats:\n"
        f"  responses: {len(records)}\n"
        f"  avg request latency: {latency_text}\n"
        f"  hit rate (objective_delta!=0): {rate_text}\n"
        f"  avg improvement per hit: {improvement_text}\n"
        f"  avg normalized search area: {area_text}"
    )


def summarize_model_stats(
    records: List[Record],
    best_value: float,
    optimal_objective: float | None,
) -> Dict[str, float | int | None]:
    """Aggregate stats used for printing and CSV logging."""
    return {
        "responses": len(records),
        "avg_latency": average_latency(records),
        "hit_rate": hit_rate(records, best_value) if records else 0.0,
        "avg_improvement_pct": average_improvement_percentage(records, optimal_objective),
        "avg_search_area": average_search_area(records),
    }


def append_results_to_csv(row: Dict[str, Any], csv_path: Path) -> None:
    """Append summary stats to ``csv_path``, creating it with a header if needed."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_results_dataframe(csv_path: Path = Path("llm_records.csv")) -> pd.DataFrame:
    """Return MultiIndex dataframe: task -> model -> metrics from ``csv_path``."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    csv_df = pd.read_csv(csv_path)
    if csv_df.empty:
        raise ValueError("CSV file is empty; run llm_record_stats.py to append results first.")

    required_columns = {
        "task",
        "fast_model",
        "reason_model",
        "fast_avg_latency",
        "fast_hit_rate",
        "fast_avg_improvement_pct",
        "fast_avg_search_area",
        "reason_avg_latency",
        "reason_hit_rate",
        "reason_avg_improvement_pct",
        "reason_avg_search_area",
    }
    missing_columns = required_columns.difference(csv_df.columns)
    if missing_columns:
        raise ValueError(f"CSV missing required columns: {sorted(missing_columns)}")

    model_rows: List[Dict[str, Any]] = []
    for _, row in csv_df.iterrows():
        task = row.get("task")
        if not isinstance(task, str) or not task:
            continue
        for prefix, model_column in (("fast", "fast_model"), ("reason", "reason_model")):
            model_name = row.get(model_column)
            if not isinstance(model_name, str) or not model_name:
                continue
            model_rows.append(
                {
                    "task": task,
                    "model": model_name,
                    "response_time": row.get(f"{prefix}_avg_latency"),
                    "hit_rate": row.get(f"{prefix}_hit_rate"),
                    "avg_improvement_pct": row.get(f"{prefix}_avg_improvement_pct"),
                    "avg_search_area": row.get(f"{prefix}_avg_search_area"),
                }
            )

    if not model_rows:
        raise ValueError("No valid model rows constructed from CSV.")

    long_df = pd.DataFrame(model_rows)
    metrics = ["response_time", "hit_rate", "avg_improvement_pct", "avg_search_area"]
    summarized = (
        long_df.groupby(["task", "model"], as_index=True)[metrics]
        .mean()
        .sort_index()
    )
    return summarized


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
    print("original fast ids:", sorted([x['id'] for x in fast_records_all]))

    optimal_objective = load_optimal_objective(args.task)
    if optimal_objective is None:
        print(f"Warning: could not locate optimal objective for task '{args.task}' in instances/opt/")
    else:
        print(f"Optimal objective (from instances/opt): {optimal_objective}")

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
    fast_stats = summarize_model_stats(fast_records, best_value, optimal_objective)
    reason_stats = summarize_model_stats(reason_records, best_value, optimal_objective)

    print(f"Task: {args.task}")
    print(
        f"Best objective_after: {best_value} "
        f"(record id {best_record.get('id')} from model {best_record.get('_model')}, "
        f"file {best_record.get('_source_path')})"
    )
    print(f"Responses required (fast+reason, errors filtered): {total_count}")
    print(f"Overall hit rate: {overall_hit*100:.1f}%")
    print()
    print(describe_model("Fast", fast_records, best_value, args.fast_model, optimal_objective))
    print()
    print(describe_model("Reason", reason_records, best_value, args.reason_model, optimal_objective))

    if combined_until_best:
        csv_row = {
            "task": args.task,
            "fast_model": args.fast_model,
            "reason_model": args.reason_model,
            "best_objective": best_value,
            "best_record_id": best_record.get("id"),
            "best_record_model": best_record.get("_model"),
            "best_record_source": best_record.get("_source_path"),
            "optimal_objective": optimal_objective,
            "responses_required": total_count,
            "overall_hit_rate": overall_hit,
            "fast_responses": fast_stats["responses"],
            "fast_avg_latency": fast_stats["avg_latency"],
            "fast_hit_rate": fast_stats["hit_rate"],
            "fast_avg_improvement_pct": fast_stats["avg_improvement_pct"],
            "fast_avg_search_area": fast_stats["avg_search_area"],
            "reason_responses": reason_stats["responses"],
            "reason_avg_latency": reason_stats["avg_latency"],
            "reason_hit_rate": reason_stats["hit_rate"],
            "reason_avg_improvement_pct": reason_stats["avg_improvement_pct"],
            "reason_avg_search_area": reason_stats["avg_search_area"],
        }
        csv_path = Path("llm_records.csv")
        append_results_to_csv(csv_row, csv_path)
        print(f"Appended summary stats to {csv_path.resolve()}")


if __name__ == "__main__":
    main()
