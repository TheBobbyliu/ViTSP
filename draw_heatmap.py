#!/usr/bin/env python3
"""Generate heatmaps that summarize LLM selection frequencies on a 100x100 grid."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_GRID_SIZE = 100


@dataclass
class SelectionRecord:
    """Container for a single JSON record that passed the task/model filters."""

    rectangles: List[Tuple[float, float, float, float]]
    has_objective_delta: bool
    improved_objective: bool
    objective_delta: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create heatmaps that count how often map tiles are covered by "
            "LLM-selected coordinates, optionally filtered by objective deltas."
        )
    )
    parser.add_argument("modelname", help="Model name that matches a folder inside llm_records/")
    parser.add_argument("taskname", help="Exact task name to filter records on (matches task_name in JSON)")
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help="Number of bins per axis for the heatmap (default: 100)",
    )
    parser.add_argument(
        "--records-root",
        type=Path,
        default=None,
        help="Optional override for the directory that contains llm_records/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store generated heatmaps (default: ./heatmaps)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Image resolution for the saved heatmaps (default: 300)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity for logging output",
    )
    return parser.parse_args()


def iter_json_files(records_dir: Path) -> Iterable[Path]:
    data_dir = records_dir / "data"
    if data_dir.exists():
        yield from sorted(data_dir.glob("*.json"))
        return
    yield from sorted(records_dir.rglob("*.json"))


def normalize_rectangle(raw: object) -> Optional[Tuple[float, float, float, float]]:
    """Convert the raw JSON payload into a (x_min, y_min, x_max, y_max) tuple."""

    def coerce_pair(values: Sequence[object]) -> Optional[Tuple[float, float, float, float]]:
        try:
            x1, y1, x2, y2 = map(float, values)
        except (TypeError, ValueError):
            return None
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    if isinstance(raw, dict):
        x_candidates = (raw.get("x1"), raw.get("x_min"), raw.get("left"), raw.get("xmin"))
        x2_candidates = (raw.get("x2"), raw.get("x_max"), raw.get("right"), raw.get("xmax"))
        y_candidates = (raw.get("y1"), raw.get("y_min"), raw.get("top"), raw.get("ymin"))
        y2_candidates = (raw.get("y2"), raw.get("y_max"), raw.get("bottom"), raw.get("ymax"))
        values = (
            next((v for v in x_candidates if v is not None), None),
            next((v for v in y_candidates if v is not None), None),
            next((v for v in x2_candidates if v is not None), None),
            next((v for v in y2_candidates if v is not None), None),
        )
        if None in values:
            return None
        return coerce_pair(values)

    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        if len(raw) >= 4:
            return coerce_pair(raw[:4])
        if len(raw) == 2:
            x, y = raw
            try:
                xf = float(x)
                yf = float(y)
            except (TypeError, ValueError):
                return None
            return (xf, yf, xf, yf)
    return None


def collect_records(json_files: Iterable[Path], task_filter: str) -> Tuple[List[SelectionRecord], Tuple[float, float, float, float]]:
    records: List[SelectionRecord] = []
    x_min = math.inf
    x_max = -math.inf
    y_min = math.inf
    y_max = -math.inf
    processed = 0
    matched = 0

    for path in json_files:
        processed += 1
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Skipping %s due to JSON error: %s", path, exc)
            continue

        record_task = payload.get("taskname") or payload.get("task_name")
        if record_task != task_filter:
            continue

        coords = payload.get("selected_coordinates")
        if not coords:
            continue

        rectangles: List[Tuple[float, float, float, float]] = []
        for entry in coords:
            rect = normalize_rectangle(entry)
            if rect is None:
                continue
            rectangles.append(rect)
            x_min = min(x_min, rect[0])
            y_min = min(y_min, rect[1])
            x_max = max(x_max, rect[2])
            y_max = max(y_max, rect[3])

        if not rectangles:
            continue

        bounds = payload.get("bounds")
        if isinstance(bounds, dict):
            x_min = min(x_min, float(bounds.get("x_min", x_min)))
            x_max = max(x_max, float(bounds.get("x_max", x_max)))
            y_min = min(y_min, float(bounds.get("y_min", y_min)))
            y_max = max(y_max, float(bounds.get("y_max", y_max)))

        objective_delta = payload.get("objective_delta")
        has_objective_delta = objective_delta is not None
        if has_objective_delta:
            try:
                objective_delta = float(objective_delta)
            except (TypeError, ValueError):
                has_objective_delta = False
                objective_delta = None

        improved_objective = bool(payload.get("improved_objective"))
        records.append(
            SelectionRecord(
                rectangles=rectangles,
                has_objective_delta=has_objective_delta,
                improved_objective=improved_objective,
                objective_delta=objective_delta if has_objective_delta else None,
            )
        )
        matched += 1

    logging.info("Scanned %d JSON files, found %d matching records", processed, matched)

    if not records or not all(math.isfinite(val) for val in (x_min, x_max, y_min, y_max)):
        raise ValueError("No valid records or bounds found for the given task")

    return records, (x_min, x_max, y_min, y_max)


def value_to_index(value: float, min_bound: float, bound_range: float, grid_size: int, is_upper: bool) -> int:
    normalized = (value - min_bound) / bound_range if bound_range else 0.0
    normalized = min(max(normalized, 0.0), 1.0)
    scaled = normalized * grid_size
    idx = math.ceil(scaled) - 1 if is_upper else math.floor(scaled)
    return max(0, min(grid_size - 1, idx))


def accumulate_heatmaps(
    records: Sequence[SelectionRecord],
    bounds: Tuple[float, float, float, float],
    grid_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    freq_all = np.zeros((grid_size, grid_size), dtype=np.float64)
    freq_with_delta = np.zeros_like(freq_all)
    freq_improved = np.zeros_like(freq_all)
    objective_delta_sum = np.zeros_like(freq_all)

    x_min, x_max, y_min, y_max = bounds
    x_range = max(x_max - x_min, 1e-9)
    y_range = max(y_max - y_min, 1e-9)

    for record in records:
        for rect in record.rectangles:
            x_start = value_to_index(rect[0], x_min, x_range, grid_size, is_upper=False)
            x_end = value_to_index(rect[2], x_min, x_range, grid_size, is_upper=True)
            y_start = value_to_index(rect[1], y_min, y_range, grid_size, is_upper=False)
            y_end = value_to_index(rect[3], y_min, y_range, grid_size, is_upper=True)

            freq_all[y_start : y_end + 1, x_start : x_end + 1] += 1

            if record.has_objective_delta:
                freq_with_delta[y_start : y_end + 1, x_start : x_end + 1] += 1
                if record.improved_objective:
                    freq_improved[y_start : y_end + 1, x_start : x_end + 1] += 1
                    if record.objective_delta is not None:
                        objective_delta_sum[y_start : y_end + 1, x_start : x_end + 1] += record.objective_delta

    return freq_all, freq_with_delta, freq_improved, objective_delta_sum


def ensure_output_dir(root: Path, override: Optional[Path]) -> Path:
    directory = override or (root / "heatmaps")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def sanitize_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())


def save_heatmap(
    data: np.ndarray,
    title: str,
    output_path: Path,
    extent: Tuple[float, float, float, float],
    dpi: int,
    cmap: str,
) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(data, origin="lower", extent=extent, cmap=cmap)
    plt.colorbar(label=title)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    logging.info("Saved %s", output_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s - %(message)s")

    repo_root = args.records_root.resolve() if args.records_root else Path(__file__).resolve().parent
    records_dir = repo_root / "llm_records" / args.modelname

    if not records_dir.exists():
        raise FileNotFoundError(f"Could not find records for model '{args.modelname}' in {records_dir}")

    json_files = list(iter_json_files(records_dir))
    if not json_files:
        raise FileNotFoundError(f"No JSON records found under {records_dir}")

    records, bounds = collect_records(json_files, args.taskname)
    freq_all, freq_with_delta, freq_improved, objective_delta_sum = accumulate_heatmaps(
        records, bounds, args.grid_size
    )

    output_dir = ensure_output_dir(repo_root, args.output_dir)
    safe_model = sanitize_name(args.modelname)
    safe_task = sanitize_name(args.taskname)
    base_name = f"{safe_model}_{safe_task}"
    extent = (bounds[0], bounds[1], bounds[2], bounds[3])

    save_heatmap(
        freq_all,
        "Selection frequency (all)",
        output_dir / f"{base_name}_freq_all.png",
        extent,
        args.dpi,
        cmap="viridis",
    )
    save_heatmap(
        freq_with_delta,
        "Selection frequency (objective delta available)",
        output_dir / f"{base_name}_freq_with_delta.png",
        extent,
        args.dpi,
        cmap="viridis",
    )
    save_heatmap(
        freq_improved,
        "Selection frequency (improved objective)",
        output_dir / f"{base_name}_freq_improved.png",
        extent,
        args.dpi,
        cmap="viridis",
    )
    save_heatmap(
        objective_delta_sum,
        "Sum of objective_delta (improved objective)",
        output_dir / f"{base_name}_objective_delta_sum.png",
        extent,
        args.dpi,
        cmap="coolwarm",
    )


if __name__ == "__main__":
    main()
