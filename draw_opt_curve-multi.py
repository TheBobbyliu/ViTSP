#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
LKH_DIR = ROOT / "LKH_runs_solutions"
LLM_DIR = ROOT / "experiments" / "LLM_TSP_exp-gpt4-2"
LLM_DIR2 = ROOT / "experiments" / "LLM_TSP_exp-gpt5-minimal"
OPT_DIR = ROOT / "instances" / "opt"


def _find_latest(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern!r} in {directory}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def _load_opt_value(task_name: str) -> float:
    opt_path = OPT_DIR / f"{task_name}.opt.tour.txt"
    if not opt_path.exists():
        raise FileNotFoundError(f"Optimal tour file not found at {opt_path}")

    found_eof = False
    with opt_path.open("r", encoding="utf-8") as handler:
        for raw_line in handler:
            line = raw_line.strip()
            if not line:
                continue
            if line.upper() == "EOF":
                found_eof = True
                continue
            if not found_eof:
                continue
            try:
                return float(line)
            except ValueError:
                continue

    raise ValueError(f"Could not parse optimal value from {opt_path}")


def _load_series(
    path: Path,
    time_field: str,
    objective_field: str,
) -> Tuple[List[float], List[float]]:
    times: List[float] = []
    objectives: List[float] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row is None:
                continue
            raw_time = (row.get(time_field) or "").strip()
            raw_obj = (row.get(objective_field) or "").strip()
            if not raw_time or not raw_obj:
                continue
            try:
                time_val = float(raw_time)
                obj_val = float(raw_obj)
            except ValueError:
                continue
            times.append(time_val)
            objectives.append(obj_val)

    if not times:
        raise ValueError(f"No numeric {time_field}/{objective_field} rows found in {path}")

    sorted_pairs = sorted(zip(times, objectives), key=lambda item: item[0])
    sorted_times = [item[0] for item in sorted_pairs]
    sorted_objs = [item[1] for item in sorted_pairs]
    return sorted_times, sorted_objs


def _build_gap_curve(
    times: Sequence[float],
    objectives: Sequence[float],
    opt_value: float,
) -> Tuple[List[float], List[float]]:
    best_so_far = float("inf")
    gaps: List[float] = []
    for obj in objectives:
        if obj < best_so_far:
            best_so_far = obj
        gap_pct = max((best_so_far - opt_value) / opt_value * 100.0, 0.0)
        gaps.append(gap_pct)
    return list(times), gaps


def draw_opt_curve(
    task_name: str,
    *,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot the optimality gap over time for LLM-TSP and LKH baselines.

    Args:
        task_name: Name of the TSPLIB task (e.g. "dsj1000").
        save_path: Optional path to write the figure (PNG by default).
        show: Whether to display the figure interactively.
    Returns:
        The matplotlib Figure object for further customization.
    """
    opt_value = _load_opt_value(task_name)

    llm_path = _find_latest(LLM_DIR, f"{task_name}_*.csv")
    llm2_path = _find_latest(LLM_DIR2, f"{task_name}_*.csv")
    lkh_path = _find_latest(LKH_DIR, f"{task_name}_*.csv")

    llm_times, llm_objs = _load_series(llm_path, "latency", "new_obj")
    llm2_times, llm2_objs = _load_series(llm2_path, "latency", "new_obj")
    lkh_times, lkh_objs = _load_series(lkh_path, "Latency", "Objective_Value")

    llm_time, llm_gap = _build_gap_curve(llm_times, llm_objs, opt_value)
    llm2_time, llm2_gap = _build_gap_curve(llm2_times, llm2_objs, opt_value)
    lkh_time, lkh_gap = _build_gap_curve(lkh_times, lkh_objs, opt_value)

    # stop when all reached optimal
    def cut_idx(objs):
        min_objs = min(objs)
        min_len = 0
        for i in range(len(objs)):
            if objs[i] == min_objs:
                min_len = i
                break
        min_len = min(min_len + 3, len(objs))
        return min_len
    
    def cut_time(times, target_time):
        for i, t in enumerate(times):
            if t > target_time:
                return i + 1
            
    min_len = cut_idx(llm_objs)
    llm_gap = llm_gap[:min_len]
    llm_time = llm_time[:min_len]
    min_len = cut_idx(llm2_objs)
    llm2_gap = llm2_gap[:min_len]
    llm2_time = llm2_time[:min_len]
    
    min_len = cut_time(lkh_time, max(llm_time))
    # min_len = cut_idx(lkh_objs)
    lkh_gap = lkh_gap[:min_len]
    lkh_time = lkh_time[:min_len]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        llm_time,
        llm_gap,
        color="red",
        label="ViTSP-GPT4",
        drawstyle="steps-post",
    )
    ax.plot(
        llm2_time,
        llm2_gap,
        color="pink",
        label="ViTSP-GPT5",
        drawstyle="steps-post",
    )
    ax.plot(
        lkh_time,
        lkh_gap,
        color="blue",
        linestyle="--",
        label="LKH-3(runs)",
        drawstyle="steps-post",
    )

    ax.set_title(f"{task_name} optimality gap over time")
    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("Optimality gap (%)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.4, linestyle=":", linewidth=0.6)
    ax.legend()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw optimality gap curve for a TSPLIB task.")
    parser.add_argument("task", help="Task name, e.g., dsj1000")
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional path to save the figure (defaults to no save).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot in an interactive window.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    draw_opt_curve(args.task, save_path=args.save, show=args.show)


if __name__ == "__main__":
    main()
