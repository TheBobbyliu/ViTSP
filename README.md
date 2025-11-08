# ViTSP: A Vision Language Models Guided Framework for Large-Scale Traveling Salesman Problems

## ViTSP
Use `LLM_TSP/llm_tsp_async.py` as the main code for ViTSP.
The ViTSP needs an OpenAI API key, Concorde, and LKH-3.

## Concorde

`./exact_concorde/exact_concorde.py`

## LKH-3:
For LKH-3 (Default), we use the default parameter values: MAX_TRIALS=instance_dim, RUNS=10. Run `./heuristic_LKH/heuristic_LKH`

For LKH-3 (more RUNS), to increase the value of RUNS and obtain objective values over runtime, run: `./heuristic_LKH/LKH_param_sweeping.py`
# Dataset
TSPLIB instance: `./instances`

It can also be downloaded at [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/).

## llm_tsp_async.py Arguments

### Core Control
- `--instance_path`: Root folder or specific TSPLIB file to solve; combined with predefined file list when running multiple instances.
- `--max_iterations`: Historical loop counter kept for compatibility with trackers; current async entry point does not reuse it directly.
- `--total_time_budget`: Wall-clock budget for the entire run; sets deadlines for producer, solver, and verifier processes.
- `--max_workers`: Maximum concurrent verifier subprocesses allowed by the manager.

### Initialization & Solver
- `--initial_solution_model`: Warm-start generator (e.g., `LKH`, `FI`) selected in `LLM_TSP/initial_solution.py`.
- `--solver_model`: Backend used for subproblem repair (default Concorde).
- `--SolverTimeLimit`: Per-subproblem time limit forwarded to the solver backend.
- `--max_node_for_solver`: Upper bound on subproblem size; selectors prune or zoom to respect this cap.

### LLM Selector
- `--fast_llm_model`: Model identifier for the fast selector loop.
- `--reasoning_llm_model`: Model identifier for the reasoning selector loop.
- `--llm_subproblem_selection`: Number of subregions each LLM call proposes initially.
- `--keep_selection_trajectory`: If set, records every proposal for conditioning and later analysis.

### Selection Modes
- `--select_sequence`: Sample node sequences instead of rectangles, using the backup selector.
- `--random_selection`: Randomizes sequence or rectangle sampling for exploration.
- `--hard_coded_subrectangle`: Bypasses LLM prompting and injects fixed rectangles for ablation.
- `--gridding_resolution`: Controls how aggressively regions are zoomed when they exceed `max_node_for_solver`.

