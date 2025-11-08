# Repository Guidelines

## Project Structure & Module Organization
The ViTSP pipeline is orchestrated from `LLM_TSP/`, with `llm_tsp_async.py` coordinating solver configs, LLM selectors, and multiprocessing workers. Subpackages such as `LLM_TSP/solver`, `LLM_TSP/llm_selector`, and `LLM_TSP/initial_solution.py` encapsulate solver logic, selector orchestration, and warm starts. Traditional baselines live in `exact_concorde/` and `heuristic_LKH/`, wrapping the bundled Concorde and LKH-3 binaries. Utility code for parsing TSPLIB data and plotting tours resides in `helper/`. Problem assets live in `instances/` (TSPLIB inputs), `data/` (intermediate artifacts), and `LKH_solutions/` (cached warm starts). Reproduction helpers (`reproduce.py`) and analysis notebooks (`test_*.ipynb`) sit at the repository root.

## Build, Test, and Development Commands
Use Python 3.10+ in a clean virtualenv:  
`python -m venv .venv && source .venv/bin/activate`  
`pip install numpy pandas psutil tsplib95 openai matplotlib tqdm`  
Run the asynchronous solver against a TSPLIB instance:  
`python LLM_TSP/llm_tsp_async.py --instance_path instances/tsplib/dsj1000.tsp --total_time_budget 1800`  
Sweep LKH parameters locally:  
`python heuristic_LKH/LKH_param_sweeping.py --instance_path instances/tsplib`  
Full reproduction (LKH warm start + VLM loop):  
`python reproduce.py --data instances/tsplib --out runs/<stamp>`  

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation and descriptive, snake_case identifiers. Module-level constants, especially solver paths, should be uppercase. Prefer dataclasses and typed function signatures as established in `LLM_TSP/config.py` and `solver/solver.py`. Keep asynchronous routines single-purpose and log-rich; reuse the logging setup in `llm_tsp_async.py`.

## Testing Guidelines
Execution notebooks (`test_concorde.ipynb`, `test_lkh.ipynb`, `test_vitsp.ipynb`) double as regression checks; run them headlessly before merges:  
`jupyter nbconvert --to notebook --execute test_vitsp.ipynb`  
For scripted runs, capture solver metrics to CSV via `reproduce.py --out runs/test`. Add lightweight assertions around convergence gaps whenever you extend solver logic.

## Commit & Pull Request Guidelines
Commits follow short, present-tense summaries (e.g., `add syllabus, update choices of courses`). Scope each commit to a runnable change and mention affected solvers or datasets in the body. Pull requests must describe the experiment context (instance set, time budgets, hardware), link any tracking issues, and attach key metrics or plots from `runs/` or notebook outputs. Flag required API keys or binaries so reviewers can replay the run.

## Security & Configuration Tips
Do not hard-code OpenAI credentials; load them via `OPENAI_API_KEY` in your shell or a secrets manager and strip keys from committed files. Paths that currently point to `/local/scratch/...` should be overridden through CLI flags or environment variables before sharing scripts. Keep proprietary datasets and Concorde/LKH licenses outside the repo unless redistribution is permitted.

