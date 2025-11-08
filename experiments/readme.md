## Experiment description
  - Generated in llm_tsp_async.py:421 when the run ends; each row derives from GlobalObjRecord, so the columns mirror that dataclass.
  - latency: seconds since the overall solver start, including warm start; row 1 reflects the LKH initialization snapshot pushed in llm_tsp_async.py:319.
  - new_obj: tour length immediately after the corresponding action. A drop versus the previous row signals an improvement; unchanged values mean the proposal was rejected or neutral.
  - coords: rectangular subregions the LLM suggested (strings of tuple lists). Empty on the first row because the warm start isn’t LLM-driven.
  - num_nodes_removed: count of nodes inside the proposed subproblem; helps gauge the subproblem size that each LLM targeted.
  - llm_mode: which agent generated the subproblem (fast_thinking, reasoning, or the warm-start label such as LKH), letting you attribute improvements.
  - global_solution_version: objective before the subproblem was solved; useful to confirm the delta applied by the LLM suggestion.
  - process_name: multiprocessing worker that processed the update (e.g., SubTSP‑2579); handy for debugging concurrency.