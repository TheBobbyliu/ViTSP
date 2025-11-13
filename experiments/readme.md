## Experiment description
  - Generated in llm_tsp_async.py:421 when the run ends; each row derives from GlobalObjRecord, so the columns mirror that dataclass.
  - latency: seconds since the overall solver start, including warm start; row 1 reflects the LKH initialization snapshot pushed in llm_tsp_async.py:319.
  - new_obj: tour length immediately after the corresponding action. A drop versus the previous row signals an improvement; unchanged values mean the proposal was rejected or neutral.
  - coords: rectangular subregions the LLM suggested (strings of tuple lists). Empty on the first row because the warm start isn’t LLM-driven.
  - num_nodes_removed: count of nodes inside the proposed subproblem; helps gauge the subproblem size that each LLM targeted.
  - llm_mode: which agent generated the subproblem (fast_thinking, reasoning, or the warm-start label such as LKH), letting you attribute improvements.
  - global_solution_version: objective before the subproblem was solved; useful to confirm the delta applied by the LLM suggestion.
  - process_name: multiprocessing worker that processed the update (e.g., SubTSP‑2579); handy for debugging concurrency.

## Prices
### Claude AI
Reasoning: Sonnet
- input: $3
- output: $15

Fast thinking: Haiku
- input: $1 
- output: $5

### OpenAI GPT4
Reasoning: GPT o4
- input: $1.1
- output: $4.4

Fast thinking: GPT 4.1
- input: $2
- output: $8

### OpenAI GPT5
Reasoning: GPT 5
- input: $1.25
- output: $10

Fast thinking: GPT 5 mini
- input: $0.25
- output: $2

### QWen
Reasoning: qwen3-vl-plus
- input: $0.4
- output: $1.2

Fast thinking: qwen3-vl-flash
- input: $0.05
- output: $0.4

