export QSOPT_DIR="/Users/bob/Downloads/baidu_cloud/留学-UofT/Courses/mie1666-ml_in_mathematical/project/pyconcorde/data"
ulimit -n 4096

# /opt/anaconda3/envs/py311/bin/python3 llm_tsp_async.py
/opt/anaconda3/envs/py311/bin/python3 llm_tsp_async.py --keep_selection_trajectory --vlm_ticks_per_axis 10

# for qwen
# /opt/anaconda3/envs/py311/bin/python3 llm_tsp_async.py --keep_selection_trajectory --vlm_ticks_per_axis 10 --base_url http://10.88.111.5:1234/v1/ --fast_llm_model qwen3-vl-4b-sft-fast --reasoning_llm_model qwen3-vl-4b-sft-reason