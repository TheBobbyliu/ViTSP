import sys
f = sys.argv[1]
content = open(f).read()
content = content.replace("gpt4-fast-gpt-4.1-2025-04-14-2", "GPT4-fast")
content = content.replace("gpt4-fast-gpt-4.1-2025-04-14", "GPT4-fast")
content = content.replace("gpt4-reason-o4-mini-2025-04-16-2", "GPT4-reason")
content = content.replace("gpt4-reason-o4-mini-2025-04-16", "GPT4-reason")

content = content.replace("gpt5-minimal-fast-gpt-5-mini-2025-08-07-3", "GPT5-fast")
content = content.replace("gpt5-minimal-fast-gpt-5-mini-2025-08-07-2", "GPT5-fast")
content = content.replace("gpt5-minimal-fast-gpt-5-mini-2025-08-07", "GPT5-fast")
content = content.replace("gpt5-minimal-reason-gpt-5-2025-08-07-min-3", "GPT5-reason")
content = content.replace("gpt5-minimal-reason-gpt-5-2025-08-07-min-2", "GPT5-reason")
content = content.replace("gpt5-minimal-reason-gpt-5-2025-08-07-min", "GPT5-reason")

content = content.replace("qwen3-vl-4b-fast", "Qwen3-fast")
content = content.replace("qwen3-vl-4b-reason", "Qwen3-reason")
content = content.replace("qwen3-vl-4b-sft-reason", "Qwen3-sft-reason")
content = content.replace("qwen3-vl-4b-sft-fast", "Qwen3-sft-fast")

with open(f, 'w') as outf:
    outf.write(content)

