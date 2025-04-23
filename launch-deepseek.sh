
python -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --tp 1 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 30001
