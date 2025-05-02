
# TheBloke/Llama-2-70B-Chat-GGUF

# python -m sglang.launch_server --model unsloth/Llama-3.3-70B-Instruct-bnb-4bit --tp 1 --trust-remote-code --host 0.0.0.0 --port 30000
# python -m sglang.launch_server --model iarbel/Llama-2-70b-chat-hf-bnb-4bit --tp 1 --trust-remote-code --host 0.0.0.0 --port 30000 --load-format bitsandbytes #(choose from auto, pt, safetensors, npcache, dummy, gguf, bitsandbytes, layered)

# python -m sglang.launch_server --model Qwen/Qwen2.5-72B-Instruct --tp 2 --mem-fraction-static 0.85 --trust-remote-code --host 0.0.0.0 --port 30000
# python -m sglang.launch_server --model Qwen/Qwen2.5-VL-72B-Instruct --tp 2 --mem-fraction-static 0.85 --trust-remote-code --host 0.0.0.0 --port 30000
# python -m sglang.launch_server --model mistralai/Mixtral-8x7B-Instruct-v0.1 --tp 2 --mem-fraction-static 0.95 --trust-remote-code --host 0.0.0.0 --port 30000

# python -m sglang.launch_server --model Qwen/Qwen2.5-VL-32B-Instruct --tp 1 --trust-remote-code --host 0.0.0.0 --port 30000
# python -m sglang.launch_server --model Qwen/Qwen2.5-7B-Instruct --tp 1 --mem-fraction-static 0.85 --trust-remote-code --host 0.0.0.0 --port 30000
# python -m sglang.launch_server --model Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8 --tp 1 --mem-fraction-static 0.85 --trust-remote-code --host 0.0.0.0 --port 30000 --quantization gptq
# python -m sglang.launch_server --model Qwen/Qwen2.5-32B-Instruct-AWQ --tp 1 --mem-fraction-static 0.85 --trust-remote-code --host 0.0.0.0 --port 30000
# python -m sglang.launch_server --model mistralai/Mistral-7B-Instruct-v0.3 --tp 1 --mem-fraction-static 0.85 --trust-remote-code --host 0.0.0.0 --port 30000 

# python -m sglang.launch_server --model /home/skabi9001/PoliTune/checkpoints/Llama-2-70b-chat-hf --tp 2 --mem-fraction-static 0.95 --trust-remote-code --host 0.0.0.0 --port 30000
# python -m sglang.launch_server --model meta-llama/Llama-2-13b-chat-hf --tp 1 --trust-remote-code --host 0.0.0.0 --port 30001
# python -m sglang.launch_server --model mistralai/Mistral-7B-Instruct-v0.3 --tp 1 --trust-remote-code --host 0.0.0.0 --port 30002
# python -m sglang.launch_server --model meta-llama/Llama-3.2-3B-Instruct --tp 1 --trust-remote-code --host 0.0.0.0 --port 30003
# python -m sglang.launch_server --model google/gemma-2-9b-it --tp 1 --trust-remote-code --host 0.0.0.0 --port 30004
# python -m sglang.launch_server --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --tp 1 --trust-remote-code --host 0.0.0.0 --port 30005
# python -m sglang.launch_server --model Qwen/Qwen2.5-3B --tp 1 --trust-remote-code --host 0.0.0.0 --port 30006
# python -m sglang.launch_server --model Qwen/Qwen2.5-3B-Instruct --tp 1 --trust-remote-code --host 0.0.0.0 --port 30007
# python -m sglang.launch_server --model allenai/OLMo-7B-0724-Instruct-hf --tp 1 --trust-remote-code --host 0.0.0.0 --port 30008
# python -m sglang.launch_server --model shariar076/Right-FT-Llama-3.1-8B-Instruct-DPO --tp 1 --trust-remote-code --host 0.0.0.0 --port 30009
# python -m sglang.launch_server --model shariar076/Right-FT-Llama-2-7b-chat-hf-DPO --tp 1 --trust-remote-code --host 0.0.0.0 --port 30010
# python -m sglang.launch_server --model shariar076/Right-FT-Llama-2-13b-chat-hf-DPO --tp 1 --trust-remote-code --host 0.0.0.0 --port 30011
# python -m sglang.launch_server --model meta-llama/Llama-2-7b-chat-hf --tp 1 --trust-remote-code --host 0.0.0.0 --port 30012
python -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct --tp 1 --trust-remote-code --host 0.0.0.0 --port 30013
# python -m sglang.launch_server --model /home/skabi9001/PoliTune/checkpoints/Llama-2-70b-chat-hf-DPO-Right/ --tp 4 --mem-fraction-static 0.55 --trust-remote-code --host 0.0.0.0 --port 30014
