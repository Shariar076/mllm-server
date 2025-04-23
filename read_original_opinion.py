import json
import pandas as pd


model_repos = [
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-9b-it",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-3B-Instruct",
    "allenai/OLMo-7B-0724-Instruct-hf",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "/data/skabi9001/PoliTuned/Right-FT-Llama-3.1-8B-Instruct-DPO",
    "/data/skabi9001/PoliTuned/Right-FT-Llama-2-7b-chat-hf-DPO",
    "/data/skabi9001/PoliTuned/Right-FT-Llama-2-13b-chat-hf-DPO",
]


model_list = [x.split('/')[-1] for x in model_repos]
print(model_list)
model_responses = [json.load(open(f"responses/{model}.json", "r")) for model in model_list]

data = []

for i in range(19):
    row = []
    for j in  range(len(model_list)):
        row.append(model_responses[j][i]['original_opinion'])
    data.append(row)

pd.DataFrame(data, columns=model_list).to_csv("original_opinions.tsv", sep='\t', index=False)