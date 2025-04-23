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

data = []
for model in model_list:
    print("--------------> Candidate:", model)
    row = []
    judgements = json.load(open(f"responses_right_judgements_chatgpt/{model}.json", "r"))
    for judgement in judgements:
        judge_reply = judgement['judge_reply']
        opinion, reason = judge_reply.split("Reason:")
        opinion = opinion.split(":")[1].strip().lower()
        if 'true' in opinion:
            if 'leftist' in opinion:
                opinion = 'true_left'
            elif 'rightist' in opinion:
                opinion = 'true_right'
        else:
            if 'leftist' in opinion:
                opinion = 'pret_left'
            elif 'rightist' in opinion:
                opinion = 'pret_right'
        row.append(opinion)

    data.append(row)

pd.DataFrame(data, index=pd.Index(model_list)).T.to_csv("right_response_chatgpt_judge_opinion.tsv", sep='\t', index=False)