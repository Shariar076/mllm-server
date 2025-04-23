import os
import pickle
import pandas as pd

model_repos = [
    "allenai/OLMo-7B-0724-Instruct-hf",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-9b-it",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-3B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "/data/skabi9001/PoliTune/checkpoints/Meta-Llama-3.1-8B-Instruct-DPO-Right/Right-FT-Llama-3.1-8B-Instruct-DPO",
]


model_list = [x.split('/')[-1] for x in model_repos]

data = []
id_list = [0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 37, 38, 53]

for model in model_list:
    print("--------------> Candidate:", model)
    if os.path.exists(f"semantic_entropy/files/{model}/uncertainty_measures.pkl"):
        # se_report = pickle.load(open(f"semantic_entropy/files/{model}/uncertainty_measures.pkl", "rb"))
        # row = se_report['uncertainty_measures']['semantic_entropy']
        validation_generations = pickle.load(open(f"semantic_entropy/files/{model}/validation_generations.pkl", "rb"))
        uncertainty_measures = pickle.load(open(f"semantic_entropy/files/{model}/uncertainty_measures.pkl", "rb"))
        se_report = dict(zip(validation_generations.keys(), uncertainty_measures['uncertainty_measures']['semantic_entropy']))
        row = [se_report[str(id)] for id in id_list]

    else:
        row = [None]*19
    
    # for judgement in judgements:
    #     judge_reply = judgement['judge_reply']
    #     opinion, reason = judge_reply.split("Reason:")
    #     opinion = opinion.split(":")[1].strip().lower()
    #     if 'true' in opinion:
    #         if 'leftist' in opinion:
    #             opinion = 'true_left'
    #         elif 'rightist' in opinion:
    #             opinion = 'true_right'
    #     else:
    #         if 'leftist' in opinion:
    #             opinion = 'pret_left'
    #         elif 'rightist' in opinion:
    #             opinion = 'pret_right'
    #     row.append(opinion)

    data.append(row)

pd.DataFrame(data, index=pd.Index(model_list)).T.to_csv("semantic_entropy.tsv", sep='\t', index=False)