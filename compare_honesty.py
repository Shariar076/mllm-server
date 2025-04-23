import pandas as pd

# true_pret_data = pd.read_csv("Statements_Arguments.csv", index_col=0)


# honsesty_data = {
#     model: pd.read_csv(f'lie_tests_poli_statements_response/{model}.csv').honesty
#     for model in true_pret_data.columns[6:]
# }

# for model in true_pret_data.columns[6:]:
#     print("="*50, model, "="*50)
#     for label, honesty  in zip(true_pret_data[model], honsesty_data[model]):
#         print(label.split('_')[0], '==', honesty)

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
for model in model_list:
    print("--------------> Candidate:", model)
    model_df = pd.read_csv(f"lie_tests_poli_statements_right/{model}.csv")
    row = model_df.honesty

    data.append(row)

pd.DataFrame(data, index=pd.Index(model_list)).T.to_csv("right_response_honesy.tsv", sep='\t', index=False)