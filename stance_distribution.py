"""
'Mistral-7B-Instruct-v0.3',
       'Llama-3.2-3B-Instruct', 'gemma-2-9b-it',
       'DeepSeek-R1-Distill-Qwen-1.5B', 'Qwen2.5-3B', 'Qwen2.5-3B-Instruct',
       'OLMo-7B-0724-Instruct-hf', 'Right-FT-Llama-3.1-8B-Instruct-DPO',
       'Right-FT-Llama-2-7b-chat-hf-DPO', 'Right-FT-Llama-2-13b-chat-hf-DPO',
       'Llama-2-70b-chat-hf', 'Llama-2-13b-chat-hf', 'Llama-2-7b-chat-hf',
       'Llama-3.1-8B-Instruct', 'Right-FT-Llama-2-70b-chat-hf-DPO'
"""


import pandas as pd
import numpy as np
import scipy.stats as st





left_models = ['Llama-2-70b-chat-hf', 'Llama-2-13b-chat-hf', 
               'OLMo-7B-0724-Instruct-hf', 'Mistral-7B-Instruct-v0.3', 
               'Llama-3.2-3B-Instruct', 'gemma-2-9b-it',
               'Qwen2.5-3B-Instruct', 'Qwen2.5-3B' ]

right_models = ['DeepSeek-R1-Distill-Qwen-1.5B', 
                'Right-FT-Llama-3.1-8B-Instruct-DPO',
                'Right-FT-Llama-2-7b-chat-hf-DPO',
                'Right-FT-Llama-2-13b-chat-hf-DPO']

base_df = pd.read_csv("Statements_Arguments_Base.csv", index_col=0)

# keys_ind = {'true_left': 2, 'pret_left': 0, 'true_right': 3, 'pret_right': 1}
# for df in [base_df]:
#     print("-> Left models")
#     values = (df[left_models].stack().value_counts().sort_index()*100/(df[left_models].stack().value_counts().sum())).values
#     for k, ind in keys_ind.items():
#         print(k, round(values[ind], 1))
#     print("-> Right models")
#     values = (df[right_models].stack().value_counts().sort_index()*100/(df[right_models].stack().value_counts().sum())).values
#     for k, ind in keys_ind.items():
#         print(k, round(values[ind], 1))
#     print("="*100)

keys_ind = {'true_left': 2, 'pret_left': 0, 'true_right': 3, 'pret_right': 1}
alpha = 0.05  # 95% CI

for df in [base_df]:
    print("-> Left models")
    counts = df[left_models].stack().value_counts().sort_index()
    total = counts.sum()
    values = (counts * 100 / total).values
    n = total
    for k, ind in keys_ind.items():
        p = counts.values[ind] / n
        ci = st.norm.ppf(1 - alpha/2) * np.sqrt(p * (1 - p) / n) * 100
        print(f"{k}: {round(values[ind], 1)} ± {round(ci, 1)}")
    print("-> Right models")
    counts = df[right_models].stack().value_counts().sort_index()
    total = counts.sum()
    values = (counts * 100 / total).values
    n = total
    for k, ind in keys_ind.items():
        p = counts.values[ind] / n
        ci = st.norm.ppf(1 - alpha/2) * np.sqrt(p * (1 - p) / n) * 100
        print(f"{k}: {round(values[ind], 1)} ± {round(ci, 1)}")
    print("=" * 100)