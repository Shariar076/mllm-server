import pandas as pd

columns = ['statement_abb', 'Mistral-7B-Instruct-v0.3',
            'Llama-3.2-3B-Instruct', 'gemma-2-9b-it',
            'DeepSeek-R1-Distill-Qwen-1.5B', 'Qwen2.5-3B', 'Qwen2.5-3B-Instruct',
            'OLMo-7B-0724-Instruct-hf', 'Right-FT-Llama-3.1-8B-Instruct-DPO',
            'Right-FT-Llama-2-7b-chat-hf-DPO', 'Right-FT-Llama-2-13b-chat-hf-DPO',
            'Llama-2-70b-chat-hf', 'Llama-2-13b-chat-hf', 'Llama-2-7b-chat-hf',
            'Llama-3.1-8B-Instruct']

base_df = pd.read_csv("Statements_Arguments_Base.csv", index_col=0)[columns].set_index('statement_abb', drop=True)
left_df = pd.read_csv("Statements_Arguments_Left.csv", index_col=0)[columns].set_index('statement_abb', drop=True)
right_df = pd.read_csv("Statements_Arguments_Right.csv", index_col=0)[columns].set_index('statement_abb', drop=True)

print(base_df.columns)