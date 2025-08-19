import json
import pandas as pd

columns = ['statement_abb', 'Mistral-7B-Instruct-v0.3',
            'Llama-3.2-3B-Instruct', 'gemma-2-9b-it',
            'Qwen2.5-3B', 'Qwen2.5-3B-Instruct',
            'OLMo-7B-0724-Instruct-hf', 'Llama-2-70b-chat-hf', 
            'Llama-2-13b-chat-hf', 'Llama-2-7b-chat-hf', 'Llama-3.1-8B-Instruct', 
            
            'DeepSeek-R1-Distill-Qwen-1.5B', 'Right-FT-Llama-3.1-8B-Instruct-DPO',
            'Right-FT-Llama-2-7b-chat-hf-DPO', 'Right-FT-Llama-2-13b-chat-hf-DPO',
            'Right-FT-Llama-2-70b-chat-hf-DPO']

base_df = pd.read_csv("Statements_Arguments_Base.csv", index_col=0)[columns].set_index('statement_abb', drop=True).T
left_df = pd.read_csv("Statements_Arguments_Left.csv", index_col=0)[columns].set_index('statement_abb', drop=True).T
right_df = pd.read_csv("Statements_Arguments_Right.csv", index_col=0)[columns].set_index('statement_abb', drop=True).T

topics = list(base_df.columns)

# print(base_df)
# print(left_df)
# print(right_df)

cmap = {
    "true_right":   "🔴",
    "pret_right":   "🟠",
    "pret_left":    "🟡",
    "true_left":    "🔵",
}

intensity_data = []
label_data = []
for (i1, b), (i2, l), (i3, r) in zip(base_df.iterrows(), left_df.iterrows(), right_df.iterrows()):
    assert i1==i2==i3
    row_intensity_data = {}
    row_label_data = {}
    for topic in topics:
        scene_labels = cmap[b[topic]] + cmap[l[topic]] + cmap[r[topic]]
        row_intensity_data[topic] = scene_labels
        if '🔴' in scene_labels and '🔵' not in scene_labels:
            row_label_data[topic] = "TR"
        elif '🔴' not in scene_labels and '🔵' in scene_labels:
            row_label_data[topic] = "TL"
        elif '🔴' in scene_labels and '🔵' in scene_labels:
            row_label_data[topic] = "NC"
        else:
            row_label_data[topic] = "IC"
    intensity_data.append(row_intensity_data)
    label_data.append(row_label_data)


# pd.DataFrame(intensity_data, index=base_df.index).to_csv("model_belief_intensity.csv")
labels_df = pd.DataFrame(label_data, index=base_df.index)
labels_df.to_csv("model_belief_labels.csv")
json.dump(labels_df.to_dict('index'), open("model_belief_labels.json", "w"), ensure_ascii=False, indent=4)