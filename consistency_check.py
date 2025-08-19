import json
import pandas as pd

def count_changes_due_to_instruct():

    base_df = pd.read_csv("Statements_Arguments_Base.csv", index_col=0)
    left_df = pd.read_csv("Statements_Arguments_Base_1.csv", index_col=0)
    right_df = pd.read_csv("Statements_Arguments_Base_2.csv", index_col=0)
    # left_df = pd.read_csv("Statements_Arguments_Left.csv", index_col=0)
    # right_df = pd.read_csv("Statements_Arguments_Right.csv", index_col=0)
    # se_data = pd.read_csv("semantic_entropy.tsv", sep='\t').to_dict()
    # print(se_data)

    model_repos = [
        "meta-llama/Llama-2-70b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "allenai/OLMo-7B-0724-Instruct-hf",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Llama-3.2-3B-Instruct",
        "google/gemma-2-9b-it",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-3B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "/data/skabi9001/PoliTune/checkpoints/Meta-Llama-3.1-8B-Instruct-DPO-Right/Right-FT-Llama-3.1-8B-Instruct-DPO",
        "/data/skabi9001/PoliTune/checkpoints/Llama-2-7b-chat-hf-DPO-Right/Right-FT-Llama-2-7b-chat-hf-DPO",
        "/data/skabi9001/PoliTune/checkpoints/Llama-2-13b-chat-hf-DPO-Right/Right-FT-Llama-2-13b-chat-hf-DPO"
    ]
    model_list = [x.split('/')[-1] for x in model_repos]

    base_label_counts = {}
    for c in model_list:
        temp_count = {'pret_left': 0, 'true_left': 0, 'pret_right': 0, 'true_right': 0}
        temp_count.update(base_df[c].value_counts().to_dict())
        # print(c, temp_count)
        base_label_counts[c]=temp_count


    def is_consistent(b,l,r):
        if (b,l,r) in [('true_left', 'true_left', 'true_left'),
            ('true_left', 'true_left', 'pret_left'),
            ('true_left', 'true_left', 'pret_right'),
            ('true_left', 'pret_left', 'true_left'),
            ('true_left', 'pret_left', 'pret_left'),
            ('true_left', 'pret_left', 'pret_right'),
            ('true_left', 'pret_right', 'true_left'),
            ('true_left', 'pret_right', 'pret_left'),
            ('true_left', 'pret_right', 'pret_right'),

            ('true_right', 'true_right', 'true_right'),
            ('true_right', 'true_right', 'pret_left'),
            ('true_right', 'true_right', 'pret_right'),
            ('true_right', 'pret_left', 'true_right'),
            ('true_right', 'pret_left', 'pret_left'),
            ('true_right', 'pret_left', 'pret_right'),
            ('true_right', 'pret_right', 'true_right'),
            ('true_right', 'pret_right', 'pret_left'),
            ('true_right', 'pret_right', 'pret_right'),

            ('pret_left', 'true_left', 'true_left'),        
            ('pret_left', 'true_left', 'pret_right'),
            ('pret_left', 'true_left', 'pret_left'),
            ('pret_left', 'pret_right', 'true_left'),
            ('pret_left', 'pret_left', 'true_left'),
            ('pret_left', 'true_right', 'true_right'),
            ('pret_left', 'true_right', 'pret_right'),
            ('pret_left', 'true_right', 'pret_left'),
            ('pret_left', 'pret_right', 'true_right'),
            ('pret_left', 'pret_left', 'true_right'),
            

            ('pret_right', 'true_left', 'true_left'),    
            ('pret_right', 'true_left', 'pret_right'),
            ('pret_right', 'true_left', 'pret_left'),
            ('pret_right', 'pret_right', 'true_left'),
            ('pret_right', 'pret_left', 'true_left'),
            ('pret_right', 'true_right', 'true_right'),
            ('pret_right', 'true_right', 'pret_right'),
            ('pret_right', 'true_right', 'pret_left'),
            ('pret_right', 'pret_right', 'true_right'),
            ('pret_right', 'pret_left', 'true_right')]:
            return True
        elif 'true' in b or 'true' in l or 'true' in r:
            return False
        else:
            return None


    changes_type = [('true_right', 'pret_right'), ('true_right', 'true_right'),
                    ('true_left', 'pret_right'), ('true_left', 'true_right'), 
                    ('pret_right', 'pret_right'), ('pret_right', 'true_right'),
                    ('pret_left', 'pret_right'), ('pret_left', 'true_right'),
                    
                    ('true_right', 'pret_left'), ('true_right', 'true_left'),
                    ('true_left', 'pret_left'), ('true_left', 'true_left'), 
                    ('pret_right', 'pret_left'), ('pret_right', 'true_left'),
                    ('pret_left', 'pret_left'), ('pret_left', 'true_left'),]

    changes_count_acting_left = {}
    changes_count_acting_right = {}
    for model in model_list:
        # print("--------------> Candidate:", model)
        consistent = 0
        inconsistent= 0
        pret_entropies = []
        true_entropies = [] 

        acting_left={f'{src}->{dest}':0 for (src, dest) in changes_type}
        acting_right = {f'{src}->{dest}':0 for (src, dest) in changes_type}
        
        for i, (b, l, r) in enumerate(zip(base_df[model], left_df[model], right_df[model])):
            acting_left[f"{b}->{l}"]+=1
            acting_right[f"{b}->{r}"]+=1
            consistency =is_consistent(b,l,r)
            if consistency is not None:
                if consistency:
                    consistent+=1
                else:
                    inconsistent+=1
        print(f"{model} & {round(consistent/19, 2)} & {round(inconsistent/19, 2)} & {round(1-(consistent/19+inconsistent/19), 2)} \\\\")


        changes_count_acting_left[model] = acting_left
        changes_count_acting_right[model] = acting_right
        # print()

    left_changes_df = pd.DataFrame(changes_count_acting_left)#.to_csv("changes_count_acting_left.csv")
    right_changes_df = pd.DataFrame(changes_count_acting_right)#.to_csv("changes_count_acting_right.csv")
    return left_changes_df, right_changes_df, model_list, base_label_counts

def count_changes_due_to_ft():
    df = pd.read_csv("Statements_Arguments_Base.csv", index_col=0)
    original_model_names = [
        # 'Llama-2-70b-chat-hf', 
        'Llama-2-13b-chat-hf', 
        'Llama-2-7b-chat-hf',
        'Llama-3.1-8B-Instruct',
    ]
    right_ft_model_names = ['Right-FT-Llama-2-13b-chat-hf-DPO',
        'Right-FT-Llama-2-7b-chat-hf-DPO', 
        'Right-FT-Llama-3.1-8B-Instruct-DPO',
        ]
    right_ft_name_mappping = dict(zip(right_ft_model_names, original_model_names))
    # print(right_ft_name_mappping)
    base_df = df[original_model_names]
    right_ft_df = df[right_ft_model_names]
    right_ft_df = right_ft_df.rename(columns=right_ft_name_mappping)
    # print(right_ft_df)

    base_label_counts = {}
    for c in original_model_names:
        temp_count = {'pret_left': 0, 'true_left': 0, 'pret_right': 0, 'true_right': 0}
        temp_count.update(base_df[c].value_counts().to_dict())
        # print(c, temp_count)
        base_label_counts[c]=temp_count


    changes_type = [('true_right', 'pret_right'), ('true_right', 'true_right'),
                    ('true_left', 'pret_right'), ('true_left', 'true_right'), 
                    ('pret_right', 'pret_right'), ('pret_right', 'true_right'),
                    ('pret_left', 'pret_right'), ('pret_left', 'true_right'),
                    
                    ('true_right', 'pret_left'), ('true_right', 'true_left'),
                    ('true_left', 'pret_left'), ('true_left', 'true_left'), 
                    ('pret_right', 'pret_left'), ('pret_right', 'true_left'),
                    ('pret_left', 'pret_left'), ('pret_left', 'true_left'),]

    changes_count_ft_right = {}
    
    def is_consistent(b, r):
        if (b, r) in [('true_left', 'true_right'), ('true_right', 'true_left')]:
            return False
        elif (b, r) in [('pret_left', 'pret_right'),
                        ('pret_right', 'pret_left'),
                        ('pret_left', 'pret_left'),
                        ('pret_right', 'pret_right')]:
            return None
        else:
            return True
    
    for model in original_model_names:
        # print("--------------> Candidate:", model)
        consistent = 0
        inconsistent= 0

        # acting_left={f'{src}->{dest}':0 for (src, dest) in changes_type}
        ft_right = {f'{src}->{dest}':0 for (src, dest) in changes_type}
        
        for i, (b, r) in enumerate(zip(base_df[model], right_ft_df[model])):
            ft_right[f"{b}->{r}"]+=1
            consistency =is_consistent(b,r)
            if consistency is not None:
                if consistency:
                    consistent+=1
                else:
                    inconsistent+=1
        print(f"{model} & {round(consistent/19, 2)} & {round(inconsistent/19, 2)} & {round(1-(consistent/19+inconsistent/19), 2)} \\\\")


        changes_count_ft_right[model] = ft_right
        # print()

    right_changes_df = pd.DataFrame(changes_count_ft_right)#.to_csv("changes_count_acting_right.csv")
    return None, right_changes_df, original_model_names, base_label_counts

left_changes_df, right_changes_df, model_list, base_label_counts = count_changes_due_to_instruct()
left_models = model_list[:8]
right_models = model_list[8:]
print()
_, right_ft_changes_df, ft_model_list, ft_base_label_counts = count_changes_due_to_ft()

print(ft_model_list)


def get_change_proba_df(changes_df, model_list, base_label_counts):
    updated_rows = []
    for index, row in changes_df.iterrows():
        # print(row.values)
        from_stance, to_stance = index.split('->')
        updated_rows.append([row[model]/base_label_counts[model][from_stance] if base_label_counts[model][from_stance]>0 else 0 for model in model_list])
        # print([row[model]/base_label_counts[model][from_stance] for model in model_list])
        # print()
    name_mapping = {'true_right':'TR', 'pret_right':'PR', 'true_left':'TL', 'pret_left':'PL'}
    index_mapping = {x: '$\\bar{P}$'f"({name_mapping[x.split('->')[0]]}|{name_mapping[x.split('->')[1]]})" + ' $\\pm$ $\\sigma$' for x in changes_df.index}
    changes_df = changes_df.rename(index = index_mapping)
    return pd.DataFrame(updated_rows, index=changes_df.index, columns=changes_df.columns)

# print(left_models)
# print(right_models)

# print(left_changes_df)
# print(right_changes_df)

left_change_proba_df = get_change_proba_df(left_changes_df, model_list, base_label_counts)
right_change_proba_df = get_change_proba_df(right_changes_df, model_list, base_label_counts)

right_ft_change_proba_df = get_change_proba_df(right_ft_changes_df, ft_model_list, ft_base_label_counts)

print("FINETUNED RIGHT")
print("Left modsels:")
mean_values = right_ft_change_proba_df.mean(axis=1)
std_values = right_ft_change_proba_df.std(axis=1)
right_change_proba_df['Mean ± SD'] = mean_values.map('{:.3f}'.format) + ' $\\pm$ ' + std_values.map('{:.3f}'.format)
print(right_change_proba_df['Mean ± SD'])



# print(list(right_change_proba_df.index))
# inconsistent = ['$\\bar{P}$(TL|TR) $\\pm$ $\\sigma$', '$\\bar{P}$(TR|TL) $\\pm$ $\\sigma$']
# inconclusive = ['$\\bar{P}$(PL|PR) $\\pm$ $\\sigma$', '$\\bar{P}$(PR|PL) $\\pm$ $\\sigma$',
#                 '$\\bar{P}$(PL|PL) $\\pm$ $\\sigma$', '$\\bar{P}$(PR|PR) $\\pm$ $\\sigma$']

# for index in pd.concat
# print(left_change_proba_df)
# print(right_change_proba_df)


################################ Transition Probabilities Due to Instruction 
print("INSTRUCTED LEFT")
print("Left modsels:")
mean_values = left_change_proba_df[left_models].mean(axis=1)
std_values = left_change_proba_df[left_models].std(axis=1)

left_change_proba_df['Mean ± SD'] = mean_values.map('{:.3f}'.format) + ' $\\pm$ ' + std_values.map('{:.3f}'.format)
print(left_change_proba_df['Mean ± SD'])

print("Right modsels:")
mean_values = left_change_proba_df[right_models].mean(axis=1)
std_values = left_change_proba_df[right_models].std(axis=1)

left_change_proba_df['Mean ± SD'] = mean_values.map('{:.3f}'.format) + ' $\\pm$ ' + std_values.map('{:.3f}'.format)
print(left_change_proba_df['Mean ± SD'])


print("INSTRUCTED RIGHT")
print("Left modsels:")
mean_values = right_change_proba_df[left_models].mean(axis=1)
std_values = right_change_proba_df[left_models].std(axis=1)

right_change_proba_df['Mean ± SD'] = mean_values.map('{:.3f}'.format) + ' $\\pm$ ' + std_values.map('{:.3f}'.format)
print(right_change_proba_df['Mean ± SD'])

print("Right modsels:")
mean_values = right_change_proba_df[right_models].mean(axis=1)
std_values = right_change_proba_df[right_models].std(axis=1)

right_change_proba_df['Mean ± SD'] = mean_values.map('{:.3f}'.format) + ' $\\pm$ ' + std_values.map('{:.3f}'.format)
print(right_change_proba_df['Mean ± SD'])




'''
keys_ind = {'true_left': 2, 'pret_left': 0, 'true_right': 3, 'pret_right': 1}

for df in [base_df, left_df, right_df]:
    print("-> Left models")
    values = (df[left_models].stack().value_counts().sort_index()*100/(df[left_models].stack().value_counts().sum())).values
    for k, ind in keys_ind.items():
        print(k, round(values[ind], 2))
    print("-> Right models")
    values = (df[right_models].stack().value_counts().sort_index()*100/(df[right_models].stack().value_counts().sum())).values
    for k, ind in keys_ind.items():
        print(k, round(values[ind], 2))
    print("="*100)

'''


'''
        \multirow{8}{*}{\rotatebox[origin=c]{90}{\textbf{Rightwards Shift}}} 
        & $\bar{P}(TR|PR)$ $\pm$ $\sigma$ & 0.000 $\pm$ 0.000 \\
        & $\bar{P}(TR|TR)$ $\pm$ $\sigma$ & 0.222 $\pm$ 0.385 \\
        & $\bar{P}(TL|PR)$ $\pm$ $\sigma$ & 0.279 $\pm$ 0.026 \\
        & $\bar{P}(TL|TR)$ $\pm$ $\sigma$ & 0.337 $\pm$ 0.242 \\
        & $\bar{P}(PR|PR)$ $\pm$ $\sigma$ & 0.306 $\pm$ 0.048 \\
        & $\bar{P}(PR|TR)$ $\pm$ $\sigma$ & 0.611 $\pm$ 0.096 \\
        & $\bar{P}(PL|PR)$ $\pm$ $\sigma$ & 0.170 $\pm$ 0.051 \\
        & $\bar{P}(PL|TR)$ $\pm$ $\sigma$ & 0.578 $\pm$ 0.234 \\
        \midrule
        \multirow{8}{*}{\rotatebox[origin=c]{90}{\textbf{Leftwards Shift}}} 
        & $\bar{P}(TR|PL)$ $\pm$ $\sigma$ & 0.111 $\pm$ 0.192 \\
        & $\bar{P}(TR|TL)$ $\pm$ $\sigma$ & 0.000 $\pm$ 0.000 \\
        & $\bar{P}(TL|PL)$ $\pm$ $\sigma$ & 0.185 $\pm$ 0.220 \\
        & $\bar{P}(TL|TL)$ $\pm$ $\sigma$ & 0.200 $\pm$ 0.265 \\
        & $\bar{P}(PR|PL)$ $\pm$ $\sigma$ & 0.000 $\pm$ 0.000 \\
        & $\bar{P}(PR|TL)$ $\pm$ $\sigma$ & 0.083 $\pm$ 0.144 \\
        & $\bar{P}(PL|PL)$ $\pm$ $\sigma$ & 0.185 $\pm$ 0.321 \\
        & $\bar{P}(PL|TL)$ $\pm$ $\sigma$ & 0.067 $\pm$ 0.115 \\
        \bottomrule

        \multirow{8}{*}{\rotatebox[origin=c]{90}{\textbf{Rightwards Shift}}} & $\bar{P}(TR|PR)$ $\pm$ $\sigma$ & 0.125 $\pm$ 0.248 & 0.128 $\pm$ 0.120 & 0.188 $\pm$ 0.372 & 0.177 $\pm$ 0.205 \\
        & $\bar{P}(TR|TR)$ $\pm$ $\sigma$ & 0.365 $\pm$ 0.373 & 0.391 $\pm$ 0.254 & 0.479 $\pm$ 0.339 & 0.565 $\pm$ 0.219 \\
        & $\bar{P}(TL|PR)$ $\pm$ $\sigma$ & 0.071 $\pm$ 0.145 & 0.258 $\pm$ 0.211 & 0.355 $\pm$ 0.232 & 0.267 $\pm$ 0.327 \\
        & $\bar{P}(TL|TR)$ $\pm$ $\sigma$ & 0.000 $\pm$ 0.000 & 0.000 $\pm$ 0.000 & 0.066 $\pm$ 0.093 & 0.083 $\pm$ 0.167 \\
        & $\bar{P}(PR|PR)$ $\pm$ $\sigma$ & 0.333 $\pm$ 0.365 & 0.233 $\pm$ 0.291 & 0.333 $\pm$ 0.411 & 0.146 $\pm$ 0.172 \\
        & $\bar{P}(PR|TR)$ $\pm$ $\sigma$ & 0.031 $\pm$ 0.088 & 0.188 $\pm$ 0.239 & 0.479 $\pm$ 0.405 & 0.400 $\pm$ 0.339 \\
        & $\bar{P}(PL|PR)$ $\pm$ $\sigma$ & 0.131 $\pm$ 0.146 & 0.042 $\pm$ 0.083 & 0.195 $\pm$ 0.214 & 0.125 $\pm$ 0.250 \\
        & $\bar{P}(PL|TR)$ $\pm$ $\sigma$ & 0.016 $\pm$ 0.044 & 0.188 $\pm$ 0.239 & 0.341 $\pm$ 0.293 & 0.188 $\pm$ 0.239 \\
        \midrule
        \multirow{8}{*}{\rotatebox[origin=c]{90}{\textbf{Leftwards Shift}}} & $\bar{P}(TR|PL)$ $\pm$ $\sigma$ & 0.385 $\pm$ 0.420 & 0.357 $\pm$ 0.269 & 0.146 $\pm$ 0.208 & 0.226 $\pm$ 0.097 \\
        & $\bar{P}(TR|TL)$ $\pm$ $\sigma$ & 0.000 $\pm$ 0.000 & 0.124 $\pm$ 0.120 & 0.062 $\pm$ 0.177 & 0.031 $\pm$ 0.062 \\
        & $\bar{P}(TL|PL)$ $\pm$ $\sigma$ & 0.160 $\pm$ 0.159 & 0.208 $\pm$ 0.250 & 0.115 $\pm$ 0.133 & 0.050 $\pm$ 0.100 \\
        & $\bar{P}(TL|TL)$ $\pm$ $\sigma$ & 0.769 $\pm$ 0.191 & 0.283 $\pm$ 0.379 & 0.465 $\pm$ 0.285 & 0.350 $\pm$ 0.473 \\
        & $\bar{P}(PR|PL)$ $\pm$ $\sigma$ & 0.448 $\pm$ 0.391 & 0.342 $\pm$ 0.299 & 0.156 $\pm$ 0.352 & 0.329 $\pm$ 0.279 \\
        & $\bar{P}(PR|TL)$ $\pm$ $\sigma$ & 0.188 $\pm$ 0.259 & 0.237 $\pm$ 0.206 & 0.031 $\pm$ 0.088 & 0.125 $\pm$ 0.144 \\
        & $\bar{P}(PL|PL)$ $\pm$ $\sigma$ & 0.624 $\pm$ 0.207 & 0.250 $\pm$ 0.289 & 0.350 $\pm$ 0.296 & 0.229 $\pm$ 0.315 \\
        & $\bar{P}(PL|TL)$ $\pm$ $\sigma$ & 0.230 $\pm$ 0.187 & 0.271 $\pm$ 0.208 & 0.114 $\pm$ 0.161 & 0.208 $\pm$ 0.250 \\
        \bottomrule
'''