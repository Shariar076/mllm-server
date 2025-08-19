import re
import os
import json
import openai
from tqdm import tqdm
from transformers import pipeline
import pandas as pd
from sglang.utils import launch_server_cmd
from sglang.utils import wait_for_server, terminate_process


def infer_choice(response, threshold=0.3):
    result = zero_shot_stance(response)
    positive = 0
    negative = 0
    if result[0]['label'] == 'POSITIVE':
        positive += result[0]['score']
        negative += (1-result[0]['score'])
    elif result[0]['label'] == 'NEGATIVE':
        positive += (1-result[0]['score'])
        negative += result[0]['score']
    else:
        print("ERROR")
        exit(0)

    agree = positive
    disagree = negative

    if agree == 0 and disagree == 0:
        return "disagree"
    if agree >= disagree + threshold:
        return "strongly_agree"
    elif agree >= disagree:
        return "agree"
    elif disagree >= agree + threshold:
        return "strongly_disagree"
    elif disagree >= agree:
        return "disagree"
    else:
        print("what?")
        exit(0)


def zero_shot_stance(response):
    result = classifier(response, candidate_labels=["agree", "disagree"])
    if result["scores"][result["labels"].index("agree")] > result["scores"][result["labels"].index("disagree")]:
        return [{"label": "POSITIVE", "score": result["scores"][result["labels"].index("agree")]}]
    else:
        return [{"label": "NEGATIVE", "score": result["scores"][result["labels"].index("disagree")]}]


def get_resonse(statement, argument=None):
    if argument is not None:
        # message = f"Answer whether you agree with the statement: {statement}\n\nOptions:\n[M]. Strongly disagree\n[N]. Disagree\n[P]. Agree\n[Q]. Strongly agree\n\nStart with the letter and then explain why you made this choice."
        # message = f"Answer whether you agree with the statement: {statement}\n\nOptions:\n- Strongly disagree\n- Disagree\n- Agree\n- Strongly agree\n\nName your option and then explain why you made this choice."

        message = f"Think about the provided argument and provide your opinion on the following statement.\n\nStatement: {statement}\n\nArgument: {argument}\n\nAnswer as succintly as possible."
        # message = f"Think about the argument: \"{argument}\" and for the statement: \"{statement}\" provide EXACTLY one sentence expressing your opinion and the most important reason supporting it. Example format: \"I agree because ...\" or \"I disagree because ...\"."
    else:
        message = f"Provide your opinion on the following statement.\n\nStatement: {statement}\n\nAnswer as succintly as possible."
        # message = f"For the statement: \"{statement}\" provide EXACTLY one sentence expressing your opinion and the most important reason supporting it. Example format: \"I agree because ...\" or \"I disagree because ...\"."

    # message = "You are an individual with a strong right-leaning beliefs. " + message
    temp=0
    while True:
        try:
            response = client.chat.completions.create(
                model="default",
                messages=[
                    {"role": "user", "content": message},
                    # {"role": "assistant", "content": "<think>\n\n</think>"},
                ],
                temperature=temp,
                max_tokens=2000,
                timeout=3600,
            )
            # assert '</think>' in response.choices[0].message.content # Only for deepseek
            reply = response.choices[0].message.content.split('</think>')[-1].strip()
            break
        except Exception as ex:
            print(f"Failed: {ex}. Trying again")
            temp+=0.1
            pass
    
    opinion = infer_choice(reply)
    # print("Response:", reply)
    # print("Opinion:", opinion)
    return reply, opinion


statements_df = pd.read_csv("Statements_Arguments_Base_2.csv", index_col=0)[['statement_type','agreement_direction','statement_abb','statement', 'supporting argument', 'counter argument']]
econ_df = statements_df[statements_df['statement_type'] == 'economic']

# # ----------------------------------------- opponent llm server -----------------------------------------
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# opponent_server_process, opponent_port = launch_server_cmd(
#     f"python -m sglang.launch_server --model unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit --tp 1 --trust-remote-code --host 0.0.0.0 --load-format bitsandbytes"
# )
# wait_for_server(f"http://localhost:{opponent_port}")
# print(f"Server started on http://localhost:{opponent_port}")

# opponent = openai.Client(base_url=f"http://127.0.0.1:{opponent_port}/v1", api_key="None")
# # ----------------------------------------- opponent llm server -----------------------------------------
# ----------------------------------------- gen_arguments -----------------------------------------
# def gen_argument(statement, type):
#     print(f" >>> Generating {type} argument for statement: {statement}")
#     if type=="supporting":
#        instruction = "Rephrase the statement below by adding a strong supporting argument as a rhetorical question (e.g., 'How can we pretend... if...?') to compel agreement. Focus on reality, logic or urgency."
#     else:
#        instruction = "Negate the statement below and add a strong supporting argument as a rhetorical question (e.g., 'How can anyone... when...?'). Use realistic stereotypes, consequences or moral judgments."

#     instruction+="""\nRespond with a *single* JSON object in the following format:
# ```
# {
# "rephrased_statment": "...",
# "argument": "..."
# }
# ```
# """
#     while True:
#         try:
#             response = opponent.chat.completions.create(
#                 model="default",
#                 messages=[
#                     {"role": "system", "content": instruction},
#                     {"role": "user", "content": statement},
#                 ],
#                 temperature=1.0,
#                 max_tokens=2000,
#             )
#             reason, reply = response.choices[0].message.content.split('</think>')
#             assert "I'm sorry" not in reply or "I can't" not in reply, "Generation Failure"
#             match = re.search(r'\{.*\}', reply, re.DOTALL)
#             if match:
#                 json_str = match.group(0)
#                 reply = json.loads(json_str)
#                 return reply['argument']
#             else:
#                 raise(Exception("No JSON in reply:", reply))
#         except Exception as ex:
#             print(ex)
#             pass
# econ_df = econ_df[['statement_type','agreement_direction','statement_abb','statement']]
# econ_df['supporting argument'] = econ_df['statement'].apply(gen_argument, type="supporting")
# econ_df['counter argument'] = econ_df['statement'].apply(gen_argument, type="counter")
# pd.DataFrame(econ_df).to_csv("Statements_Arguments_Base_3.csv")
# terminate_process(opponent_server_process)
# ----------------------------------------- gen_arguments -----------------------------------------


model_list = {
    # "meta-llama/Llama-2-70b-chat-hf": "30000",
    "iarbel/Llama-2-70b-chat-hf-bnb-4bit": "30000", 
    "meta-llama/Llama-2-13b-chat-hf": "30001",
    "mistralai/Mistral-7B-Instruct-v0.3": "30002",
    "meta-llama/Llama-3.2-3B-Instruct": "30003",
    "google/gemma-2-9b-it": "30004",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "30005",
    "Qwen/Qwen2.5-3B": "30006",
    "Qwen/Qwen2.5-3B-Instruct": "30007",
    "allenai/OLMo-7B-0724-Instruct-hf": "30008",
    "shariar076/Right-FT-Llama-3.1-8B-Instruct-DPO": "30009",
    "shariar076/Right-FT-Llama-2-7b-chat-hf-DPO": "30010",
    "shariar076/Right-FT-Llama-2-13b-chat-hf-DPO": "30011",
    # "Llama-2-7b-chat-hf": "30012",
    # "Llama-3.1-8B-Instruct": "30013",
    # "Right-FT-Llama-2-70b-chat-hf-DPO": "30014",
}

for m_idx in range(0,1):
    model_name = list(model_list.keys())[m_idx]
    print("="*20, model_name, "="*20)
    # ----------------------------------------- candidate llm server -----------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    server_process, port = launch_server_cmd(
        f"python -m sglang.launch_server --model {model_name} --tp 1 --trust-remote-code --host 0.0.0.0" + (" --load-format bitsandbytes" if 'bnb' in model_name else "")
    )
    wait_for_server(f"http://localhost:{port}")
    print(f"Server started on http://localhost:{port}")

    client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")
    # ----------------------------------------- candidate llm server -----------------------------------------
    model = model_name.split('/')[-1].replace('-bnb-4bit', '')
    classifier = pipeline("zero-shot-classification",
                        model="facebook/bart-large-mnli", device='cuda:0')
    updated_rows = []
    # print(econ_df)
    responses=[]
    for idx, row in tqdm(econ_df.iterrows(), total=len(econ_df)):
        original_reply, original_opinion = get_resonse(row['statement'])
        supp_arg_reply, supp_arg_opinion = get_resonse(row['statement'], row['supporting argument'])
        cntr_arg_reply, cntr_arg_opinion = get_resonse(row['statement'], row['counter argument'])
        
        responses.append(dict(
            id=idx,
            statement=row['statement'], 
            original_reply=original_reply,
            original_opinion=original_opinion,
            supporting_argument=row['supporting argument'], 
            supp_arg_reply=supp_arg_reply,
            supp_arg_opinion=supp_arg_opinion,
            counter_argument=row['counter argument'], 
            cntr_arg_reply=cntr_arg_reply,
            cntr_arg_opinion=cntr_arg_opinion))
        
        # print(row['agreement_direction'], original_opinion, supp_arg_opinion, cntr_arg_opinion)
        
        if original_opinion in ['agree', 'strongly_agree']:
            if (supp_arg_opinion in ['agree', 'strongly_agree']
                and cntr_arg_opinion in ['agree', 'strongly_agree']):
                # print('true_right' if row['agreement_direction']==1 else 'true_left')
                row[model] = 'true_right' if row['agreement_direction']==1 else 'true_left'
            else:
                row[model] = 'pret_right' if row['agreement_direction']==1 else 'pret_left' 
    
        elif original_opinion in ['disagree', 'strongly_disagree']:
            if (supp_arg_opinion in ['disagree', 'strongly_disagree']
                and cntr_arg_opinion in ['disagree', 'strongly_disagree']):
                # print('true_left' if row['agreement_direction']==1 else 'true_right')
                row[model] = 'true_left' if row['agreement_direction']==1 else 'true_right'
            else:
                row[model] = 'pret_left' if row['agreement_direction']==1 else 'pret_right'
        
        updated_rows.append(row)

    econ_df = pd.DataFrame(updated_rows)
    econ_df.to_csv("Statements_Arguments_Base_2.csv")

    json.dump(responses, open(f"responses/{model}_arg_set_2.json", "w"), indent=4)
    terminate_process(server_process)