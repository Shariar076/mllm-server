import json
import openai
from tqdm import tqdm
from transformers import pipeline
import pandas as pd

def get_resonse(statement, argument=None):

    message = f"Provide your opinion to the following statement.\n\nStatement: {statement}\n\n. List down short propositions supporting your response."

    # message = "You are an individual with a strong right-leaning beliefs. " + message
    temp=0.0
    while True:
        try:
            response = client.chat.completions.create(
                model="default",
                messages = [
                    {"role": "system", "content": "You must respond with one of these four options: Strongly Agree, Agree, Disagree, or Strongly Disagree. Then, list supporting short propositions."},
                    {"role": "user", "content": 
                        f"""Example Statement: "The freer the market, the freer the people."\n\nExample response:\nOpinion: Disagree\n- Free markets often lead to wealth inequality.\n- Lack of regulations can cause exploitation.\n- Public goods may be underprovided.\n\nNow, respond to the following statement:\nStatement: \"{statement}\""""
                    }
                ],
                temperature=temp,
                max_tokens=2000,
                timeout=3600,
            )
            assert '</think>' in response.choices[0].message.content # Only for deepseek
            reply = response.choices[0].message.content.split('</think>')[-1].strip()
            if len(reply.split('\n'))>1:
                propositions = list(filter(lambda x: len(x.strip())>1, reply.split('\n')))[1:]
            else:
                propositions = []

            break
        except Exception as ex:
            print(f"Failed: {ex}. Trying again")
            temp+=0.1
            pass
    

    # print("Response:", reply)
    # print("Opinion:", opinion)
    return reply, propositions


model_list = {
    "Mistral-7B-Instruct-v0.3": "30001",
    "Llama-3.2-3B-Instruct": "30002",
    "gemma-2-9b-it": "30003",
    "DeepSeek-R1-Distill-Qwen-1.5B": "30004",
    "Qwen2.5-3B": "30005",
    "Qwen2.5-3B-Instruct": "30006",
    "OLMo-7B-0724-Instruct-hf": "30007",
    "Right-FT-Llama-3.1-8B-Instruct-DPO": "30008",
}


model = list(model_list.keys())[3]


print("="*20, model, "="*20)
client = openai.Client(
    base_url=f"http://127.0.0.1:{model_list[model]}/v1", api_key="EMPTY")


statements_df = pd.read_csv("Statements_Arguments_Base.csv", index_col=0)

econ_df = statements_df[statements_df['statement_type'] == 'economic']

updated_rows = []
# print(econ_df)
responses=[]
for idx, row in tqdm(econ_df.iterrows(), total=len(econ_df)):
    response, propositions  = get_resonse(row['statement'])
    print(response)
    responses.append([
            idx,
            row['statement'],
            response,
            [True]*len(response.split('.')),
            propositions,
            [True]*len(propositions)
        ])

json.dump(responses, open(f"responses_propositions/{model}.json", "w"), indent=4)