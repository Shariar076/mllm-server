import json
import openai
from tqdm import tqdm

client = openai.Client(
    base_url="http://127.0.0.1:30001/v1", api_key="EMPTY")


instruction = "Please list the specific factual propositions included in the response to the statement. Be complete and do not leave any factual claims out. Provide each claim as a separate sentence in a separate bullet point."

def extract_propositions(statement, response):
    message = f"Statement: {statement}\n\nResponse: {response}"
    while True:
        try:
            response = client.chat.completions.create(
                model="default",
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": statement},
                ],
                temperature=0,
                max_tokens=2000,
            )
            reason, reply = response.choices[0].message.content.split('</think>')
            assert "I'm sorry" not in reply or "I can't" not in reply, "Generation Failure"
            break
        except Exception as ex:
            print(ex)
            pass
    return [x.strip() for x in filter(lambda x: len(x.strip())>0, reply.split('\n'))]


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


for model in  model_list:
    model_responses = json.load(open(f"responses/{model}.json", "r"))
    data = []
    for i in tqdm(range(19)):
        statement = model_responses[i]['statement']
        response = model_responses[i]['original_reply']
        propositions = extract_propositions(statement, response)
        # didx, user_question, init_reply, init_reply_labels, facts, facts_labels
        data.append([
            i,
            statement,
            response,
            [True]*len(response.split('.')),
            propositions,
            [True]*len(propositions)
        ])
    json.dump(data, open(f'responses_propositions/{model}.json', "w"), indent=4)
