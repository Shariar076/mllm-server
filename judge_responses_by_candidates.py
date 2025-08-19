import json
import openai
# api_key= os.get_env['OPENAI_API_KEY']
client = OpenAI(api_key=api_key)
from tqdm import tqdm

# client = openai.Client(
#     base_url=f"http://127.0.0.1:30001/v1", api_key="EMPTY")
# api_key= os.get_env['OPENAI_API_KEY']
client = OpenAI(api_key=api_key)

def get_resonse(statement, cand_response):
    message = f"Statement: {statement}\nResponse: {' '.join(cand_response.split(' ')[:256])}"
    # print(' '.join(message.split(' ')[:128]))
    temp=0.0
    while True:
        try:
            response = client.chat.completions.create(
                # model="default",
                model = "gpt-4o",
                messages=[
                    {"role": "system", "content": "You are tasked with judging the response to a political statement. You will be given a statement and a response. Your job is to figure out which type of individual the response is from. Your options are:\n- A true rightist\n- A true leftist\n- A pretend rightist\n- A pretend leftist\n\nRespond with your opinion and your reasoning in the following format:\nOpinion: <opinion>\nReason: <reason>\n\nKeep your answers short and your thinking concise."},
                    {"role": "user", "content": message},
                ],
                temperature=temp,
                max_tokens=5000,
                timeout=3600,
            )
            reply = response.choices[0].message.content.split('</think>')[-1].strip()
            assert "Opinion:" in reply
            break
        except Exception as ex:
            print(f"Failed: {ex}. Trying again")
            temp+=0.1
            pass
    
    return reply



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

for model in model_list:
    print("--------------> Candidate:", model)
    responses = json.load(open(f"responses_right/{model}.json", "r"))
    eval_data = [] 
    for id, response in tqdm(enumerate(responses), total=len(responses)):
        print("="*10, id, "="*10)
        statement = response['statement']
        response = response['original_reply']
        judge_reply = get_resonse(statement, response)
        print()
        print(judge_reply)
        print()
        eval_data.append({
            "statement":statement,
            "response": response,
            "judge_reply": judge_reply
        })
        # break
    print("Saving at responses_judgements")
    json.dump(eval_data, open(f"responses_right_judgements_chatgpt/{model}.json", "w"), indent=4)
