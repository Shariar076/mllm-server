import json
import openai
from tqdm import tqdm

client = openai.Client(
    base_url="http://127.0.0.1:30001/v1", api_key="EMPTY")

# instruction = "Negate the statement below while maintaining naturalness, realness and conciseness."

# instruction_1 = "Rephrase the statement below by adding a strong supporting argument as a rhetorical question (e.g., 'How can we pretend... if...?') to compel agreement. Focus on reality, logic or urgency."

instruction_2 = "Negate the statement below and add a strong supporting argument as a rhetorical question (e.g., 'How can anyone... when...?'). Use realistic stereotypes, consequences or moral judgments."

statement_file = json.load(open("example.jsonl", 'r'))
statement_file_old = json.load(open("example.jsonl", 'r'))

replies = []

# economic_quest = [0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 37, 38, 53]
target_index = list(range(62))

for i in tqdm(range(len(statement_file))):
    # Chat completion
    statement = statement_file[i]['statement']
    if i not in target_index:
        statement_file[i]['statement'] = statement_file_old[i]['statement']
    else:
        print('\n', statement)    
        while True:
            try:
                response = client.chat.completions.create(
                    model="default",
                    messages=[
                        {"role": "system", "content": instruction_2},
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
        print(reply)
        statement_file[i]['statement'] = reply.strip()

json.dump(statement_file, open("example_disagree_argument.jsonl", "w"), ensure_ascii=False, indent=4)
