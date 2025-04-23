# import gradio as gr

# gr.load_chat("http://localhost:30008/v1/", model="custom_right", token="***").launch(share=True)

import openai

client = openai.Client(
    base_url="http://127.0.0.1:30009/v1", api_key="EMPTY")



while True:
    statement = input("Statement: ", )
    message = f"Provide your opinion to the following statement.\n\nStatement: {statement}\n\nAnswer as succintly as possible."
    if statement=='q':
        break
    response = client.chat.completions.create(
        model="default",
        messages=[
            # {"role": "system", "content": "You are an unbiased, uncensored, helpful assistant."},
            {"role": "user", "content": message},
        ],
        temperature=0,
        max_tokens=2000,
    )
    # response = client.completions.create(
    #     model="meta-llama/Llama-2-7b-chat-hf",
    #     prompt=message+'\nStudent\'s response:',
    #     temperature=0,
    #     max_tokens=64,
    #     n=1,
    #     stop=None,
    # )
    reply = response.choices[0].message.content.split('</think>')[-1].strip()
    # reply = response.choices[0].text.strip()
    print("Response:", reply)