import torch
import math
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def tokenize_and_truncate(example: dict,
                          completion_length: int = None,
                          prompt_length: int = None,
                          hf_model_name: str = None,
                          tokenizer = None,
                          model_max_seq_len: int = 4096):
    """take hf dataset entry and preprocess it for completion by a model"""
    assert hf_model_name is not None, "need model name to know whether to adjust wrt special tokens"
    assert "text" in example, "expects 'text' field to be present"
    # tokenize
    inputs = tokenizer.encode(example["text"], return_tensors="pt", truncation=True, max_length=model_max_seq_len)
    example.update({"untruncated_inputs": inputs})

    if (completion_length is not None) and (prompt_length is None):
        # leave at least one token as prefix # FIXME I think plus 1 since 0 is start tok
        slice_length = min(inputs.shape[1]-1, completion_length)
    elif (prompt_length is not None) and (completion_length is None):
        desired_comp_len = (inputs.shape[1]-1) - prompt_length
        slice_length = desired_comp_len if desired_comp_len > 0 else 0
    else:
        raise ValueError((f"Can only tokenize and truncate based on either the desired prompt length or desired completion length,",
                          f" but got completion_length:{completion_length},prompt_length:{prompt_length}"))

    # truncate
    inputs = inputs[:,:inputs.shape[1]-slice_length]
    # logic depending on special tokens for the model
    if "t5" in hf_model_name or "T0" in hf_model_name:
        inputs[0,-1] = 1
    # else: pass
    example.update({"inputs": inputs})
    return example


def compute_ppl_single(input_text = None,
                        output_text = None,
                        oracle_model_name = None,
                        oracle_model = None,
                        oracle_tokenizer = None):

    with torch.no_grad():
        tokd_prefix = tokenize_and_truncate({"text":input_text}, completion_length=0, hf_model_name=oracle_model_name, tokenizer=oracle_tokenizer, model_max_seq_len=oracle_model.config.max_position_embeddings)["inputs"]
        tokd_inputs = tokd_prefix
        # if only want to score the "generation" part we need the suffix tokenization length
        tokd_suffix = tokenize_and_truncate({"text":output_text}, completion_length=0, hf_model_name=oracle_model_name, tokenizer=oracle_tokenizer, model_max_seq_len=oracle_model.config.max_position_embeddings)["inputs"]

        tokd_inputs = tokd_inputs.to(oracle_model.device)
        # make labels, mark if not including all positions
        tokd_labels = tokd_inputs.clone().detach()
        tokd_labels[:,:tokd_labels.shape[1]-tokd_suffix.shape[1]+1] = -100

        outputs = oracle_model(input_ids=tokd_inputs, labels=tokd_labels)
        loss = outputs.loss # avg CE loss all positions (except -100, TODO plz check that this is working correctly)
        ppl = torch.tensor(math.exp(loss))

    return loss.item(), ppl.item()



def calculate_perplexity(text, model, tokenizer, stride=512) -> float:
    """
    Calculate perplexity of a model on given text using Hugging Face's
    pipeline and accelerate.
    Adapted from https://huggingface.co/docs/transformers/en/perplexity

    Args:
        model_id (str): The HF model ID (e.g., 'meta-llama/Meta-Llama-3-8B')
        text (str): The text to calculate perplexity on
        stride (int): The stride to use for calculating perplexity

    Returns:
        float: The perplexity of the model on the given text
    """

    encodings = tokenizer(text, return_tensors="pt")
    max_length = model.config.max_position_embeddings
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride),
                          desc="Calculating perplexity...",
                          leave=False,
                          colour="green"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone().to(model.device)
        # We don’t want the log-likelihood for the tokens we’re just treating
        # as context to be included in our loss, so we can set these targets
        # to -100 so that they are ignored.
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    loss= torch.stack(nlls).mean()
    ppl = torch.exp(loss)
    return loss.item(), ppl.item()

# Load the oracle model for PPL measurement
# oracle_model_name = "facebook/opt-13b"
# print(f"Loading oracle model: {oracle_model_name}")

# oracle_tokenizer = AutoTokenizer.from_pretrained(oracle_model_name)
# oracle_model = AutoModelForCausalLM.from_pretrained(oracle_model_name).to("cuda")
# oracle_model.eval()




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

model_responses = [json.load(open(f"responses/{model}.json", "r")) for model in model_list]

data = []

for i in  range(len(model_list)):
    model_name =  model_repos[i]
    print("--------------> Candidate:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_repos[i])
    model = AutoModelForCausalLM.from_pretrained(model_repos[i]).to("cuda")
    model.eval()

    row = []
    for j in tqdm(range(19)):
        statement = model_responses[i][j]['statement']
        message = f"Provide your opinion to the following statement.\n\nStatement: {statement}\n\nAnswer as succintly as possible."
        # message="You are an individual with a strong right-leaning beliefs. "+ message
        # if 'gemma' not in model_name:
        #     prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # else:
        prompt=message
        loss, ppl = calculate_perplexity(prompt, model, tokenizer)
        row.append(ppl)
    data.append(row)

    del model, tokenizer

pd.DataFrame(data, index=pd.Index(model_list)).T.to_csv("original_response_ppl_own.tsv", sep='\t', index=False)