from tqdm import tqdm
import torch
from nlp import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
device = 'cuda'
model_id = 'gpt2-large'
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
max_length = model.config.n_positions
stride = 512

def encode(text_list):
    return tokenizer('\n\n'.join(text_list), return_tensors='pt')

def calculate_perplexity(encodings):
    lls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl

def eval(text=None):
    """

    Args:
        text (list): A list of text

    Returns:

    """

    if text is None:
        text = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')['text'][:15]
    encodings = encode(text)
    ppl = calculate_perplexity(encodings)
    print(ppl)
    print(text)
    return ppl

if __name__=='__main__':
    text = ["This is some text I wrote.", "Here is some more text.", "Can you guess what I'm going to say next?"]
    eval(text)
    text = ["This text bad.", "Some more text here it.", "Guess can you I'm next to going say?"]
    eval(text)