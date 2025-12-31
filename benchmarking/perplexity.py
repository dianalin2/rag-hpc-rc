from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch, math


_model = GPT2LMHeadModel.from_pretrained("gpt2")
_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def calculate_perplexity(text: str) -> float:
    """Return perplexity score for generated text."""
    encodings = _tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = _model(**encodings, labels=encodings["input_ids"])
    loss = outputs.loss
    return math.exp(loss.item())
