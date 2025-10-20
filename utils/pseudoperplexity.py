"""
Heuristic pseudoperplexity helpers for encoder-decoder (T5-style) and masked-LM models.
"""
from typing import List
import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel
import math

def t5_pseudoperplexity(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, sequences: List[str], device: torch.device, batch_size: int = 8):
    model.to(device)
    model.eval()
    results = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss.item()
            for _ in batch:
                results.append(math.exp(loss))
    return results

def masked_lm_pseudoperplexity(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, sequences: List[str], device: torch.device, batch_size: int = 8):
    model.to(device)
    model.eval()
    mask_token_id = tokenizer.mask_token_id
    results = []
    import torch.nn.functional as F
    with torch.no_grad():
        for seq in sequences:
            enc = tokenizer(seq, return_tensors="pt")
            input_ids = enc["input_ids"].squeeze(0)
            n = input_ids.shape[0]
            neg_logprob = 0.0
            positions = list(range(n))
            for pos in positions:
                masked = input_ids.clone()
                masked[pos] = mask_token_id
                out = model(masked.unsqueeze(0))
                logits = out.logits[0, pos]
                logp = torch.log_softmax(logits, dim=-1)[input_ids[pos]]
                neg_logprob -= logp.item()
            avg_neg = neg_logprob / n
            results.append(math.exp(avg_neg))
    return results
