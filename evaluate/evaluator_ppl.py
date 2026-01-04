# ppl_calculator.py
# -*- coding: utf-8 -*-
"""
纯困惑度计算器（仅依赖 transformers + torch）
用法：
    from ppl_calculator import compute_ppl
    ppl_list, avg_ppl = compute_ppl(model, tokenizer, src_list, tgt_list)
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import List


class _PPLDataset(Dataset):
    """(src, tgt) 对 -> 模型输入"""
    def __init__(self, src: List[str], tgt: List[str], tokenizer,
                 max_src_len: int = 512, max_tgt_len: int = 128):
        assert len(src) == len(tgt), "src 与 tgt 长度必须一致"
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.src = src
        self.tgt = tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        s = str(self.src[idx])
        t = str(self.tgt[idx])
        src_enc = self.tokenizer(
            s,
            max_length=self.max_src_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        tgt_enc = self.tokenizer(
            t,
            max_length=self.max_tgt_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            "input_ids": src_enc["input_ids"].squeeze(0),
            "attention_mask": src_enc["attention_mask"].squeeze(0),
            "labels": tgt_enc["input_ids"].squeeze(0)
        }


@torch.no_grad()
def compute_ppl(model, tokenizer, src_list: List[str], tgt_list: List[str],
                batch_size: int = 8, max_src_len: int = 512,
                max_tgt_len: int = 128, device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = _PPLDataset(src_list, tgt_list, tokenizer,
                          max_src_len, max_tgt_len)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    ppl_list = []
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # 1. 前向
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_dict=True)

        # 2. 手动逐 token 交叉熵
        logits = outputs.logits               # (B, T, V)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()          # 下一位预测
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id,
                                             reduction='none')
        # (B, T-1)
        nll_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                             shift_labels.view(-1)).view(shift_labels.size())

        # 3. 每条平均
        valid_tokens = (shift_labels != tokenizer.pad_token_id).sum(dim=1)  # (B,)
        nll_sample   = nll_token.sum(dim=1) / valid_tokens                 # (B,)
        ppl_batch    = nll_sample.exp()                                    # (B,)
        ppl_list.extend(ppl_batch.cpu().tolist())

    return ppl_list, np.mean(ppl_list)