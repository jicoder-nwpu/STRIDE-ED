import torch
import numpy as np

def compute_ppl(model, tokenizer, src_list, tgt_list, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    适配Seq2Seq模型（BART/T5）的PPL计算函数
    :param model: AutoModelForSeq2SeqLM实例
    :param tokenizer: AutoTokenizer实例
    :param src_list: 源文本列表
    :param tgt_list: 目标文本列表
    :return: 每条文本的PPL列表，平均PPL
    """
    model = model.to(device).eval()  # 模型设为评估模式
    tokenizer.padding_side = "right"  # 关键：右填充（适配Seq2Seq）
    ppl_list = []
    
    with torch.no_grad():  # 禁用梯度，节省显存
        for src_text, tgt_text in zip(src_list, tgt_list):
            # 1. 编码源文本（编码器输入）
            src_encoding = tokenizer(
                src_text,
                return_tensors="pt",
                padding="longest",  # 仅填充到当前文本长度（避免多余padding）
                truncation=True,
                max_length=512
            ).to(device)
            
            # 2. 编码目标文本（解码器输入+标签，核心：移位处理）
            tgt_encoding = tokenizer(
                tgt_text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=512
            ).to(device)
            
            # 解码器输入：目标文本去掉最后一个token（前n-1个）
            decoder_input_ids = tgt_encoding.input_ids[:, :-1].contiguous()
            # 标签：目标文本去掉第一个token（后n-1个），且padding掩码为-100（BART忽略-100的标签）
            labels = tgt_encoding.input_ids[:, 1:].contiguous()
            labels = torch.where(labels == tokenizer.pad_token_id, -100, labels)
            
            # 3. 计算loss（BART的loss是交叉熵，已对有效token取均值）
            outputs = model(
                input_ids=src_encoding.input_ids,  # 编码器输入（源文本）
                attention_mask=src_encoding.attention_mask,
                decoder_input_ids=decoder_input_ids,  # 解码器输入（移位后的目标文本）
                labels=labels  # 预测目标（移位+掩码后的标签）
            )
            loss = outputs.loss.item()  # loss是“负对数似然均值”
            
            # 4. 计算PPL：PPL = exp(loss)
            ppl = np.exp(loss)
            ppl_list.append(ppl)
    
    # 计算平均PPL
    avg_ppl = np.mean(ppl_list) if ppl_list else 0.0
    return ppl_list, avg_ppl

# 你的原有代码（替换compute_ppl后运行）
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("/sdb/jihongru/empathetic/empatheticdialogue/evaluate/bart-base")
model     = AutoModelForSeq2SeqLM.from_pretrained("/sdb/jihongru/empathetic/empatheticdialogue/evaluate/bart-base")

src_list = [
    "speaker: Yeah about 10 years ago I had a horrifying experience. It was 100% their fault but they hit the water barrels and survived. They had no injuries but they almost ran me off the road.",
]
tgt_list = [
    "speaker: Yeah about 10 years ago I had a horrifying experience. It was 100% their fault but they hit the water barrels and survived. They had no injuries but they almost ran me off the road.",
]

ppl_list, avg_ppl = compute_ppl(model, tokenizer, src_list, tgt_list)
print("每条 PPL:", [round(p, 2) for p in ppl_list])  # 正常会输出15~40之间的数值
print("平均 PPL:", round(avg_ppl, 2))