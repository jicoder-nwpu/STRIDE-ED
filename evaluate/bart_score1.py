import torch
import torch.nn as nn
import traceback
import json
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration
from typing import List


class BARTScorer:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint='/sdb/jihongru/empathetic/empatheticdialogue/evaluate/bart-large-cnn'):
        # 初始化模型和分词器
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # 初始化损失函数
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path=None):
        """加载微调后的模型（可选）"""
        if path is None:
            path = 'models/bart.pth'
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def score(self, srcs, tgts, batch_size=4):
        """计算源文本与目标文本的BARTScore"""
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_batch = srcs[i: i + batch_size]
            tgt_batch = tgts[i: i + batch_size]
            try:
                with torch.no_grad():  # 关闭梯度计算，节省内存
                    # 编码源文本和目标文本
                    encoded_src = self.tokenizer(
                        src_batch,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_batch,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )

                    # 移动到指定设备
                    src_ids = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)
                    tgt_ids = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)  # 每个目标文本的有效长度

                    # 模型前向传播（以源文本为输入，目标文本为标签）
                    output = self.model(
                        input_ids=src_ids,
                        attention_mask=src_mask,
                        labels=tgt_ids
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)  # 展平为(batch*seq_len, vocab)
                    
                    # 计算负对数似然损失，并转换为分数
                    loss = self.loss_fct(self.lsm(logits), tgt_ids.view(-1))  # 展平目标文本计算损失
                    loss = loss.view(tgt_ids.shape[0], -1)  # 恢复为(batch, seq_len)
                    loss = loss.sum(dim=1) / tgt_len  # 按有效长度归一化
                    batch_scores = [-x.item() for x in loss]  # 损失取负作为分数（分数越高越好）
                    score_list.extend(batch_scores)

            except RuntimeError as e:
                traceback.print_exc()
                print(f"处理批次失败：源文本={src_batch}，目标文本={tgt_batch}")
                exit(1)
        return score_list

    def multi_ref_score(self, srcs, tgts: List[List[str]], agg="mean", batch_size=4):
        """多参考文本的评分（每个源文本对应多个目标文本）"""
        ref_counts = [len(refs) for refs in tgts]
        if len(set(ref_counts)) > 1:
            raise ValueError("所有样本的参考文本数量必须一致")
        
        ref_num = len(tgts[0])
        score_matrix = []
        for i in range(ref_num):
            curr_tgts = [refs[i] for refs in tgts]
            scores = self.score(srcs, curr_tgts, batch_size)
            score_matrix.append(scores)
        
        if agg == "mean":
            return list(np.mean(score_matrix, axis=0))
        elif agg == "max":
            return list(np.max(score_matrix, axis=0))
        else:
            raise NotImplementedError(f"不支持的聚合方式：{agg}")


def read_json_data(model_json_path, gold_json_path, model_key="output", gold_key="gold"):
    """
    从两个JSON文件中读取模型输出和真实结果（一一对应）
    :param model_json_path: 模型输出JSON文件路径
    :param gold_json_path: 真实结果JSON文件路径
    :param model_key: 模型输出在JSON中的键名（如"prediction"）
    :param gold_key: 真实结果在JSON中的键名（如"reference"）
    :return: 模型输出列表、真实结果列表
    """
    # 读取模型输出
    with open(model_json_path, 'r', encoding='utf-8') as f:
        model_data = json.load(f)
    if not isinstance(model_data, list):
        raise ValueError("模型输出JSON必须是列表格式（每个元素为一个样本）")
    
    # 读取真实结果
    with open(gold_json_path, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
    if not isinstance(gold_data, list):
        raise ValueError("真实结果JSON必须是列表格式（每个元素为一个样本）")
    
    # 提取文本并检查数量一致性
    model_outputs = [sample.get(model_key, "").strip() for sample in model_data]
    gold_standards = [sample.get(gold_key, "").strip() for sample in gold_data]
    
    if len(model_outputs) != len(gold_standards):
        raise ValueError(
            f"样本数量不匹配：模型输出{len(model_outputs)}条，真实结果{len(gold_standards)}条"
        )
    
    return model_outputs, gold_standards


if __name__ == "__main__":
    # --------------------------
    # 配置参数（根据你的实际情况修改）
    # --------------------------
    MODEL_JSON_PATH = "test.json"  # 模型输出JSON文件路径
    GOLD_JSON_PATH = "gold.json"  # 真实结果JSON文件路径
    MODEL_KEY = "response"                    # 模型输出在JSON中的键名（如"pred"）
    GOLD_KEY = "gold"                       # 真实结果在JSON中的键名（如"ref"）
    BATCH_SIZE = 8                          # 批量大小（根据设备内存调整）
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"  # 设备（GPU/CPU）

    # --------------------------
    # 执行流程
    # --------------------------
    print(f"使用设备：{DEVICE}")
    
    # 1. 读取JSON数据
    print("读取JSON文件...")
    model_outputs, gold_standards = read_json_data(
        model_json_path=MODEL_JSON_PATH,
        gold_json_path=GOLD_JSON_PATH,
        model_key=MODEL_KEY,
        gold_key=GOLD_KEY
    )
    print(f"成功读取{len(model_outputs)}条样本")

    # 2. 初始化BART评分器
    print("加载BART模型...（首次运行会下载模型，可能需要几分钟）")
    scorer = BARTScorer(device=DEVICE)

    # 3. 计算BARTScore
    print("计算BARTScore...")
    scores = scorer.score(
        srcs=model_outputs,
        tgts=gold_standards,
        batch_size=BATCH_SIZE
    )

    # 4. 输出结果
    print("\n===== 评分结果 =====")
    print(f"样本总数：{len(scores)}")
    print(f"前5个样本的分数：{[round(s, 4) for s in scores[:5]]}...")
    print(f"平均BARTScore：{np.mean(scores):.4f}")
    print(f"分数标准差：{np.std(scores):.4f}")
    print(f"最高分：{np.max(scores):.4f}")
    print(f"最低分：{np.min(scores):.4f}")