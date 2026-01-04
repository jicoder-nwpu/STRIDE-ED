import json
import re
import numpy as np
import torch
import torch.nn as nn
import traceback
from typing import Optional, List
from evaluate import evaluator_bleu_dist_rouge
from transformers import BartTokenizer, BartForConditionalGeneration
import argparse


# BART评分器类
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

                    # 模型前向传播
                    output = self.model(
                        input_ids=src_ids,
                        attention_mask=src_mask,
                        labels=tgt_ids
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    
                    # 计算负对数似然损失，并转换为分数
                    loss = self.loss_fct(self.lsm(logits), tgt_ids.view(-1))
                    loss = loss.view(tgt_ids.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    batch_scores = [-x.item() for x in loss]
                    score_list.extend(batch_scores)

            except RuntimeError as e:
                traceback.print_exc()
                print(f"处理批次失败：源文本={src_batch}，目标文本={tgt_batch}")
                exit(1)
        return score_list

    def multi_ref_score(self, srcs, tgts: List[List[str]], agg="mean", batch_size=4):
        """多参考文本的评分"""
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


# 原有评估工具函数
def extract_answer(text):
    answer = text.split("<trend>")[1]
    answer = answer.split("</trend>")[0]
    return answer.strip()


# 策略分类映射
SUB_2_CAT = {
    "mirroring"              : "情绪回应类",
    "labeling"               : "情绪回应类",
    "soothing"               : "情绪回应类",
    "positive reinforcement" : "情绪回应类",
    "restating"              : "内容回应类",
    "reframing"              : "内容回应类",
    "sharing experience"     : "内容回应类",
    "informational support"  : "内容回应类",
    "collaborative exploration":"行动回应类",
    "concrete suggestion"    : "行动回应类",
    "resource directing"     : "行动回应类",
    "encouraging expression" : "行动回应类",
}

# 别名映射
ALIAS_2_STD = {
    "mirror"                : "mirroring",
    "label"                 : "labeling",
    "encourage"             : "encouraging expression",
    "encourage expression"  : "encouraging expression",
    "mirror the"            : "mirroring",
    "label the"             : "labeling",
    "encourage them"        : "encouraging expression",
    "encourage speaker"     : "encouraging expression",
}

# 策略提取正则
STD_KEYS = sorted(SUB_2_CAT.keys(), key=len, reverse=True)
pattern = re.compile('|'.join(map(re.escape, STD_KEYS)), flags=re.IGNORECASE)

def extract_sub(text: str) -> str | None:
    m = pattern.search(text)
    if m:
        key = m.group(0).lower()
        return ALIAS_2_STD.get(key, key)
    # 兜底：别名表匹配
    for alias in sorted(ALIAS_2_STD.keys(), key=len, reverse=True):
        if alias in text.lower():
            return ALIAS_2_STD[alias]
    return None

# 情绪提取
EMOTION_PATTERN = re.compile(r"<Emotion>\s*([^\n<]+)", re.IGNORECASE)
def extract_emotion_from_thinking(thinking: str) -> Optional[str]:
    if not thinking:
        return None
    match = EMOTION_PATTERN.search(thinking)
    if match:
        return match.group(1).strip()
    return None

def emotion_reward(predict_str: str, ground_truth: str) -> float:
    predict_emotion = extract_emotion_from_thinking(predict_str)
    if predict_emotion is None:
        return 0
    predict_emotion = predict_emotion.lower()
    emotion = extract_emotion_from_thinking(ground_truth)
    if emotion is None:
        return 0
    emotion = emotion.lower()
    emotion = emotion.strip('[')
    emotion = emotion.strip(']')
    emotion = emotion.strip('.')
    if emotion == predict_emotion or emotion in predict_emotion or predict_emotion in emotion:
        return 1
    return 0

def strategy_reward(predict_str: str, ground_truth: str) -> float:
    predict_strategy = extract_sub(predict_str)
    if predict_strategy is None:
        return 0
    predict_strategy = predict_strategy.lower()
    strategy = extract_sub(ground_truth)
    if strategy is None:
        return 0
    strategy = strategy.lower()
    return 1 if strategy == predict_strategy else 0

def update_json_file(file_path, new_item):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        if not isinstance(data, list):
            print(f"文件 {file_path} 的内容不是列表格式")
            return
        
        data.append(new_item)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        
        print(f"已成功更新文件 {file_path}")
    
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在")
    except json.JSONDecodeError:
        print(f"文件 {file_path} 的内容不是有效的JSON格式")
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")

def get_evaluate(path, ex_name, bart_batch_size=8):
    records = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    results = []
    grundtruths = []  # 真实回应列表
    inferences = []   # 模型生成回应列表
    exm_cnt = 0
    emotion_acc = 0
    strategy_acc = 0
    
    for rec in records:
        exm_cnt += 1
        exm = {}
        exm['conv_id'] = rec['conv_id']
        exm['u_id'] = rec['u_id']
        exm['emotion'] = rec['emotion']
        
        # 处理真实结果
        grundtruth = rec['response']
        thinking = grundtruth.split('</thinking>')[0]
        thinking = thinking.split('<thinking>')[-1]
        response = grundtruth.split('</thinking>')[1]
        response = response.split('')[-1]
        response = response.split('')[0]
        response = response.replace("listener:", '')
        response = response.strip()
        exm['truth_response'] = response
        exm['truth_thinking'] = thinking
        grundtruths.append(response)
        
        # 处理模型输出
        try:
            inference_full = rec['responses'][0]
            thinking = inference_full.split('</thinking>')[0]
            thinking = thinking.split('<thinking>')[-1]
            response = inference_full.split('</thinking>')[1]
            response = response.split('')[0]
            response = response.split('')[0]
            response = response.replace("listener:", '')
            response = response.strip()
        except Exception as e:
            print(e)
            response = ''
            thinking = ''
        
        exm['inference_response'] = inference_full
        exm['inference_thinking'] = thinking
        emotion_acc += emotion_reward(exm['inference_thinking'], exm['truth_thinking'])
        strategy_acc += strategy_reward(exm['inference_thinking'], exm['truth_thinking'])
        inferences.append(response)
        results.append(exm)

    # 计算原有评估指标
    result = evaluator_bleu_dist_rouge.dialogue_evaluation(inferences, grundtruths)
    result['emotion_acc'] = emotion_acc / exm_cnt
    result['strategy_acc'] = strategy_acc / exm_cnt

    # 新增BARTScore计算
    print("加载BART模型计算BARTScore...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    bart_scorer = BARTScorer(device=device)
    bart_scores = bart_scorer.score(
        srcs=inferences, 
        tgts=grundtruths, 
        batch_size=bart_batch_size
    )
    
    # 计算BARTScore统计值
    result['bart_score_avg'] = np.mean(bart_scores)
    result['bart_score_std'] = np.std(bart_scores)
    result['bart_score_max'] = np.max(bart_scores)
    result['bart_score_min'] = np.min(bart_scores)

    # 打印所有评估结果
    print("\n===== 完整评估结果 =====")
    for k, v in result.items():
        print(f'{k:<15}: {v:.6f}')

    # 保存详细结果
    with open("qwen3_4b_final_results.json", "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=4)

    # 更新汇总结果
    update_json_file("/sdb/jihongru/empathetic/evalue_results.json", {ex_name: result})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入JSONL文件路径")
    parser.add_argument("--ex_name", required=True, help="实验名称")
    parser.add_argument("--bart_batch_size", type=int, default=8, help="BARTScore计算的批量大小")
    args = parser.parse_args()

    get_evaluate(args.input, args.ex_name, args.bart_batch_size)