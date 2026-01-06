import json
from evaluate import evaluator_bleu_dist_rouge
import re
from typing import Optional
import traceback, nltk
from transformers import AutoTokenizer


def extract_answer(text):
    answer = text.split("<trend>")[1]
    answer = answer.split("</trend>")[0]
    return answer.strip()


# # ---------- 配置 ----------
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

# 别名->标准 映射（继续补充）
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

# 仍按「标准词长度降序」做正则，避免短串截断
STD_KEYS = sorted(SUB_2_CAT.keys(), key=len, reverse=True)
pattern = re.compile('|'.join(map(re.escape, STD_KEYS)), flags=re.IGNORECASE)

def extract_sub(text: str) -> str | None:
    m = pattern.search(text)
    if m:
        key = m.group(0).lower()
        return ALIAS_2_STD.get(key, key)   # 命中标准词
    # === 兜底：别名表再扫一次（从长到短） ===
    for alias in sorted(ALIAS_2_STD.keys(), key=len, reverse=True):
        if alias in text.lower():
            return ALIAS_2_STD[alias]
    return None

EMOTION_PATTERN = re.compile(r"<Emotion>\s*([^\n<]+)", re.IGNORECASE)
def extract_emotion_from_thinking(thinking: str) -> Optional[str]:
    """从thinking字段中提取情绪"""
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
    predict_emotion = predict_emotion.strip('.')
    predict_emotion = predict_emotion.strip()
    emotion = extract_emotion_from_thinking(ground_truth)
    emotion = emotion.lower()
    emotion = emotion.strip(".")
    emotion = emotion.strip("]")
    emotion = emotion.strip("[")
    emotion = emotion.strip()
    if emotion in predict_emotion or predict_emotion in emotion:
        return 1
    return 1 if emotion == predict_emotion else 0

def strategy_reward(predict_str: str, strategy: str) -> float:
    predict_strategy = extract_sub(predict_str)
    if predict_strategy is None:
        return 0
    predict_strategy = predict_strategy.lower()
    strategy = strategy.lower()
    return 1 if strategy == predict_strategy else 0

def update_json_file(file_path, new_item):
    """
    读取JSON文件，将其内容解析为列表，添加新内容后保存回文件
    :param file_path: JSON文件的路径
    :param new_item: 要添加到列表中的新内容
    """
    try:
        # 打开并读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # 确保读取的内容是列表
        if not isinstance(data, list):
            print(f"文件 {file_path} 的内容不是列表格式")
            return
        
        # 向列表中添加新内容
        data.append(new_item)
        
        # 将更新后的列表保存回JSON文件
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        
        print(f"已成功更新文件 {file_path}")
    
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在")
    except json.JSONDecodeError:
        print(f"文件 {file_path} 的内容不是有效的JSON格式")
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")

import metric
def clac_metric(decoder_preds, decoder_labels, no_glove=False):
    ref_list = []
    hyp_list = []
    for ref, hyp in zip(decoder_labels, decoder_preds):
        ref = ' '.join(nltk.word_tokenize(ref.lower()))
        hyp = ' '.join(nltk.word_tokenize(hyp.lower()))
        if len(hyp) == 0:
            hyp = '&'
        ref_list.append(ref)
        hyp_list.append(hyp)

    from metric import NLGEval
    metric = NLGEval(no_glove=no_glove)
    metric_res, metric_res_list = metric.compute_metrics([ref_list], hyp_list)
    return metric_res

def get_evaluate(path, ex_name):
    records = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:         # 跳过空行
                continue
            records.append(json.loads(line))

    results = []
    grundtruths = []
    inferences = []
    exm_cnt = 0
    emotion_acc = 0
    strategy_acc = 0
    for rec in records:
        exm_cnt += 1
        exm = {}
        exm['conv_id'] = rec['conv_id']
        exm['u_id'] = rec['u_id']
        exm['emotion'] = rec['emotion']
        thinking = rec['thinking']
        grundtruth = rec['response']
        response = grundtruth.split('</thinking>')[1]
        response = response.split('</answer>')[0]
        response = response.split('<answer>')[-1]
        response = response.replace("listener:", '')
        response = response.strip()
        grundtruths.append(response)
        exm['truth_response'] = response
        exm['truth_thinking'] = thinking
        try:
            grundtruth = rec['responses'][0]
            thinking = grundtruth.split('</thinking>')[0]
            thinking = thinking.split('<thinking>')[-1]
            response = grundtruth.split('</thinking>')[1]
            response = response.split('<answer>')[1]
            response = response.split('</answer>')[0]
            response = re.sub(r'\(.*?\)', '', response).strip()
            response = response.replace("listener:", '')
            response = response.strip()
        except Exception as e:
            response = ''
            thinking = ''
        exm['inference_response'] = response
        exm['inference_thinking'] = thinking
        emotion_acc += emotion_reward(exm['inference_thinking'], exm['truth_thinking'])
        
        inferences.append(response)
        results.append(exm)

    result = evaluator_bleu_dist_rouge.dialogue_evaluation(inferences, grundtruths)
    nlgEval_res = clac_metric(inferences, grundtruths)
    print(nlgEval_res)
    result['emotion_acc'] = emotion_acc / exm_cnt
    result['strategy_acc'] = strategy_acc / exm_cnt

    # 竖排打印
    for k, v in result.items():
        print(f'{k:<8}: {v:.6f}')

    with open("qwen3_4b_final_results.json", "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=4)

    update_json_file("./evalue_results.json", {ex_name: result})
    
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入 parquet 文件路径")
    parser.add_argument("--ex_name", required=True)
    args = parser.parse_args()

    get_evaluate(args.input, args.ex_name)