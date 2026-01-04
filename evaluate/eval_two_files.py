from evaluator_bleu_dist_rouge import dialogue_evaluation   # 你贴的那份代码
import json

def load_field(path, field):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return [item[field] for item in data]

if __name__ == '__main__':   # ← 加上这一行
    refs  = load_field('gold.json',  'gold')   # 字段名按需改
    preds = load_field('test.json', 'response')

    assert len(preds) == len(refs), f'长度不一致: {len(preds)} vs {len(refs)}'

    result = dialogue_evaluation(preds, refs)
    # 竖排打印
    for k, v in result.items():
        print(f'{k:<8}: {v:.6f}')