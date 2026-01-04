

import re
# from  import evaluator_bleu_dist_rouge

_SOLUTION_CLIP_CHARS = 300

from transformers import AutoTokenizer
from typing import Optional


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<thinking>.*</thinking><answer>.*</answer>$", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)

    score = 1.0 if (match_result and predict_str.count("</answer>") == 1) else 0

    return score

def extract_answer(text):
    answer = text.split("<trend>")[1]
    answer = answer.split("</trend>")[0]
    return answer.strip()


STRATEGY_WHITELIST = {
    "Exploring Thoughts and Cognition",
    "Exploring Actions and Intentions",
    "Exploring Feelings and Emotions",
    "Restatement or Paraphrasing",
    "Reflection of Feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Positive Reinforcement",
    "Gratitude Prompting",
    "Neutral Validation",
    "Providing Suggestions",
    "Information",
    "Cognitive Reframing",
    "Others"
}

ALIAS_2_STD = {
    "exploring thoughts and cognition": "Exploring Thoughts and Cognition",
    "exploring actions and intentions": "Exploring Actions and Intentions",
    "exploring feelings and emotions": "Exploring Feelings and Emotions",
    "restatement or paraphrasing": "Restatement or Paraphrasing",
    "reflection of feelings": "Reflection of Feelings",
    "self-disclosure": "Self-disclosure",
    "affirmation and reassurance": "Affirmation and Reassurance",
    "positive reinforcement": "Positive Reinforcement",
    "gratitude prompting": "Gratitude Prompting",
    "neutral validation": "Neutral Validation",
    "providing suggestions": "Providing Suggestions",
    "information": "Information",
    "cognitive reframing": "Cognitive Reframing",
    "others": "Others",
}

STD_KEYS = sorted(STRATEGY_WHITELIST, key=len, reverse=True)
pattern = re.compile('|'.join(map(re.escape, STD_KEYS)), flags=re.IGNORECASE)

def extract_sub(text: str) -> str | None:
    m = pattern.search(text)
    if m:
        key = m.group(0).strip()
        std_key = next((k for k in STRATEGY_WHITELIST if k.lower() == key.lower()), None)
        if std_key:
            return std_key
    
    text_lower = text.lower().strip()
    for alias, std_key in ALIAS_2_STD.items():
        if alias in text_lower:
            if std_key in STRATEGY_WHITELIST:
                return std_key
    
    return None

def extract_answer_strategy(predict_str: str) -> str | None:
    ans_pattern = re.compile(r'</thinking><answer>listener:\s*\(([^)]+)\)', re.IGNORECASE)
    match = ans_pattern.search(predict_str)
    if match:
        strategy_text = match.group(1).strip()
        return strategy_text
    return None

def strategy_reward(predict_str: str, strategy: str) -> float:
    predict_strategy_text = extract_answer_strategy(predict_str)
    if not predict_strategy_text:
        return 0.0
    
    predict_std_strategy = extract_sub(predict_strategy_text)
    print(predict_std_strategy)
    if not predict_std_strategy:
        return 0.0
    
    target_std_strategy = extract_sub(strategy)
    if not target_std_strategy:
        return 0.0
    
    return 0 if predict_std_strategy != target_std_strategy else 1

EMOTION_PATTERN = re.compile(r"<Emotion>\s*([^\n<]+)", re.IGNORECASE)
def extract_emotion_from_thinking(thinking: str) -> Optional[str]:
    if not thinking:
        return None
    match = EMOTION_PATTERN.search(thinking)
    if match:
        return match.group(1).strip()
    return None

def emotion_reward(predict_str: str, emotion: str) -> float:
    predict_emotion = extract_emotion_from_thinking(predict_str)
    if predict_emotion is None:
        return 0
    predict_emotion = predict_emotion.lower()
    emotion = emotion.lower()
    return 1 if (emotion == predict_emotion or emotion in predict_emotion or predict_emotion in emotion) else 0

def compute_score(data_source, solution_str, ground_truth, extra_info=None):

    format_score = format_reward(solution_str)
    bleu4 = 0
    return format_score * (1 + emotion_reward(solution_str, extra_info['emotion']) + strategy_reward(solution_str, extra_info['strategy']))
