import numpy as np
import math
from typing import List
from six.moves import xrange  # pylint: disable=redefined-builtin
import re
import more_itertools
from nltk.stem import porter
from nltk import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from collections import defaultdict, Counter, namedtuple
import six
import sacrebleu

def mean(lst):
    return sum(lst) / len(lst)

def _calc_ngram_dict(tokens:List[str], ngram:int, dict_ref=None):
    ngram_dict = defaultdict(int) if dict_ref is None else dict_ref
    total = len(tokens)
    for i in range(0, total - ngram + 1):
        item = tuple(tokens[i:i + ngram])
        ngram_dict[item] += 1
    return ngram_dict

def _calc_cover(cand, gold, ngram):
    cand_dict = _calc_ngram_dict(cand, ngram)
    gold_dict = _calc_ngram_dict(gold, ngram)
    cover = 0
    total = 0
    for token, freq in cand_dict.items():
        if token in gold_dict:
            cover += min(freq, gold_dict[token])
        total += freq
    return cover, total

def _calc_cover_rate(cands, golds, ngram):
    """
    calc_cover_rate
    """
    cover = 0.0
    total = 0.000001
    for cand_tokens, gold_tokens in zip(cands, golds):
        cur_cover, cur_total = _calc_cover(cand_tokens, gold_tokens, ngram)
        cover += cur_cover
        total += cur_total
    return cover / total

def _calc_bp(cands, golds):
    c_count = 0.000001
    r_count = 0.0
    for cand_tokens, gold_tokens in zip(cands, golds):
        c_count += len(cand_tokens)
        r_count += len(gold_tokens)
    bp = 1
    if c_count < r_count:
        bp = math.exp(1 - r_count / c_count)
    return bp



# BLEU 
def calc_corpus_bleu_new(hypothesis, references):
    # hypothesis = [normalize_answer(hyp).split(" ") for hyp in hypothesis]
    # references = [[normalize_answer(ref).split(" ")] for ref in references]
    references = [[gold] for gold in references]
    sf = SmoothingFunction(epsilon=1e-12).method1
    b1 = corpus_bleu(references, hypothesis, weights=(1.0/1.0,), smoothing_function=sf)
    b2 = corpus_bleu(references, hypothesis, weights=(1.0/2.0, 1.0/2.0), smoothing_function=sf)
    b3 = corpus_bleu(references, hypothesis, weights=(1.0/3.0, 1.0/3.0, 1.0/3.0), smoothing_function=sf)
    b4 = corpus_bleu(references, hypothesis, weights=(1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0), smoothing_function=sf)
    return b1, b2, b3, b4

# def calc_corpus_bleu_new(hypothesis, references):
#     """
#     hypothesis : list[str]   模型输出句子
#     references : list[str]   参考句子（单参考）
#     return     : (bleu1, bleu2, bleu3, bleu4)
#     """
#     # sacrebleu 要求 references 放在外层，且一个列表对应一个样本
#     refs_T = [[r] for r in references]   # 转置成 shape (n_sample, 1)
#     # 一次性算出 bleu1~4，smooth='exp' 对应 NLTK 的 method1
#     scores = sacrebleu.corpus_bleu(
#         hypothesis,
#         refs_T,
#         smooth_method='exp',   # 平滑方式与 NLTK method1 最接近
#         # max_ngram_order=4            # 输出 4-gram
#     ).scores                  # 返回 [bleu1, bleu2, bleu3, bleu4]
#     return tuple(scores)      # 保持原接口不变

def calc_sentence_bleu(cands, golds):
    bleu1 = []
    bleu2 = []
    bleu3 = []
    sf = SmoothingFunction().method7
    for hyp, ref in zip(cands, golds):
        try:
            b1 = sentence_bleu([ref], hyp, smoothing_function=sf, weights=[1, 0, 0, 0])
        except ZeroDivisionError:
            b1 = 0.0
        try:
            b2 = sentence_bleu([ref], hyp, smoothing_function=sf, weights=[0.5, 0.5, 0, 0])
        except ZeroDivisionError:
            b2 = 0.0
        try:
            b3 = sentence_bleu([ref], hyp, smoothing_function=sf, weights=[0.34, 0.33, 0.33, 0])
        except ZeroDivisionError:
            b3 = 0.0
        bleu1.append(b1)
        bleu2.append(b2)
        bleu3.append(b3)
    return mean(bleu1), mean(bleu2), mean(bleu3)

# DIST
def calc_corpus_distinct(cands):
    distinct1 = _calc_distinct_ngram(cands, 1)
    distinct2 = _calc_distinct_ngram(cands, 2)
    return distinct1, distinct2

def calc_sentence_distinct(cands):
    distinct1 = mean([_calc_sent_distinct_ngram(c, 1) for c in cands])
    distinct2 = mean([_calc_sent_distinct_ngram(c, 2) for c in cands])
    return distinct1, distinct2

def _calc_distinct_ngram(cands, ngram):
    ngram_total = 0.00001
    ngram_distinct_count = 0.00001
    pred_dict = defaultdict(int)
    for cand_tokens in cands:
        _calc_ngram_dict(cand_tokens, ngram, pred_dict)
    for key, freq in pred_dict.items():
        ngram_total += freq
        ngram_distinct_count += 1
    return ngram_distinct_count / ngram_total

def _calc_sent_distinct_ngram(cand, ngram):
    ngram_total = 0.0000000001
    ngram_distinct_count = 0.0
    ngram_dict = defaultdict(int)
    for i in range(0, len(cand) - ngram + 1):
        item = tuple(cand[i:i + ngram])
        ngram_dict[item] += 1
    for _, freq in ngram_dict.items():
        ngram_total += freq
        ngram_distinct_count += 1
    return ngram_distinct_count / ngram_total


def calc_corpus_f1(cands, golds):
    golden_word_total = 0.00000001
    pred_word_total = 0.00000001
    hit_word_total = 0.00000001
    for response, golden_response in zip(cands, golds):
        common = Counter(response) & Counter(golden_response)
        hit_word_total += sum(common.values())
        golden_word_total += len(golden_response)
        pred_word_total += len(response)
    p = hit_word_total / pred_word_total
    r = hit_word_total / golden_word_total
    f1 = 2 * p * r / (p + r)
    return f1

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    re_art = re.compile(r'\b(a|an|the)\b')
    re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

    def remove_articles(text):
        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()
    if isinstance(s, list):
        s = ' '.join(s)  # 把列表拼接成字符串
    return white_space_fix(remove_articles(remove_punc(lower(s)))).split(' ')


def dialogue_evaluation(ori_cands, ori_golds):
    assert len(ori_cands) == len(ori_golds), f"num cand: {len(ori_cands)}, num gold: {len(ori_golds)}"
    cands = []
    golds = []
    help_tokenize = lambda x: word_tokenize(x.lower())
    for cand, gold in zip(ori_cands, ori_golds):
        cands.append(help_tokenize(cand.lower()))
        golds.append(help_tokenize(gold.lower()))
    cbleu1, cbleu2, cbleu3, cbleu4 = calc_corpus_bleu_new(cands, golds)
    sbleu1, sbleu2, sbleu3 = calc_sentence_bleu(cands, golds)
    cdiv1, cdiv2 = calc_corpus_distinct(cands)
    sdiv1, sdiv2 = calc_sentence_distinct(cands)
    results1 = RougeEvaluator(num_parallel_calls=1, tokenization_fn=normalize_answer).run_evaluation(cands, golds)
    rouge1, rouge2, rougel = results1['rouge1'], results1['rouge2'], results1['rougeL']
    cf1 = calc_corpus_f1(cands, golds)
    result = {
        'cf1': cf1,
        'bleu1': cbleu1,
        'bleu2': cbleu2,
        'bleu3': cbleu3,
        'bleu4': cbleu4,
        'dist1': cdiv1,
        'dist2': cdiv2,
        'rouge1': rouge1,
        'rouge2': rouge2,
        'rougel': rougel
    }
    # result.update(rouge_result)
    result = {k: round(100 * v, 6) for k, v in result.items()}
    return result


def tokenize(text, stemmer):
  """Tokenize input text into a list of tokens.
  This approach aims to replicate the approach taken by Chin-Yew Lin in
  the original ROUGE implementation.
  Args:
    text: A text blob to tokenize.
    stemmer: An optional stemmer.
  Returns:
    A list of string tokens extracted from input text.
  """

  # Convert everything to lowercase.
  text = text.lower()
  # Replace any non-alpha-numeric characters with spaces.
  text = re.sub(r"[^a-z0-9]+", " ", text)

  tokens = re.split(r"\s+", text)
  if stemmer:
    # Only stem words more than 3 characters long.
    tokens = [stemmer.stem(x) if len(x) > 3 else x for x in tokens]

  # One final check to drop any empty or invalid tokens.
  tokens = [x for x in tokens if re.match(r"^[a-z0-9]+$", x)]

  return tokens



class RougeScorer(object):
  """Calculate rouges scores between two blobs of text.
  Sample usage:
    scorer = RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score('The quick brown fox jumps over the lazy dog',
                          'The quick brown dog jumps on the log.')
  """

  def __init__(self, rouge_types, use_stemmer=False, tokenization_fn=None):
    """Initializes a new RougeScorer.
    Valid rouge types that can be computed are:
      rougen (e.g. rouge1, rouge2): n-gram based scoring.
      rougeL: Longest common subsequence based scoring.
    Args:
      rouge_types: A list of rouge types to calculate.
      use_stemmer: Bool indicating whether Porter stemmer should be used to
        strip word suffixes to improve matching. (Only available with default tokenizer)
      tokenization_fn: Function that take string as input, and list of tokens as return
    Returns:
      A dict mapping rouge types to Score tuples.
    """

    self.rouge_types = rouge_types
    self._stemmer = porter.PorterStemmer() if use_stemmer else None
    self._tokenization_fn = tokenization_fn

  def score(self, target, prediction):
    """Calculates rouge scores between the target and prediction.
    Args:
      target: Text containing the target (ground truth) text.
      prediction: Text containing the predicted text.
    Returns:
      A dict mapping each rouge type to a Score object.
    Raises:
      ValueError: If an invalid rouge type is encountered.
    """

    if self._tokenization_fn:
        target_tokens = self._tokenization_fn(target)
        prediction_tokens = self._tokenization_fn(prediction)
    else:
        target_tokens = tokenize(target, self._stemmer)
        prediction_tokens = tokenize(prediction, self._stemmer)
    result = {}

    for rouge_type in self.rouge_types:
      if rouge_type == "rougeL":
        # Rouge from longest common subsequences.
        scores = _score_lcs(target_tokens, prediction_tokens)
      elif re.match(r"rouge[0-9]$", rouge_type):
        # Rouge from n-grams.
        n = int(rouge_type[5:])
        if n <= 0:
          raise ValueError("rougen requires positive n: %s" % rouge_type)
        target_ngrams = _create_ngrams(target_tokens, n)
        prediction_ngrams = _create_ngrams(prediction_tokens, n)
        scores = _score_ngrams(target_ngrams, prediction_ngrams)
      else:
        raise ValueError("Invalid rouge type: %s" % rouge_type)
      result[rouge_type] = scores

    return result

def _create_ngrams(tokens, n):
  """Creates ngrams from the given list of tokens.
  Args:
    tokens: A list of tokens from which ngrams are created.
    n: Number of tokens to use, e.g. 2 for bigrams.
  Returns:
    A dictionary mapping each bigram to the number of occurrences.
  """

  ngrams = Counter()
  for ngram in (tuple(tokens[i:i + n]) for i in xrange(len(tokens) - n + 1)):
    ngrams[ngram] += 1
  return ngrams

class Score(namedtuple("Score", ["precision", "recall", "fmeasure"])):
  """Tuple containing precision, recall, and f-measure values."""


def fmeasure(precision, recall):
  """Computes f-measure given precision and recall values."""

  if precision + recall > 0:
    return 2 * precision * recall / (precision + recall)
  else:
    return 0.0


def _score_lcs(target_tokens, prediction_tokens):
  """Computes LCS (Longest Common Subsequence) rouge scores.
  Args:
    target_tokens: Tokens from the target text.
    prediction_tokens: Tokens from the predicted text.
  Returns:
    A Score object containing computed scores.
  """

  if not target_tokens or not prediction_tokens:
    return Score(precision=0, recall=0, fmeasure=0)

  # Compute length of LCS from the bottom up in a table (DP appproach).
  cols = len(prediction_tokens) + 1
  rows = len(target_tokens) + 1
  lcs_table = np.zeros((rows, cols))
  for i in xrange(1, rows):
    for j in xrange(1, cols):
      if target_tokens[i - 1] == prediction_tokens[j - 1]:
        lcs_table[i, j] = lcs_table[i - 1, j - 1] + 1
      else:
        lcs_table[i, j] = max(lcs_table[i - 1, j], lcs_table[i, j - 1])
  lcs_length = lcs_table[-1, -1]

  precision = lcs_length / len(prediction_tokens)
  recall = lcs_length / len(target_tokens)
  fmeasures = fmeasure(precision, recall)

  return Score(precision=precision, recall=recall, fmeasure=fmeasures)

def _score_ngrams(target_ngrams, prediction_ngrams):
  """Compute n-gram based rouge scores.
  Args:
    target_ngrams: A Counter object mapping each ngram to number of
      occurrences for the target text.
    prediction_ngrams: A Counter object mapping each ngram to number of
      occurrences for the prediction text.
  Returns:
    A Score object containing computed scores.
  """

  intersection_ngrams_count = 0
  for ngram in six.iterkeys(target_ngrams):
    intersection_ngrams_count += min(target_ngrams[ngram],
                                     prediction_ngrams[ngram])
  target_ngrams_count = sum(target_ngrams.values())
  prediction_ngrams_count = sum(prediction_ngrams.values())

  precision = intersection_ngrams_count / max(prediction_ngrams_count, 1)
  recall = intersection_ngrams_count / max(target_ngrams_count, 1)
  fmeasures = fmeasure(precision, recall)

  return Score(precision=precision, recall=recall, fmeasure=fmeasures)


class RougeEvaluator(object):
    """Calculate rouges scores two blobs of single-sentence text by using
    google's python rouge scripts.
    (If you wnat to get sentence-level ROUGE-L, use Rouge155Evaluator)
    Sample usage:
        evaluator = language_evaluation.RougeEvaluator(
            rouge_types=["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        results = evaluator.run_evaluation(
            ['i am a boy', 'she is a girl'],
            ['am i a boy ?', 'is she a girl ?'])
    """
    def __init__(self,
                 num_parallel_calls: int = 1,
                 rouge_types=["rouge1", "rouge2", "rougeL"],
                 use_stemmer=True,
                 tokenization_fn=None,
                 average=True):
        self._num_parallel_calls = num_parallel_calls
        self.rouge_types = rouge_types
        self.use_stemmer = use_stemmer
        self._tokenization_fn = tokenization_fn
        self.average = average

    def run_evaluation(self, predicts, answers):
        n_predicts = self._split_list(predicts, self._num_parallel_calls)
        n_answers = self._split_list(answers, self._num_parallel_calls)
        from multiprocessing import Pool
        p = Pool(self._num_parallel_calls)
        import time
        start = time.time()
        results = p.map(self._run_evaluation, zip(n_predicts, n_answers))
        p.close()
        p.join()
        end = time.time()
        print(f"Takes {end-start} seconds for rouge evaluation with \
              {self._num_parallel_calls} processes")

        # results = self._run_evaluation([predicts, answers])
        # Average results form processes
        averaged_result = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        for result in results:
            for key, value in result.items():
                averaged_result[key].append(value)
        if self.average:
            for key, value in averaged_result.items():
                # TODO : Currently, we assume each process has same numver of
                # predict-answer pairs
                averaged_result[key] = sum(value) / len(value)

        return averaged_result

    def _run_evaluation(self, predicts_and_answers):
        predicts, answers = predicts_and_answers
        scorer = RougeScorer(self.rouge_types, self.use_stemmer, self._tokenization_fn)
        scores = {rouge_type: [] for rouge_type in self.rouge_types}
        for predict, answer in zip(predicts, answers):
            # TODO : support multi-reference
            score = scorer.score(answer, predict)
            for key, value in score.items():
                scores[key].append(value.fmeasure)

        # Averaging
        for key in scores.keys():
            if self.average:
                scores[key] = np.mean(np.array(scores[key]))
            else:
                scores[key] = np.array(scores[key])

        return scores

    def _split_list(self, in_list, num_splits):
        return [list(c) for c in more_itertools.divide(num_splits, in_list)]
