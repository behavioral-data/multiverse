from transformers import EvalPrediction
from rouge import Rouge

import numpy as np
from scipy.special import softmax

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from nltk.translate.gleu_score import corpus_gleu

from functools import partial
from typing import NamedTuple
import logging
logger = logging.getLogger(__name__)

from src.models.CORAL_BART.utils import safe_decode, safe_decode_coral

from tokenize import tokenize, untokenize, NUMBER, STRING, NAME, OP
from io import BytesIO


def get_avg_scores(hyps, refs):
    rouge = Rouge(metrics=["rouge-l"])
    scores = {m: {s: 0 for s in rouge.stats} for m in rouge.metrics}
    if rouge.return_lengths:
        scores["lengths"] = {"hyp": 0, "ref": 0}

    count = 0
    for (hyp, ref) in zip(hyps, refs):
        # hyp = [" ".join(_.split()) for _ in hyp.split(".") if len(_) > 0]
        # ref = [" ".join(_.split()) for _ in ref.split(".") if len(_) > 0]
        # hyp = hyp.split()
        # ref = ref.split()
        
        for m in rouge.metrics:
            fn = Rouge.AVAILABLE_METRICS[m]
            sc = fn(hyp, ref, exclusive=rouge.exclusive)
            scores[m] = {s: scores[m][s] + sc[s] for s in rouge.stats}


        count += 1
    avg_scores = {
        m: {s: scores[m][s] / count for s in rouge.stats}
        for m in rouge.metrics
    }

    return avg_scores

def find_all_between_tags(lst, start_tag, end_tag):
    search_from = 0
    try:
        while True:
            start_index = lst.index(start_tag, search_from)
            end_index = lst.index(end_tag, start_index + 1) 
            yield lst[start_index + 1:end_index]
            search_from = end_index + 1
    except ValueError:
        pass

def insert_space_to_tokenized_code(string):
    try:
        g = tokenize(BytesIO(string.encode('utf-8')).readline)

        result = []

        for toknum, tokval, _, _, _ in g:
            result.append(tokval)
        result = result[1:]
    except:
        result = string.split()
    return ' '.join(result)


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def ids_to_space_sep_tokens(ids,tokenizer):
    token_str = " ".join(tokenizer.convert_ids_to_tokens(ids,skip_special_tokens=True))
    if len(token_str) == 0:
        return "<EMPTY>"
    return token_str
    
def calc_rouge_from_tokens(pred_tokens,label_tokens, tokenizer):
    predictions  = [ids_to_space_sep_tokens(x,tokenizer) for x in pred_tokens]
    labels = [ids_to_space_sep_tokens(x,tokenizer) for x in label_tokens]
    return get_avg_scores(predictions,labels)

def calc_gleu_from_tokens(pred_tokens, label_tokens, tokenizer):
    predictions = [tokenizer.convert_ids_to_tokens(x, skip_special_tokens=True) for x in pred_tokens]
    labels = [[tokenizer.convert_ids_to_tokens(x, skip_special_tokens=True)] for x in label_tokens]
    return corpus_gleu(labels,predictions)

def remove_ids(ids):
    return [x for x in ids if not x in [-100]]

def calc_span_aware_rouge(prediction_ids,label_ids, tokenizer):
    # What if theres nothing in the span?
    # What if theres no span? 
    start_token = tokenizer.convert_tokens_to_ids(["<INSERTED>"])[0]
    end_token = tokenizer.convert_tokens_to_ids(["</INSERTED>"])[0]

    all_pred_tokens = []
    all_label_tokens = []

    for pred, label in zip(prediction_ids,label_ids):
        pred_spans = list(find_all_between_tags(pred,start_token,end_token))
        label_spans = list(find_all_between_tags(label,start_token,end_token))

        # In cases where there's just a deletion, the correct result is 
        # an empty span, which looks like this:
        # if len(label_spans) == 0:
        #     label_spans = [[end_token]]
        # if len(pred_spans) == 0:
        #     pred_spans = [[end_token]]

        pred_tokens = [item for sublist in pred_spans for item in sublist]
        label_tokens = [item for sublist in label_spans for item in sublist]

        all_pred_tokens.append(pred_tokens)
        all_label_tokens.append(label_tokens)

    return calc_rouge_from_tokens(all_pred_tokens,all_label_tokens,tokenizer)

def remove_masked_ids(ids, masks):
    span_start_ids = [i for i, v in enumerate(
        masks) if masks[i] == 1 and (i == 0 or masks[i - 1] == 0)]
    span_end_ids = [i for i, v in enumerate(
        masks) if i > 0 and masks[i] == 0 and masks[i - 1] == 1]
    assert len(span_start_ids) == len(span_end_ids)

    spans = []
    spans = [ids[s:e] for s, e in zip(span_start_ids, span_end_ids)]
    return spans


def get_seq2seq_eval(tokenizer, coral=False, idx2word=None, word2idx=None, span_aware_rouge=True):
    if coral:
        assert idx2word is not None and word2idx is not None

    def seq2seq_eval(tokenizer, predictions: NamedTuple, span_aware_rouge=span_aware_rouge):
        scores = {}
        # rouge = Rouge()

        prediction_ids = list(map(remove_ids, predictions.predictions))
        label_ids = list(map(remove_ids, predictions.label_ids))

        # Rouge scores
        rogue_scores = calc_rouge_from_tokens(predictions.predictions, predictions.label_ids, tokenizer)
        
        logger.info("Full Rouge:")
        logger.info(rogue_scores)
        scores["rouge-l-p"] = rogue_scores["rouge-l"]["p"]
        scores["rouge-l-f"] = rogue_scores["rouge-l"]["f"]
        scores["rouge-l-r"] = rogue_scores["rouge-l"]["r"]

        scores["gleu"] = calc_gleu_from_tokens(predictions.predictions, predictions.label_ids, tokenizer)
        
        if span_aware_rouge:
            span_aware_rouge = calc_span_aware_rouge(predictions.predictions ,predictions.label_ids, tokenizer)
            logger.info("Span Aware Rouge:") 
            logger.info(span_aware_rouge)
            scores["span_aware_rouge_l_p"] = span_aware_rouge["rouge-l"]["p"]
            scores["span_aware_rouge_l_r"] = span_aware_rouge["rouge-l"]["r"]
            scores["span_aware_rouge_l_f"] = span_aware_rouge["rouge-l"]["f"]

        if not coral:
            prediction_tokens = [tokenizer.convert_ids_to_tokens(
                x, skip_special_tokens=True) for x in prediction_ids]
            label_tokens = [tokenizer.convert_ids_to_tokens(
                x, skip_special_tokens=True) for x in label_ids]
            avg_edit_distance = np.mean([levenshteinDistance(
                a, b) for a, b in zip(prediction_tokens, label_tokens)])
            scores["avg_edit_distance"] = avg_edit_distance

        return scores

    return partial(seq2seq_eval, tokenizer)


def get_multitask_eval(tokenizer, coral=False, idx2word=None, word2idx=None, wandb=False, threshold=0.1):
    if coral:
        assert idx2word is not None and word2idx is not None
    seq2seq_eval = get_seq2seq_eval(
        tokenizer, coral=False, idx2word=None, word2idx=None)

    def multitask_eval(wandb, predictions):

        results = seq2seq_eval(predictions)

        input_ids = np.array([x for y in predictions.input_ids for x in y])
        pad_mask = input_ids != 1

        input_labels = np.array(
            [x for y in predictions.input_labels for x in y])
        input_logits = np.array(
            [x for y in predictions.input_logits for x in y])

        # Remove indices with pad tokens:

        input_labels = input_labels[pad_mask]
        input_logits = input_logits[pad_mask]
        input_probs = softmax(input_logits, axis=1)
        classes = (input_probs[:, -1] >= threshold)[input_labels != -100]

        results["classification_precision"] = precision_score(
            input_labels, classes)
        results["classification_recall"] = recall_score(input_labels, classes)
        results["classification_accuracy"] = accuracy_score(
            input_labels, classes)
        results["classification_f1"] = f1_score(input_labels, classes)
        results["classification_roc_auc"] = roc_auc_score(input_labels,input_probs[:,-1])
        return results

    return partial(multitask_eval, wandb)


def classification_eval(predictions):
    input_ids = np.array([x for y in predictions.input_ids for x in y])
    pad_mask = input_ids != 1

    input_labels = np.array(
        [x for y in predictions.input_labels for x in y])
    input_logits = np.array(predictions.predictions)

    # Remove indices with pad tokens:

    input_labels = input_labels[pad_mask]
    input_logits = np.array(
            [x for y in predictions.predictions for x in y])[pad_mask]

    input_probs = softmax(input_logits, axis=1)
    classes = (input_probs[:, -1] >= 0.15)
    
    results = {}
    results["classification_precision"] = precision_score(
        input_labels, classes)
    results["classification_recall"] = recall_score(input_labels, classes)
    results["classification_accuracy"] = accuracy_score(
        input_labels, classes)
    results["classification_f1"] = f1_score(input_labels, classes)
    results["roc_auc"] = roc_auc_score(input_labels,input_probs[:,-1])

    return results


