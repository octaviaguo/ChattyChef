import argparse
import json
import os
from metrics import F1Metric, normalize_answer
from datasets import load_metric
from bert_score import score as bertscore
from bleurt import score as bleurtscore
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np


def get_diversity_score(text, ngram=1):
    tokens = word_tokenize(text)

    result = None
    if ngram == 1:
        unique_tokens = set(tokens)
        result = len(unique_tokens)/len(tokens)
    elif ngram == 2:
        bigram = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
        unique_bigram = set(bigram)
        if len(bigram) == 0:
            result = 0
        else:
            result = len(unique_bigram)/len(bigram)
    return result


def get_micro_diversity_score(texts, ngram=1):
    tokens = []
    for text in texts:
        tokens.append(word_tokenize(text))

    result = None
    if ngram == 1:
        all_tokens = []
        for tokens_per_text in tokens:
            all_tokens.extend(tokens_per_text)
        unique_tokens = set(all_tokens)
        result = len(unique_tokens)/len(all_tokens)
    elif ngram == 2:
        bigram = []
        for tokens_per_text in tokens:
            bigram.extend([(tokens_per_text[i], tokens_per_text[i + 1]) for i in range(len(tokens_per_text) - 1)])
        unique_bigram = set(bigram)
        result = len(unique_bigram)/len(bigram)
    return result


def get_length_stats(texts):
    total = 0
    length_list = []
    for text in texts:
        # total += len(word_tokenize(text))
        length_list.append(len(word_tokenize(text)))
    length_list = np.array(length_list)
    return length_list.mean(), length_list.std()


def main(args):
    print("Evaluating:............................")
    data = []
    with open(args.input_file) as f:
        lines = f.read().splitlines()
        for line in lines:
            data.append(json.loads(line))

    with open(args.prediction_file) as f:
        predictions = json.load(f)

    if args.partial_eval:
        data = data[:131]
        predictions = predictions[:131]

    print("****  Num predictions: {} ****".format(str(len(predictions))))
    print("****  Num labels: {} ****".format(str(len(data))))
    print()

    knowledge = []
    gold_responses = []
    for i, d in enumerate(data):
        knowledge.append(d['Knowledge'])
        gold_responses.append(d['Response'])

    if args.remove_prompt_prefix:
        processed_predictions = []
        for i, pred in enumerate(predictions):
            tmp = pred.split(" | ")
            if len(tmp) == 2:
                processed_predictions.append(tmp[1])
            elif len(tmp) == 1:
                processed_predictions.append(pred)
            else:
                print(pred)
                exit(0)

            if i == 0:
                print("{} ---> {}".format(pred, processed_predictions[-1]))
                print()
        predictions = processed_predictions

    print("Bert score:............................")
    _, _, bert_F1 = bertscore(predictions, gold_responses, lang='en')
    print(f"System level F1 score: {bert_F1.mean():.3f}")
    print()

    print("F1 score:............................")
    f1k = 0.
    for kb, response in zip(knowledge, predictions):
        f1k += F1Metric.compute(response, [kb]).value()
    f1k = f1k/len(predictions)

    print("F1K:", f1k)
    print()
    f1r = 0.

    preds = []
    refs = []
    print("BLEU score:............................")
    metric_bleu = load_metric('../utils/bleu_metric.py')

    for response, gold_response in zip(predictions, gold_responses):
        f1r += F1Metric.compute(response, [gold_response]).value()

        _response = normalize_answer(response)
        _gold = normalize_answer(gold_response)

        metric_bleu.add(prediction=_response.split(), references=[_gold.split()])
        preds.append(response)
        refs.append(gold_response)

    hf_bleu = metric_bleu.compute()

    f1r = f1r/len(predictions)
    # bleu = bleu/len(predictions)

    print("Bleurt:..........bleurt...........................")
    bleurt_scorer = bleurtscore.BleurtScorer(args.bleurt_checkpoint)
    bleurt = bleurt_scorer.score(references=refs, candidates=preds)
    bleurt = sum(bleurt)/len(bleurt)

    print("Diversity score:............................")
    unigram_scores = []
    bigram_scores = []

    for response in predictions:
        unigram_scores.append(get_diversity_score(response, 1))
        bigram_scores.append(get_diversity_score(response, 2))

    micro_unigram_score = get_micro_diversity_score(predictions, 1)
    micro_bigram_scores = get_micro_diversity_score(predictions, 2)

    print("F1R:", f1r)
    print("HF Bleu:", hf_bleu)
    print("Bleurt:", bleurt)
    print("Micro Unigram diversity:", micro_unigram_score)
    print("Micro Bigram diversity:", micro_bigram_scores)

    length_mean, length_std = get_length_stats(predictions)
    print("Avg length/std:", length_mean, length_std)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, type=str)
    parser.add_argument("--prediction_file", required=True, type=str)
    parser.add_argument("--prefix", type=str, default='')
    parser.add_argument("--remove_prompt_prefix", action="store_true")
    parser.add_argument("--bleurt_checkpoint", type=str, default=None)
    parser.add_argument("--partial_eval", action="store_true")
    args = parser.parse_args()

    main(args)