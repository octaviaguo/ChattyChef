import argparse
import constant.cookdial_constant as cookdial_constant
import constant.newdataset_constant as newdataset_constant
from utils.intent_utils import load_jsonl_file, load_json_file, get_description2intent, parse_prediction
import numpy as np
from sklearn.metrics import f1_score


def evaluate_from_file(args, intent_description, intent2idx):
    desc2intent = get_description2intent(intent_description)
    num_intents = len(desc2intent)

    with open(args.gold_file) as f:
        gold_data = f.read().splitlines()

    gold_intents = np.zeros((len(gold_data), num_intents), dtype=int)
    for idx, line in enumerate(gold_data):
        intents = line.strip().split(' ')
        for intent in intents:
            gold_intents[idx, intent2idx[intent[:-1]]] = 1

    with open(args.prediction_file) as f:
        pred_data = f.read().splitlines()
    pred_intents = np.zeros((len(pred_data), num_intents), dtype=int)

    for idx, line in enumerate(pred_data):
        line = line.replace('[', '')
        line = line.replace(']', '')
        line = line.replace('\'', '')
        line = line.replace(',', '')
        line = line.strip()
        intents = line.split(' ')
        for intent in intents:
            pred_intents[idx, intent2idx[intent]] = 1
    print(f1_score(gold_intents, pred_intents, average='micro'))


def evaluate_incontext_learning(args, intent_description, intent2idx):
    desc2intent = get_description2intent(intent_description)
    num_intents = len(desc2intent)

    gold_data = load_jsonl_file(args.gold_file)
    gold_intents = np.zeros((len(gold_data), num_intents), dtype=int)
    for idx, d in enumerate(gold_data):
        intents = d['intents'].strip().split('; ')
        for intent in intents:
            gold_intents[idx, intent2idx[intent]] = 1

    pred_data = load_json_file(args.prediction_file)
    pred_intents = np.zeros((len(gold_data), num_intents), dtype=int)
    # for idx, intents in enumerate(pred_data):
    #     for intent in intents:
    #         pred_intents[idx, intent2idx[intent]] = 1
    # print(f1_score(gold_intents, pred_intents, average='micro'))

    for idx, d in enumerate(pred_data):
        intent_desc_str = d.strip()
        intent_desc_str = intent_desc_str[:intent_desc_str.find('\n###\n')]
        for desc in intent_desc_str.split(', '):
            intent = desc2intent.get(desc, 'other')
            pred_intents[idx, intent2idx[intent]] = 1
    print(f1_score(gold_intents, pred_intents, average='micro'))


def main(args):
    if args.description == 'cookdial':
        intent_description = cookdial_constant.INTENT_DESCRIPTIONS
        intent2idx = cookdial_constant.INTENT2IDX
    else:
        intent_description = newdataset_constant.INTENT_DESCRIPTIONS
        intent2idx = newdataset_constant.INTENT2IDX

    if args.mode == 0:
        desc2intent = get_description2intent(intent_description)

        num_intents = len(desc2intent)
        gold_data = load_jsonl_file(args.gold_file)
        gold_intents = np.zeros((len(gold_data), num_intents), dtype=int)
        for idx, d in enumerate(gold_data):
            if 'intent' in d:
                raw_intents = d['intent'].split(' ')
            else:
                raw_intents = d['intents'].split('; ')

            for raw_intent in raw_intents:
                if raw_intent.endswith(';'):
                    raw_intent = raw_intent[:-1]
                intent_idx = intent2idx[raw_intent]
                gold_intents[idx, intent_idx] = 1

        pred_data = load_json_file(args.prediction_file)
        pred_intents = np.zeros((len(gold_data), num_intents), dtype=int)
        for idx, d in enumerate(pred_data):
            _pred_intents = parse_prediction(d, desc2intent, delimiter=args.delimiter)
            for pred_intent in _pred_intents:
                pred_intents[idx, intent2idx[pred_intent]] = 1
            if idx == len(gold_data) - 1:
                break
        print(f1_score(gold_intents, pred_intents, average='micro'))
    elif args.mode == 1:
        evaluate_from_file(args, intent_description, intent2idx)
    elif args.mode == 2:
        evaluate_incontext_learning(args, intent_description, intent2idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_file", required=True, type=str)
    parser.add_argument("--prediction_file", required=True, type=str)
    parser.add_argument("--delimiter", type=str, default=':')
    parser.add_argument("--mode", type=int)
    parser.add_argument("--description", type=str, choices=['cookdial', 'newdataset'])
    args = parser.parse_args()

    main(args)
