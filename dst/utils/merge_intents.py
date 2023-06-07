import sys
sys.path.append('../')
import argparse
import json
import jsonlines
import os
from utils.intent_utils import load_jsonl_file, load_json_file, get_description2intent, parse_prediction
from constant.newdataset_constant import INTENT_DESCRIPTIONS, INTENT2IDX


def main(args):
    desc2intent = get_description2intent(INTENT_DESCRIPTIONS)

    intent_data = load_json_file(args.intent_file)
    intent_predictions = []
    for idx, d in enumerate(intent_data):
        _pred_intents = parse_prediction(d, desc2intent)
        intent_predictions.append(_pred_intents)

    dialog_data = load_jsonl_file(args.dialogue_file)
    print(len(dialog_data))
    print(len(intent_data))
    assert len(dialog_data) == len(intent_data)
    for i in range(len(dialog_data)):
        data[i]['intents'] = '; '.join(intent_data[i])

    with jsonlines.open(args.output_file, 'w') as writer:
        writer.write_all(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogue_file", required=True, type=str)
    parser.add_argument("--intent_file", required=True, type=str)
    parser.add_argument("--output_file", required=True, type=str)
    args = parser.parse_args()

    main(args)