import sys
sys.path.append('../')
import argparse
import json
from utils.intent_utils import load_json_file, get_description2intent, parse_prediction
import constant.cookdial_constant as cookdial_constant
import constant.newdataset_constant as newdataset_constant


class Parser:
    @staticmethod
    def parse_incontext_file(input_path, output_path, mode='newdata'):
        data = load_json_file(input_path)
        if mode == 'newdata':
            intent_desc = newdataset_constant.INTENT_DESCRIPTIONS
        else:
            intent_desc = cookdial_constant.INTENT_DESCRIPTIONS

        desc2intent = get_description2intent(intent_desc)
        all_intents = []
        for d in data:
            intent_desc_str = d.strip()
            intent_desc_str = intent_desc_str[:intent_desc_str.find('\n###\n')]
            intents = set()
            for desc in intent_desc_str.split(', '):
                intent = desc2intent.get(desc, 'other')
                intents.add(intent)
            intents = list(intents)
            all_intents.append(intents)

        with open(output_path, 'w') as f:
            json.dump(all_intents, f, indent=2)

    @staticmethod
    def parse_description_file(input_path, output_path, mode='newdata'):
        data = load_json_file(input_path)
        if mode == 'newdata':
            intent_desc = newdataset_constant.INTENT_DESCRIPTIONS
        else:
            intent_desc = cookdial_constant.INTENT_DESCRIPTIONS

        all_intents = []
        desc2intent = get_description2intent(intent_desc)
        for idx, d in enumerate(data):
            intents = parse_prediction(d, desc2intent)
            all_intents.append(intents)

        with open(output_path, 'w') as f:
            json.dump(all_intents, f, indent=2)


def main(args):
    if args.mode == 0:
        Parser.parse_incontext_file(input_path=args.input_path,
                                    output_path=args.output_path,
                                    mode=args.desc_mode)
    elif args.mode == 1:
        Parser.parse_description_file(input_path=args.input_path,
                                      output_path=args.output_path,
                                      mode=args.desc_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--desc_mode", type=str, choices=["newdata", "cookdial"], default="newdata")
    args = parser.parse_args()
    main(args)
