import argparse
import os
import json
import jsonlines
import random


def get_intent2description(schema):
    intent2description = {}
    unique_descriptions =set()
    for service in schema:
        for intent in service['intents']:
            intent_desc = intent['description'].lower()
            if intent_desc == 'buy tickets for an event' or intent_desc == 'reserve a selected hotel for given dates':
                intent_desc = f"{service['service_name'].lower()} {intent['description'].lower()}"

            intent2description[intent['name']] = intent_desc
    return intent2description


def convert(original_data_path, output_dir, delimiter=':'):
    splits = ['train', 'dev', 'test']
    for split in splits:
        schema_path = os.path.join(original_data_path, split, 'schema.json')
        with open(schema_path) as f:
            schema = json.load(f)

        intent2description = get_intent2description(schema)
        split_data_path = os.path.join(original_data_path, split)
        examples = []
        for f_name in os.listdir(split_data_path):
            if f_name == 'schema.json':
                continue
            absolute_f_path = os.path.join(split_data_path, f_name)
            with open(absolute_f_path) as f:
                data = json.load(f)
            id2dialogue = {}
            for d in data:
                id2dialogue[d['dialogue_id'].lower()] = d
            dialogue_ids = sorted(list(id2dialogue.keys()))
            for dialogue_id in dialogue_ids:
                dialogue = id2dialogue[dialogue_id]

                turns = dialogue['turns']
                histories = []

                for turn_id, turn in enumerate(turns):
                    speaker = turn['speaker'].lower()
                    histories.append(f"[{speaker}] {turn['utterance']}")

                    if speaker != 'user':
                        continue

                    active_intents = set()
                    for frame in turn['frames']:
                        state = frame['state']
                        if state['active_intent'] != 'NONE':
                            active_intents.add(state['active_intent'])

                    if len(active_intents) == 0:
                        # print(split, f_name, dialogue_id, turn['turn_id'])
                        continue

                    intent_names = list(intent2description.keys())
                    random.shuffle(intent_names)

                    intent_descs = []
                    tgt_intent_ids = []

                    for i, intent_name in enumerate(intent_names):
                        desc = f'{i}{delimiter}{intent2description[intent_name]}'
                        intent_descs.append(desc)

                        if intent_name in active_intents:
                            tgt_intent_ids.append(str(i))

                    desc_prompt = ' '.join(intent_descs)
                    src = ' '.join(histories)
                    tgt_intent_str = ' '.join(tgt_intent_ids)

                    src_prompt = f'{desc_prompt} {src}'
                    tgt_prompt = f'[intents] {tgt_intent_str}'

                    example = {
                        'index': len(examples),
                        'src': src_prompt,
                        'tgt': tgt_prompt,
                        'dialog_id': dialogue_id,
                        'turn': turn_id + 1
                    }
                    examples.append(example)
        split_name = split if split != 'dev' else 'valid'
        absolute_f_outpath = os.path.join(output_dir, f'{split_name}.jsonl')
        print(split, len(examples))
        with jsonlines.open(absolute_f_outpath, 'w') as writer:
            writer.write_all(examples)


def main(args):
    random.seed(0)
    convert(args.input_dir, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    main(args)
