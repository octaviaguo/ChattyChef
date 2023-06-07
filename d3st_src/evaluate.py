import re
import argparse
import json
from parser import Parser


class Evaluator:
    @staticmethod
    def compare_slots(hyp_slots, ref_slots):
        hyp_slot2values = {}
        for slot in hyp_slots:
            hyp_slot2values[slot['slot_name']] = slot['slot_values']
        ref_slot2values = {}
        for slot in ref_slots:
            ref_slot2values[slot['slot_name']] = slot['slot_values']

        hyp_services = list(hyp_slot2values.keys())
        ref_services = list(ref_slot2values.keys())

        if set(hyp_services) == set(ref_services):
            final_score = 1.
            for service in hyp_services:
                hyp_slot_value = hyp_slot2values[service][0]
                score = 0.
                for ref_slot_value in ref_slot2values[service]:
                    if hyp_slot_value.lower() == ref_slot_value.lower():
                        score = 1.
                        break
                final_score *= score
        else:
            final_score = 0.
        return final_score

    @staticmethod
    def evaluate(hyps, refs):
        assert len(hyps) == len(refs)
        all_scores = []
        for hyp, ref in zip(hyps, refs):
            turn_score = Evaluator.compare_slots(hyp['slots'], ref['slots'])
            all_scores.append(turn_score)
        return sum(all_scores)/len(all_scores)


def main(args):
    with open(args.schema_path) as f:
        schema = json.loads(f.read().lower())

    service2slots = {}
    for item in schema:
        service = item['service_name']
        service2slots[service] = {}
        for slot in item['slots']:
            slot_name = slot['name']
            is_categorical = slot['is_categorical']

            possible_values = None
            if is_categorical:
                possible_values = slot['possible_values']

            service2slots[service][slot_name] = {
                'possible_values': possible_values,
                'is_categorical': is_categorical
            }

    desc2slot = {}
    with open(args.slot_description_path) as f:
        slot_descriptions_raw = json.loads(f.read().lower())
        slot_descriptions = {}
        for key, val in slot_descriptions_raw.items():
            # To be consistent with the keys from extract_belief_state(), rename
            # "book" slots. e.g. "hotel-book people" -> "hotel-people".
            key = key.replace('book ', '')
            # slot_descriptions.json has a "bus-arriveby" slot that doesn't actually
            # exist.
            if key in ('bus-arriveby', 'bus-people'):
                continue

            slot_descriptions[key] = val

    for key in slot_descriptions:
        desc2slot[slot_descriptions[key][0]] = key

    with open(args.gold_file) as f:
        gold_data = json.load(f)

    with open(args.input_file) as f:
        predict_data = json.load(f)

    formatted_predict_data = []
    for i in range(len(gold_data)):
        predict_text = predict_data[i].split(' <|SEP|> ')[1].strip()
        item = {
            "src": gold_data[i]["src"],
            "tgt": predict_text,
            "dialog_id": gold_data[i]["dialog_id"],
            "turn": gold_data[i]["turn"]
        }
        formatted_predict_data.append(item)
    formatted_predict_data = Parser.parse2(formatted_predict_data, service2slots, desc2slot)
    formatted_gold_data = Parser.parse2(gold_data, service2slots, desc2slot)
    print(Evaluator.evaluate(formatted_predict_data, formatted_gold_data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema_path", type=str, required=True)
    parser.add_argument("--slot_description_path", type=str, required=True)
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)