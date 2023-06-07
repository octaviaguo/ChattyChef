import re
import argparse
import json


PATTERN = r'^(\d)+=| (\d)+='
CATEGORICAL_PATTERN = r' (\d)+[a-z]\)'


class Parser:
    @staticmethod
    def parse_one_sample(src, tgt, schema, desc2slot):
        end_desc_idx = src.find('[user]')
        assert end_desc_idx > 0
        desc = src[:end_desc_idx].strip()

        idx2slot = {}
        i = 0

        desc_start_ids = [m.start() for m in re.finditer(PATTERN, desc)] + [len(desc)]
        for i in range(len(desc_start_ids) - 1):
            slot_span = desc[desc_start_ids[i]:desc_start_ids[i + 1]].strip()
            tmp = slot_span.split('=')
            assert len(tmp) == 2
            index = tmp[0]
            service, second_part = tmp[1].split('-')

            end_idx = len(second_part)
            for j in range(len(second_part)):
                if second_part[j].isalpha() or second_part[j] == ' ':
                    continue
                end_idx = j
                break

            slot_desc = second_part[:end_idx].strip()
            slot_name = desc2slot[slot_desc]

            item = {
                'slot_name': slot_name,
                'possible_values': {}
            }

            if schema[service][slot_name]['is_categorical']:
                category_start_ids = [m.start() for m in re.finditer(CATEGORICAL_PATTERN, slot_span)] + [len(slot_span)]
                for j in range(len(category_start_ids) - 1):
                    category_span = slot_span[category_start_ids[j]:category_start_ids[j + 1]].strip()

                    tmp2 = category_span.split(' ')
                    category_index = tmp2[0][:-1]
                    category_value = ' '.join(tmp2[1:])
                    item['possible_values'][category_index] = category_value

            idx2slot[index] = item

        assert tgt.startswith('[states] ')
        start_idx = 9
        end_idx = tgt.find('[intents]')

        results = {
            'intents': None,
            'slots': []
        }
        predict_slots = tgt[start_idx:end_idx].strip()
        slot_start_idx = [m.start() for m in re.finditer(PATTERN, predict_slots)] + [len(predict_slots)]

        for i in range(len(slot_start_idx) - 1):
            predict_slot_span = predict_slots[slot_start_idx[i]:slot_start_idx[i + 1]].strip()
            tmp = predict_slot_span.split('=')
            slot_index = tmp[0]
            slot_value = predict_slot_span[len(slot_index) + 1:]
            slot_name = idx2slot[slot_index]['slot_name']

            pattern = f'({slot_index}[a-z] )|({slot_index}[a-z]$)'
            # if len(idx2slot[slot_index]['possible_values']) > 0:
            if re.match(pattern, slot_value):
                if slot_value not in idx2slot[slot_index]['possible_values']:
                    slot_values = [slot_value]
                else:
                    slot_values = [idx2slot[slot_index]['possible_values'][slot_value]]
            elif ' | ' in slot_value:
                slot_values = slot_value.split(' | ')
            else:
                slot_values = [slot_value]
            results['slots'].append({
                'slot_name': slot_name,
                'slot_values': slot_values
            })
        return results

    @staticmethod
    def parse(data, schema, desc2slot):
        formatted_data = []

        all_services = list(schema.keys())

        idx2dialogue = {}
        for d in data:
            dialogue_id = d['dialog_id']
            turn_id = int(d['turn'])
            src = d['src']
            tgt = d['tgt']
            if dialogue_id not in idx2dialogue:
                idx2dialogue[dialogue_id] = []
            idx2dialogue[dialogue_id].append([turn_id, src, tgt])

        for dialogue_id in idx2dialogue:
            dialogue = idx2dialogue[dialogue_id]
            dialogue = sorted(dialogue, key=lambda x: x[0])

            item = {
                "dialogue_id": dialogue_id,
                "services": None,
                "turns": []
            }
            active_services = set()
            for turn in dialogue:
                turn_id, src, tgt = turn
                last_user_idx = src.rfind('[user]')
                utterance = src[last_user_idx + 6:].strip()
                predict_results = Parser.parse_one_sample(src=src, tgt=tgt, schema=schema, desc2slot=desc2slot)
                service2slots = {}
                active_services_in_turn = set()
                for slot in predict_results['slots']:
                    slot_name = slot['slot_name']
                    service = slot_name.split('-')[0]
                    active_services.add(service)
                    active_services_in_turn.add(service)
                    if service not in service2slots:
                        service2slots[service] = {}
                    service2slots[service][slot_name] = slot['slot_values']

                turn_obj = {
                    "frames": [],
                    "speaker": "USER",
                    "turn_id": str(turn_id - 1),
                    "utterance": utterance
                }
                for service in all_services:
                    frame = {
                        "service": service,
                        "state": {
                            "active_intent": "NONE",
                            "requested_slots": [],
                            "slot_values": {}
                        }
                    }
                    if service in active_services_in_turn:
                        for slot_name in service2slots[service]:
                            frame["state"]["slot_values"][slot_name] = service2slots[service][slot_name]

                    turn_obj["frames"].append(frame)
                item["turns"].append(turn_obj)
            item["services"] = sorted(list(active_services))
            formatted_data.append(item)
        return formatted_data

    @staticmethod
    def parse2(data, schema, desc2slot):
        formatted_data = []
        for d in data:
            turn_id, src, tgt = d['turn'], d['src'], d['tgt']
            predict_results = Parser.parse_one_sample(src=src, tgt=tgt, schema=schema, desc2slot=desc2slot)
            formatted_data.append(predict_results)
        return formatted_data


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

    assert len(gold_data) == len(predict_data)
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
    formatted_predict_data = Parser.parse(formatted_predict_data, service2slots, desc2slot)

    with open(args.output_file, 'w') as f:
        json.dump(formatted_predict_data, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema_path", type=str, required=True)
    parser.add_argument("--slot_description_path", type=str, required=True)
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
