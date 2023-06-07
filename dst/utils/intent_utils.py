import re
import json
import jsonlines


def load_jsonl_file(f_name):
    data = []
    with jsonlines.open(f_name) as reader:
        for obj in reader:
            data.append(obj)
    return data


def load_json_file(f_name):
    with open(f_name) as f:
        data = json.load(f)
    return data


def get_description2intent(intent_desc):
    desc2intent = {}
    for intent in intent_desc:
        desc2intent[intent_desc[intent]] = intent
    return desc2intent


def parse_prediction(text, desc2intent, delimiter=':'):
    if ' [intent] ' in text:
        tmp = text.split(' [intent] ')
    else:
        tmp = text.split(' [intents] ')
    if len(tmp) != 2:
        return ['other']

    input_text, intent_str = tmp[0], tmp[1]
    intents = intent_str.strip().split(' ')

    intent_name_list = []
    for intent in intents:
        pattern = f' {intent}{delimiter}|^{intent}{delimiter}'
        match = re.search(pattern, input_text)
        if not match:
            print(input_text)
            intent_name_list.append('other')
            continue
        end_span = -1
        start_span = match.start()
        if start_span != 0:
            start_span += 1
        for i, c in enumerate(text[start_span + len(intent):]):
            if i == 0:
                if c != delimiter:
                    print(intent)
                    print(text)
                    exit(0)
                continue
            if c.isdigit() or c == '[':
                end_span = start_span + len(intent) + i
                break
        intent_desc = text[start_span + len(intent) + 1:end_span].strip()
        if intent_desc == 'ask for about the ingredients':
            intent_desc = 'ask about the ingredients'
        elif intent_desc == 'ask to explain in more detail':
            intent_desc = 'ask to explain the reason or explain in more detail'
        intent_name = desc2intent[intent_desc.strip()]
        intent_name_list.append(intent_name)
    return intent_name_list
