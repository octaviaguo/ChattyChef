#!/usr/bin/env python
#  coding=utf-8
#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.

from abc import ABC, abstractmethod
import sys
sys.path.insert(0, '../')
import jsonlines
import json
import copy
import random
import fire
import os
import math

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.meteor_score import meteor_score

CONTEXT_WINDOW = 10
random.seed(0)

class Converter(ABC):

    def __init__(self, filepath) -> None:
        super().__init__()

        self.filepath = filepath

    def convert(self):
        """
        Implement your convert logics in this function
        """
        self.start()
        self.process()
        self.end()
        pass

    def start(self):
        print(f'Start processing {self.__class__.__name__} at {self.filepath}')

    def end(self):
        print(
            f'Finish processing {self.__class__.__name__} at {self.filepath}')

    @abstractmethod
    def process(self):
        """
        Implement your convert logics in this function
        """


class CookDialConverter(Converter):
    def process(self):
        all_data = self.load_data()
        num_conversations = len(all_data)
        shuffle_index_List = list(range(num_conversations))
        random.shuffle(shuffle_index_List)

        train_end_idx = math.floor(0.8*num_conversations)
        val_end_idx = math.floor(0.9*num_conversations)

        split2ids = {
            'train': shuffle_index_List[:train_end_idx],
            'valid': shuffle_index_List[train_end_idx:val_end_idx],
            'test': shuffle_index_List[val_end_idx:]
        }

        for split in split2ids:
            examples = []

            for idx in split2ids[split]:
                data = all_data[idx]
                f_name, conversation = data['File'], data['Conversation']

                history = []
                for utt in conversation:
                    if utt['bot']:
                        speaker = 'system'
                    else:
                        speaker = 'user'
                    utterance = utt['utterance']

                    if not utt['bot']:
                        if len(history) == 0:
                            history_str = 'none'
                        elif len(history) > CONTEXT_WINDOW:
                            history_str = ' '.join(history[-CONTEXT_WINDOW:])
                        else:
                            history_str = ' '.join(history)

                        usr_utterance = f'[{speaker}] {utterance}'
                        example = {
                            "index": len(examples),
                            "file": f_name,
                            "utt_id": utt['utt_id'],
                            "history": history_str,
                            "user_utt": usr_utterance,
                            "intent": json.loads(utt['annotations'])['intent']
                        }
                        examples.append(example)
                    history.append(f'[{speaker}] {utterance}')

            print("Num example of {}: {}".format(split, str(len(examples))))
            with jsonlines.open(
                    '../data/dst/cookingdial/cooking_intent_{}.jsonl'.format(split), mode='w') as writer:
                writer.write_all(examples)

        return super().process()

    def load_data(self):
        all_data = []
        for f_name in os.listdir(self.filepath):
            with open(os.path.join(self.filepath, f_name)) as f:
                data = json.load(f)
            all_data.append({'File': f_name,
                             'Conversation': data['messages']})
        return all_data


def convert(class_name, file_path):
    eval(class_name)(file_path).convert()


def main():
    fire.Fire(convert)


if __name__ == '__main__':
    main()
