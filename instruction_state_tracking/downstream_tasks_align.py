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

WINDOW_SIZE = 10
FEW_SHOT_PERCENTAGE = None
random.seed(0)
ALL_DIALOG_ACT = {'answer question', 'ask question', 'other', 'ask to move on', 'answer question', 'teach new step'}


class Converter(ABC):
    def __init__(self, filepath, silver_alignment_path, gold_alignment_path) -> None:
        super().__init__()

        self.filepath = filepath
        self.silver_alignment_path = silver_alignment_path
        self.gold_alignment_path = gold_alignment_path

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


def load_data(f_path, split):
    conversations = []
    recipes = []
    f_name_list = []
    for f_name in os.listdir(os.path.join(f_path, split)):
        absolute_f_path = os.path.join(f_path, split, f_name)
        with open(absolute_f_path) as f:
            data = json.load(f)

        if len(data['conversation']) <= 2:
            print("skip: {}/{}".format(split, f_name))
            continue

        turns = []
        for turn in data['conversation']:
            if 'dialog_act' not in turn or turn['dialog_act'] == '':
                dialog_act = 'other'
            else:
                dialog_act = turn['dialog_act']
            assert dialog_act in ALL_DIALOG_ACT
            turns.append([turn['id'], turn['text'], dialog_act])
        conversations.append(turns)

        steps = []
        for step in data['recipe']['procedure']:
            steps.append([step['summary'], step['description']])
        recipes.append(steps)

        f_name_list.append(f_name)
    return f_name_list, conversations, recipes


class CookingRecipeConverter(Converter):
    def process(self):
        splits = ['train', 'val', 'test']
        with open(self.silver_alignment_path) as f:
            silver_alignments = json.load(f)
        with open(self.gold_alignment_path) as f:
            gold_alignments = json.load(f)

        for split in splits:
            f_name_list, conversations, recipes = load_data(self.filepath, split)

            examples = []
            for f_name, conversation, recipe in zip(f_name_list, conversations, recipes):
                silver_alignment_one_file = silver_alignments[split][f_name]
                gold_alignment_one_file = gold_alignments[split][f_name]

                recipe_steps = []
                for step_idx, step in enumerate(recipe):
                    recipe_steps.append(' <|step|> '.join(step))
                recipe_prompt = ' <|EOS|> '.join(recipe_steps)

                example = {}
                history = []
                current_step_idx = 0
                dialog_acts = []
                current_response_idx = 0
                for i, turn in enumerate(conversation):
                    speaker = turn[0]
                    text = turn[1]
                    dialog_act = turn[2]

                    if speaker == 'Agent' and i > 0:
                        response = text
                        if i > 0:
                            if len(history) > WINDOW_SIZE:
                                example['Context'] = ' <|EOS|> '.join(history[-WINDOW_SIZE:])
                            else:
                                example['Context'] = ' <|EOS|> '.join(history)
                        else:
                            example['Context'] = 'none'

                        example['File'] = f_name
                        example['Index'] = len(examples)
                        example['Knowledge'] = recipe_prompt
                        example['Response'] = response.strip()
                        example['Turn_id'] = i
                        if current_response_idx == 0:
                            example['Current_step_idx'] = 0  # response_idx2step_idx[0]
                            example['Next_step_idx'] = silver_alignment_one_file[0]
                            example['Gold_current_step_idx'] = 0
                            example['Gold_next_step_idx'] = gold_alignment_one_file[0]
                        else:
                            example['Current_step_idx'] = silver_alignment_one_file[current_response_idx - 1]
                            example['Next_step_idx'] = silver_alignment_one_file[current_response_idx]
                            example['Gold_current_step_idx'] = gold_alignment_one_file[current_response_idx - 1]
                            example['Gold_next_step_idx'] = gold_alignment_one_file[current_response_idx]

                        if len(dialog_acts) > 0:
                            example['User_act'] = dialog_acts[-1]
                        else:
                            example['User_act'] = 'None'
                        example['System_act'] = dialog_act

                        examples.append(copy.deepcopy(example))
                        example = {}
                    if speaker == 'Agent':
                        current_response_idx += 1

                    if speaker == 'User':
                        speaker = 'user'
                    elif speaker == 'Agent':
                        speaker = 'system'

                    history.append(f'[{speaker}] {text.strip()}')
                    dialog_acts.append(dialog_act)
            if split == 'val':
                split = 'valid'
            print("Num example of {}: {}".format(split, str(len(examples))))
            with jsonlines.open('../data/cooking_v4/cooking_{}.jsonl'.format(split), mode='w') as writer:
                    writer.write_all(examples)

        return super().process()


def convert(class_name, file_path, silver_alignment_path, gold_alignment_path):
    eval(class_name)(file_path, silver_alignment_path, gold_alignment_path).convert()


def main():
    fire.Fire(convert)


if __name__ == '__main__':
    main()
