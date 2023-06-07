import re
import random
import jsonlines
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import constant.cookdial_constant as cookdial_constant
import constant.newdataset_constant as newdataset_constant


class IncontextDataset(Dataset):
    def __init__(self, train_path, predict_path, num_examples, context_window, intent2desc, intent2idx, tokenizer, max_length):
        idx2intent = {}
        for intent in intent2idx:
            idx2intent[intent2idx[intent]] = intent

        intent_option_prompt = []
        for idx in range(len(idx2intent)):
            intent = idx2intent[idx]
            intent_option_prompt.append(intent2desc[intent])
        intent_option_prompt = ' | '.join(intent_option_prompt)
        intent_option_prompt = f'Intent options:\n{{{{ {intent_option_prompt} }}}}\n\n'
        tokenized_intent_option_prompt = tokenizer.encode(intent_option_prompt, add_special_tokens=False)

        train_examples = self.load_and_process_data(train_path, context_window, intent2desc)
        predict_examples = self.load_and_process_data(predict_path, context_window, intent2desc, include_intent=False)

        self.data = []

        index_list = list(range(len(train_examples)))
        for i in range(len(predict_examples)):
            tokenized_predict_example = tokenizer.encode(' ' + predict_examples[i], add_special_tokens=False)

            example_ids = index_list  # random.choices(index_list, k=num_examples)
            tokenized_incontext_examples = [tokenizer.encode(train_examples[_] + '\n###\n', add_special_tokens=False) for _ in example_ids]

            max_incontext_length = max_length - len(tokenized_intent_option_prompt) - len(tokenized_predict_example)
            tokenized_incontext_example_list = []
            current_incontext_length = 0
            for incontext_example in tokenized_incontext_examples:
                if current_incontext_length + len(incontext_example) < max_incontext_length:
                    tokenized_incontext_example_list.extend(incontext_example)
                    current_incontext_length += len(incontext_example)

            input_ids = tokenized_intent_option_prompt + tokenized_incontext_example_list + tokenized_predict_example
            attention_mask = [1] * len(input_ids)

            item = {}
            item["sample_idx"] = i
            item["input_ids"] = torch.LongTensor(input_ids)
            item["attention_mask"] = torch.LongTensor(attention_mask)
            item["prompt_length"] = len(tokenizer.decode(input_ids, skip_special_tokens=True))
            self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def load_and_process_data(f_path, context_window, intent2desc, include_intent=True):
        data = []
        with jsonlines.open(f_path) as reader:
            for obj in reader:
                history = obj['history']
                user_utt = obj['user_utt']

                if not user_utt.startswith('[user]'):
                    user_utt = f'[user] {user_utt}'
                if history != 'none':
                    system_turns = [m.start() for m in re.finditer('\[system\]', history)]
                    user_turns = [m.start() for m in re.finditer('\[user\]', history)]
                    turns = system_turns + user_turns
                    turns = sorted(turns, reverse=True)

                    history_start_idx = turns[min(context_window - 1, len(turns) - 1)]
                    conversation = f'{history[history_start_idx:]} {user_utt}'
                else:
                    conversation = user_utt

                intent_desc = None
                if 'intent' in obj:
                    intents = obj['intent']
                    intent_desc = []
                    for intent in intents.split(' '):
                        intent_desc.append(intent2desc[intent[:-1]])
                    intent_desc = ', '.join(intent_desc)

                if include_intent:
                    prompt = f'{conversation} \n The [user] want to: {intent_desc}'
                else:
                    prompt = f'{conversation} \n The [user] want to:'
                data.append(prompt)
        return data


class DataModule(LightningDataModule):
    def __init__(
            self,
            tokenizer_path: str,
            num_incontext_examples: int = 1,
            context_window: int = 1,
            max_source_length: int = 128,
            max_target_length: int = 128,
            eval_batch_size: int = 32,
            train_dataset_path: str = None,
            predict_dataset_path: str = None,
            description: str = None,
            from_file: str = None,
            preprocessing_num_workers: int = None,
            **kwargs,
    ):
        super().__init__()
        self.tokenizer_path = tokenizer_path
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.context_window = context_window
        self.num_incontext_examples = num_incontext_examples

        self.eval_batch_size = eval_batch_size
        self.train_dataset_path = train_dataset_path
        self.predict_dataset_path = predict_dataset_path
        self.preprocessing_num_workers = preprocessing_num_workers

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if description == 'cookdial':
            self.intent_description = cookdial_constant.INTENT_DESCRIPTIONS
            self.intents = cookdial_constant.INTENTS
            self.intent2idx = cookdial_constant.INTENT2IDX
        if description == 'newdataset':
            self.intent_description = newdataset_constant.INTENT_DESCRIPTIONS
            self.intents = newdataset_constant.INTENTS
            self.intent2idx = newdataset_constant.INTENT2IDX

        self.dev_data = None
        self.train_data = None
        self.test_data = None
        self.infer_data = None
        self.dataset = None

    def save_tokenizer(self, tokenizer_path):
        self.tokenizer.save_pretrained(tokenizer_path)

    def setup(self, stage: str = None):
        splits = []
        columns = ["sample_idx", "input_ids", "attention_mask", "labels"]

        self.dataset = IncontextDataset(train_path=self.train_dataset_path,
                                        predict_path=self.predict_dataset_path,
                                        num_examples=self.num_incontext_examples,
                                        context_window=self.context_window,
                                        intent2desc=self.intent_description,
                                        intent2idx=self.intent2idx,
                                        tokenizer=self.tokenizer,
                                        max_length=self.max_source_length)

    def predict_dataloader(self):
        return DataLoader(self.dataset, collate_fn=self.collate_fn, batch_size=self.eval_batch_size,
                          num_workers=self.preprocessing_num_workers)

    def get_num_samples(self):
        return len(self.dataset['train'])

    def collate_fn(self, batch):
        max_length = 0
        for _batch in batch:
            length = len(_batch["input_ids"])
            max_length = max_length if max_length > length else length

        batch_size = len(batch)
        batch_sample_ids = torch.LongTensor(batch_size).fill_(0)
        batch_input_ids = torch.LongTensor(batch_size, max_length).fill_(self.tokenizer.pad_token_id)
        batch_attention_mask = torch.LongTensor(batch_size, max_length).fill_(0)
        batch_prompt_lengths = torch.LongTensor(batch_size).fill_(0)

        for i, _batch in enumerate(batch):
            batch_sample_ids[i] = _batch["sample_idx"]
            length = len(_batch["input_ids"])
            batch_input_ids[i, max_length - length:] = _batch["input_ids"]
            batch_attention_mask[i, -length:] = _batch["attention_mask"]
            batch_prompt_lengths[i] = _batch["prompt_length"]

        features = {"sample_idx": batch_sample_ids,
                    "input_ids": batch_input_ids,
                    "attention_mask": batch_attention_mask,
                    "prompt_length": batch_prompt_lengths}
        return features

    def decode_predictions(self, predictions):
        results = []
        for batch in predictions:
            sample_idx, encoded_tokens, prompt_lengths, _ = batch
            batch_results = self.tokenizer.batch_decode(encoded_tokens, skip_special_tokens=True)
            for i, result in enumerate(batch_results):
                # generated_text = result[prompt_lengths[i]:]
                # end_char_idx = generated_text.find('\n')
                # generated_text = generated_text[:end_char_idx]
                results.append([sample_idx[i].item(), result[prompt_lengths[i]:]])
            # results.extend(batch_results)
        return results
