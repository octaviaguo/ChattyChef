import re
import random
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
import constant.cookdial_constant as cookdial_constant
import constant.newdataset_constant as newdataset_constant


class DataModule(LightningDataModule):
    def __init__(
            self,
            tokenizer_path: str,
            context_window: int = 1,
            max_source_length: int = 128,
            max_target_length: int = 128,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            model_class_name: str = "t5",
            dataset_name: str = None,
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

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.model_class_name = model_class_name
        self.from_file = from_file
        self.dataset_name = dataset_name
        self.preprocessing_num_workers = preprocessing_num_workers

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True,
                                                       model_max_length=max_source_length)
        if description == 'cookdial':
            self.intent_description = cookdial_constant.INTENT_DESCRIPTIONS
            self.intents = cookdial_constant.INTENTS
            self.intent2idx = cookdial_constant.INTENT2IDX
        if description == 'newdataset':
            self.intent_description = newdataset_constant.INTENT_DESCRIPTIONS
            self.intents = newdataset_constant.INTENTS
            self.intent2idx = newdataset_constant.INTENT2IDX

    def save_tokenizer(self, tokenizer_path):
        self.tokenizer.save_pretrained(tokenizer_path)

    def setup(self, stage: str = None):
        splits = []
        dataset_mapping_fnc = self.dataset_mapping_function

        columns = ["sample_idx", "input_ids", "attention_mask", "labels"]

        if stage == "fit" or stage is None:
            splits = ["train", "validation"]
        if stage == "validation":
            splits.append("validation")
        if stage == "test":
            splits.append("test")
        if stage == "predict":
            splits.append("predict")
            dataset_mapping_fnc = self.dataset_mapping_function_for_prediction
            columns = ["sample_idx", "input_ids", "attention_mask"]
        if stage == "predict_cooking":
            splits.append("predict")
            dataset_mapping_fnc = self.dataset_mapping_function_for_cooking_prediction
            columns = ["sample_idx", "input_ids", "attention_mask"]

        self.dataset = {}

        # Downloading and loading a dataset from the hub.
        if stage == 'predict' or stage == 'predict_cooking':
            raw_datasets = load_dataset("json", data_files=self.dataset_name)
            lm_dataset = raw_datasets['train'].map(
                dataset_mapping_fnc,
                batched=True,
                num_proc=1,  # self.preprocessing_num_workers,
                load_from_cache_file=False,
                desc=f"Processing dataset",
            )
            lm_dataset.set_format(type="torch", columns=columns)
            self.dataset['predict'] = lm_dataset
        else:
            raw_datasets = load_dataset(self.dataset_name, split=splits)

            for i in range(len(splits)):
                column_names = ['src', 'tgt', 'dialog_id', 'turn', 'index']

                lm_dataset = raw_datasets[i].map(
                    dataset_mapping_fnc,
                    batched=True,
                    remove_columns=column_names,
                    num_proc=1,  # self.preprocessing_num_workers,
                    load_from_cache_file=False,
                    desc=f"Processing dataset",
                )

                lm_dataset.set_format(type="torch", columns=columns)
                split_name = splits[i] if splits[i] != 'train_full' else 'train'
                self.dataset[split_name] = lm_dataset

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], shuffle=True, collate_fn=self.collate,
                          batch_size=self.train_batch_size, num_workers=self.preprocessing_num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], collate_fn=self.collate, batch_size=self.eval_batch_size,
                          num_workers=self.preprocessing_num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], collate_fn=self.collate, batch_size=self.eval_batch_size,
                          num_workers=self.preprocessing_num_workers)

    def predict_dataloader(self):
        return DataLoader(self.dataset["predict"], collate_fn=self.collate_for_predict, batch_size=self.eval_batch_size,
                          num_workers=self.preprocessing_num_workers)

    def get_num_samples(self):
        return len(self.dataset['train'])

    def collate(self, batch):
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_sample_ids = torch.LongTensor(len(batch)).fill_(0)
        batch_prompt_lengths = torch.zeros(len(batch))

        for i, _batch in enumerate(batch):
            batch_input_ids.append(_batch["input_ids"])
            batch_attention_mask.append(_batch["attention_mask"])
            batch_labels.append(_batch["labels"])
            batch_sample_ids[i] = _batch["sample_idx"]

        batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch_attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)
        batch_labels = pad_sequence(batch_labels, batch_first=True, padding_value=-100)

        features = {"input_ids": batch_input_ids,
                    "attention_mask": batch_attention_mask,
                    "sample_idx": batch_sample_ids,
                    "prompt_length": batch_prompt_lengths,
                    "labels": batch_labels}
        return features

    def dataset_mapping_function(self, examples):
        sample_ids = examples['index']
        src_texts = examples['src']
        tgt_texts = examples['tgt']

        all_sample_ids = []
        all_inputs = []
        all_attention_masks = []
        all_lm_labels = []

        for sample_idx, src, tgt in zip(sample_ids, src_texts, tgt_texts):
            # if src.count('=') == 11:
            #     src = src.replace('=', ':')
            system_turns = [m.start() for m in re.finditer('\[system\]', src)]
            user_turns = [m.start() for m in re.finditer('\[user\]', src)]
            turns = system_turns + user_turns
            turns = sorted(turns, reverse=True)

            label_description = src[:turns[-1]]
            history_start_idx = turns[min(self.context_window, len(turns) - 1)]
            src_text = label_description + src[history_start_idx:]
            # src_text = src

            tokenized_src_text = self.tokenizer(src_text)
            tokenized_tgt_text = self.tokenizer(tgt)

            all_sample_ids.append(sample_idx)
            all_inputs.append(tokenized_src_text.input_ids)
            all_attention_masks.append(tokenized_src_text.attention_mask)
            all_lm_labels.append(tokenized_tgt_text.input_ids)

        model_inputs = {"sample_idx": all_sample_ids,
                        "input_ids": all_inputs,
                        "attention_mask": all_attention_masks,
                        "labels": all_lm_labels}

        return model_inputs

    def collate_for_predict(self, batch):
        batch_input_ids = []
        batch_attention_mask = []
        batch_sample_ids = torch.LongTensor(len(batch)).fill_(0)
        batch_prompt_lengths = torch.zeros(len(batch))

        for i, _batch in enumerate(batch):
            batch_input_ids.append(_batch["input_ids"])
            batch_attention_mask.append(_batch["attention_mask"])
            batch_sample_ids[i] = _batch["sample_idx"]

        batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch_attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)

        features = {"input_ids": batch_input_ids,
                    "attention_mask": batch_attention_mask,
                    "sample_idx": batch_sample_ids,
                    "prompt_length": batch_prompt_lengths}
        return features

    def dataset_mapping_function_for_prediction(self, examples):
        sample_ids = examples['index']
        src_texts = examples['src']
        tgt_text = examples['tgt']

        all_sample_ids = []
        all_inputs = []
        all_attention_masks = []

        for sample_idx, src, tgt in zip(sample_ids, src_texts, tgt_text):
            src_text = src

            tokenized_src_text = self.tokenizer(src_text)

            all_sample_ids.append(sample_idx)
            all_inputs.append(tokenized_src_text.input_ids)
            all_attention_masks.append(tokenized_src_text.attention_mask)

        model_inputs = {"sample_idx": all_sample_ids,
                        "input_ids": all_inputs,
                        "attention_mask": all_attention_masks}

        return model_inputs

    def dataset_mapping_function_for_cooking_prediction(self, examples):
        sample_ids = examples['index']
        histories = examples['history']
        user_utts = examples['user_utt']

        all_sample_ids = []
        all_inputs = []
        all_attention_masks = []

        for sample_idx, history, user_utt in zip(sample_ids, histories, user_utts):
            shuffle_id_list = list(range(len(self.intent_description)))
            random.shuffle(shuffle_id_list)
            intent_id_remap = {}
            for i, shuffle_idx in enumerate(shuffle_id_list):
                intent_id_remap[shuffle_idx] = i

            description_prompt = []
            for i, intent_idx in enumerate(shuffle_id_list):
                _intent = self.intents[intent_idx]
                description_prompt.append(f'{str(i)}:{self.intent_description[_intent]}')
            description_prompt = ' '.join(description_prompt)

            if not user_utt.startswith('[user]'):
                user_utt = f'[user] {user_utt}'
            if history != 'none':
                system_turns = [m.start() for m in re.finditer('\[system\]', history)]
                user_turns = [m.start() for m in re.finditer('\[user\]', history)]
                turns = system_turns + user_turns
                turns = sorted(turns, reverse=True)

                history_start_idx = turns[min(self.context_window - 1, len(turns) - 1)]
                conversation = f'{history[history_start_idx:]} {user_utt}'
            else:
                conversation = f'{user_utt}'

            src_text = f'{description_prompt} {conversation} [intent]'
            tokenized_src_text = self.tokenizer(src_text)

            all_sample_ids.append(sample_idx)
            all_inputs.append(tokenized_src_text.input_ids)
            all_attention_masks.append(tokenized_src_text.attention_mask)

        model_inputs = {"sample_idx": all_sample_ids,
                        "input_ids": all_inputs,
                        "attention_mask": all_attention_masks}

        return model_inputs

    def decode_predictions(self, predictions):
        results = []
        for batch in predictions:
            sample_idx, encoded_tokens, _, input_ids = batch
            batch_results = self.tokenizer.batch_decode(encoded_tokens, skip_special_tokens=True)
            batch_input_ids = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            for i in range(len(batch_results)):
                results.append([sample_idx[i].item(), batch_input_ids[i] + ' <|SEP|> ' + batch_results[i]])
            # results.extend(batch_results)
        return results
