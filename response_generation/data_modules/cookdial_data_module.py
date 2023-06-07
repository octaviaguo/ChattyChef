import re
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from dst.constant.cookdial_constant import *


class DataModule(LightningDataModule):
    def __init__(
            self,
            tokenizer_path: str,
            context_window: int = 2,
            max_source_length: int = 128,
            max_recipe_length: int = 386,
            max_history_length: int = 386,
            max_target_length: int = 128,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            model_class_name: str = "t5",
            dataset_name: str = None,
            from_file: str = None,
            preprocessing_num_workers: int = None,
            val_collate: int = 0,
            test_collate: int = 1,
            align_mode: int = 0,
            use_intent: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.tokenizer_path = tokenizer_path
        self.context_window = context_window
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_recipe_length = max_recipe_length
        self.max_history_length = max_history_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.model_class_name = model_class_name
        self.from_file = from_file
        self.dataset_name = dataset_name
        self.preprocessing_num_workers = preprocessing_num_workers

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.dev_data = None
        self.train_data = None
        self.test_data = None
        self.infer_data = None
        self.dataset = None
        self.data_collator = None

        self.val_collate_fn = self.collate if val_collate == 0 else self.collate_for_generation
        self.test_collate_fn = self.collate if test_collate == 0 else self.collate_for_generation

        self.align_mode = align_mode
        self.use_intent = use_intent

    def save_tokenizer(self, tokenizer_path):
        self.tokenizer.save_pretrained(tokenizer_path)

    def setup(self, stage: str = None):
        splits = []
        if stage == "fit" or stage is None:
            splits = ["train", "validation"]
        if stage == "validation":
            splits.append("validation")
        if stage == "test":
            splits.append("test")
        if stage == "test_gold_intent":
            splits.append("test_gold_intent")
        if stage == "predict":
            splits.append("predict")

        self.dataset = {}

        if stage == "fit":
            mapping_func = self.dataset_mapping_function
        else:
            mapping_func = self.dataset_mapping_for_generation_function

        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(self.dataset_name, split=splits)

        column_names = ['Context', 'Response', 'Knowledge', 'Index', 'File', "Intents", "Turn_id", "Completed_step"]
        for i in range(len(splits)):
            if stage == "fit":  # splits[i] in ['train', 'validation']:
                columns = ["input_ids", "labels"]
            else:
                columns = ["sample_idx", "input_ids", "attention_mask", "labels", "prompt_length"]
            lm_dataset = raw_datasets[i].map(
                mapping_func,
                batched=True,
                remove_columns=column_names,
                num_proc=1, # self.preprocessing_num_workers,
                load_from_cache_file=False,
                desc=f"Processing dataset",
            )

            lm_dataset.set_format(type="torch", columns=columns)
            split_name = splits[i] if splits[i] != 'test_gold_intent' else 'test'
            self.dataset[split_name] = lm_dataset

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], shuffle=True, collate_fn=self.collate,
                          batch_size=self.train_batch_size, num_workers=self.preprocessing_num_workers
    )

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], collate_fn=self.val_collate_fn, batch_size=self.eval_batch_size,
                          num_workers=self.preprocessing_num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], collate_fn=self.test_collate_fn, batch_size=self.eval_batch_size,
                          num_workers=self.preprocessing_num_workers)

    def get_num_samples(self):
        return len(self.dataset['train'])

    def collate(self, batch):
        batch_input_ids = []
        batch_labels = []

        for _batch in batch:
            batch_input_ids.append(_batch["input_ids"])
            batch_labels.append(_batch["labels"])

        batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch_labels = pad_sequence(batch_labels, batch_first=True, padding_value=-100)

        features = {"input_ids": batch_input_ids,
                    "labels": batch_labels}
        return features

    def dataset_mapping_function(self, examples):
        contextes = examples['Context']
        responses = examples['Response']
        kbs = examples['Knowledge']
        intents = examples['Intents']
        completed_steps = examples['Completed_step']

        all_inputs = []
        all_lm_labels = []
        # for context, kb, response, user_act, system_act in zip(contextes, kbs, responses, user_acts, system_acts):
        for context, kb, response, intent, completed_step in zip(contextes, kbs, responses, intents, completed_steps):
            context = context.replace(' <|EOS|> ', ' ')
            history_ids = self.tokenizer.encode(context + ' <|Knowledge|> ', add_special_tokens=False)
            recipe_title, ingredients, recipe_steps = kb.split('<|EOS|>')
            recipe_steps = recipe_steps.split('<|STEP|>')
            if self.align_mode == 0:
                instruction_prompt = ' '.join(recipe_steps)
            else:
                if completed_step == 'title' or completed_step.startswith('ing-'):
                    instruction_prompt = ' '.join(recipe_steps)
                elif self.align_mode == 1:
                    current_step_idx = int(completed_step.split('inst-')[1])
                    instruction_prompt = ' '.join(recipe_steps[current_step_idx:])
                else:
                    current_step_idx = int(completed_step.split('inst-')[1])
                    tmp1 = max(0, current_step_idx - 1)
                    tmp2 = min(len(recipe_steps), current_step_idx + 2)
                    instruction_prompt = ' '.join(recipe_steps[tmp1:tmp2])
            kb_prompt = f"Title: {recipe_title}. Ingredients: {ingredients}. Instructions: {instruction_prompt}"
            grounded_kb_ids = self.tokenizer.encode(kb_prompt, add_special_tokens=False)

            if self.use_intent:
                intent_desc = [INTENT_DESCRIPTIONS[_[:-1]] for _ in intent.split(" ")]
                intent_desc = ', '.join(intent_desc)
                intent_prompt = f" [user] want to: {intent_desc}."
                gen_token = self.tokenizer.encode(f"{intent_prompt} {self.tokenizer.eos_token} => [system] ",
                                                  add_special_tokens=False)
                # gen_token = self.tokenizer.encode(" => [system] " + self.tokenizer.eos_token, add_special_tokens=False)
            else:
                gen_token = self.tokenizer.encode(" => [system] " + self.tokenizer.eos_token, add_special_tokens=False)

            response_ids = self.tokenizer.encode(response, add_special_tokens=False)

            if len(grounded_kb_ids) > self.max_recipe_length - len(gen_token):
                grounded_kb_ids = grounded_kb_ids[:self.max_recipe_length - len(gen_token)]
            max_history_length = self.max_source_length - len(grounded_kb_ids) - len(gen_token)

            if len(history_ids) > max_history_length:
                history_ids = history_ids[len(history_ids) - max_history_length:]

            _input = history_ids + grounded_kb_ids + gen_token

            if len(response_ids) > self.max_target_length - 1:
                response_ids = response_ids[:self.max_target_length - 1]

            # input_length = len(_input) + len(response_ids) + 2
            input_ids = _input + response_ids + [self.tokenizer.eos_token_id]
            lm_label = [-100] * (len(_input)) + response_ids + [self.tokenizer.eos_token_id]

            all_inputs.append(input_ids)
            all_lm_labels.append(lm_label)

        model_inputs = {"input_ids": all_inputs,
                        "labels": all_lm_labels}

        return model_inputs

    def collate_for_generation(self, batch):
        max_length = 0
        max_label_length = 0
        for _batch in batch:
            length = len(_batch["input_ids"])
            max_length = max_length if max_length > length else length
            label_length = len(_batch["labels"])
            max_label_length = max_label_length if max_label_length > label_length else label_length

        batch_size = len(batch)
        batch_sample_ids = torch.LongTensor(batch_size).fill_(0)
        batch_input_ids = torch.LongTensor(batch_size, max_length).fill_(self.tokenizer.pad_token_id)
        batch_attention_mask = torch.LongTensor(batch_size, max_length).fill_(0)
        batch_labels = torch.LongTensor(batch_size, max_label_length).fill_(self.tokenizer.pad_token_id)
        batch_prompt_lengths = torch.LongTensor(batch_size).fill_(0)

        for i, _batch in enumerate(batch):
            batch_sample_ids[i] = _batch["sample_idx"]
            length = len(_batch["input_ids"])
            batch_input_ids[i, max_length - length:] = _batch["input_ids"]
            batch_attention_mask[i, -length:] = _batch["attention_mask"]
            batch_prompt_lengths[i] = _batch["prompt_length"]
            label_length = len(_batch["labels"])
            batch_labels[i, :label_length] = _batch["labels"]

        features = {"sample_idx": batch_sample_ids,
                    "input_ids": batch_input_ids,
                    "attention_mask": batch_attention_mask,
                    "prompt_length": batch_prompt_lengths,
                    "labels": batch_labels}
        return features

    def dataset_mapping_for_generation_function(self, examples):
        sample_ids = examples['Index']
        contextes = examples['Context']
        responses = examples['Response']
        kbs = examples['Knowledge']
        intents = examples['Intents']
        completed_steps = examples['Completed_step']

        all_inputs = []
        all_attention_masks = []
        all_lm_labels = []
        all_prompt_lengths = []
        all_sample_ids = []

        for idx, context, kb, response, intent, completed_step in zip(sample_ids, contextes, kbs, responses, intents,
                                                                      completed_steps):
            context = context.replace(' <|EOS|> ', ' ')
            history_ids = self.tokenizer.encode(context + ' <|Knowledge|> ', add_special_tokens=False)
            recipe_title, ingredients, recipe_steps = kb.split('<|EOS|>')
            recipe_steps = recipe_steps.split('<|STEP|>')
            if self.align_mode == 0:
                instruction_prompt = ' '.join(recipe_steps)
            else:
                if completed_step == 'title' or completed_step.startswith('ing-'):
                    instruction_prompt = ' '.join(recipe_steps)
                elif self.align_mode == 1:
                    current_step_idx = int(completed_step.split('inst-')[1])
                    instruction_prompt = ' '.join(recipe_steps[current_step_idx:])
                else:
                    current_step_idx = int(completed_step.split('inst-')[1])
                    tmp1 = max(0, current_step_idx - 1)
                    tmp2 = min(len(recipe_steps), current_step_idx + 2)
                    instruction_prompt = ' '.join(recipe_steps[tmp1:tmp2])
            kb_prompt = f"Title: {recipe_title}. Ingredients: {ingredients}. Instructions: {instruction_prompt}"
            grounded_kb_ids = self.tokenizer.encode(kb_prompt, add_special_tokens=False)

            if self.use_intent:
                intent_desc = [INTENT_DESCRIPTIONS[_[:-1]] for _ in intent.split(" ")]
                intent_desc = ', '.join(intent_desc)
                intent_prompt = f" [user] want to: {intent_desc}."
                gen_token = self.tokenizer.encode(f"{intent_prompt} {self.tokenizer.eos_token} => [system] ",
                                                  add_special_tokens=False)
            else:
                gen_token = self.tokenizer.encode(" => [system] " + self.tokenizer.eos_token, add_special_tokens=False)

            if len(grounded_kb_ids) > self.max_recipe_length - len(gen_token):
                grounded_kb_ids = grounded_kb_ids[:self.max_recipe_length - len(gen_token)]
            max_history_length = self.max_source_length - len(grounded_kb_ids) - len(gen_token)

            if len(history_ids) > max_history_length:
                history_ids = history_ids[len(history_ids) - max_history_length:]

            input_ids = history_ids + grounded_kb_ids + gen_token

            attention_mask = [1]*len(input_ids)
            lm_label = self.tokenizer.encode(response, max_length=self.max_target_length, truncation=True)

            all_inputs.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_lm_labels.append(lm_label)
            all_prompt_lengths.append(len(self.tokenizer.decode(input_ids, skip_special_tokens=True)))
            all_sample_ids.append(idx)

        model_inputs = {"sample_idx": all_sample_ids,
                        "input_ids": all_inputs,
                        "attention_mask": all_attention_masks,
                        "prompt_length": all_prompt_lengths,
                        "labels": all_lm_labels}

        return model_inputs

    def decode_predictions(self, predictions):
        results = []
        for batch in predictions:
            sample_idx, encoded_tokens, prompt_lengths = batch
            batch_results = self.tokenizer.batch_decode(encoded_tokens, skip_special_tokens=True)
            for i, result in enumerate(batch_results):
                results.append([sample_idx[i].item(), result[prompt_lengths[i]:]])
        return results
