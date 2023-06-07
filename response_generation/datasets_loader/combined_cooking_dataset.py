# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
import datasets
import jsonlines


_DESCRIPTION = """\
ChattyChef
"""

_CITATION = """\
ChattyChef
"""

_WEBPAGE = ""


class Cooking(datasets.GeneratorBasedBuilder):
    """Cooking"""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "File": datasets.Value("string"),
                    "Context": datasets.Value("string"),
                    "Response": datasets.Value("string"),
                    "Knowledge": datasets.Value("string"),
                    "Index": datasets.Value("int32"),
                    "Current_step_idx": datasets.Value("int32"),
                    "Next_step_idx": datasets.Value("int32"),
                    "intents": datasets.Value("string")
                }
            ),
            homepage=_WEBPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):

        train_path = '../data/cooking_v4/cooking_train.jsonl'
        validation_path = '../data/cooking_v4/cooking_valid.jsonl'
        test_path = '../data/cooking_v4/cooking_test.jsonl'
        # test_gold_intent_path = '../data/cooking_v4/cooking_test_gold_intent.jsonl'
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                                    "filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={
                                    "filepath": validation_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={
                                    "filepath": test_path}),
            # datasets.SplitGenerator(name="test_gold_intent", gen_kwargs={
            #     "filepath": test_gold_intent_path})
        ]

    def _generate_examples(self, filepath):

        key = 0
        with open(filepath, "r", encoding="utf-8") as reader:

            for item in jsonlines.Reader(reader):
                yield key, item
                key += 1
