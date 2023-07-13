# Improved Instruction Ordering in Recipe-Grounded Conversation
This repo contains the code and the new dataset <img src="images/chattychef.png" alt="Icon" width="30px"> ChattyChef for our ACL 2023 paper: <a href="https://arxiv.org/abs/2305.17280">Improved Instruction Ordering in Recipe-Grounded Conversation</a>.


## Installation
This project uses python 3.7.13
```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -r requirements.txt
```

Install BLEURT by following this instruction <a href="https://github.com/google-research/bleurt#installation"> link </a>


## <img src="images/chattychef.png" alt="Icon" width="30px"> ChattyChef Dataset

You can find the processed dataset at `data/cooking_v4`. The dataset is already splitted to train, validation and test. Each line in a file is a data example, which has the following fields:
* `File`: Name of the file contains this conversation
* `Index`: Index of the example
* `Context`: The conversation history
* `Knowledge`: The grounded recipe of the conversation
* `Response`: The golden response
* `Current_step_idx`: Instruction state of the current turn (output of the Instruction State tracking module)
* `Next_step_idx`: Instruction state of the next turn (output of the Instruction State tracking module)
* `intents`: User intent of the last user utterance (output of the User Intent Detection module)

The file `data/cooking_v4/cooking_test_gold_intent.jsonl` contains the human-annotated user intents. 


## User Intent Detection

### Train on MultiWOZ 2.2 / SGD

* Download the datasets: <a href="https://github.com/budzianowski/multiwoz"> MultiWOZ 2.2 </a>, <a href="https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/tree/master"> SGD </a>
* Preprocess
```
cd d3st_src

# MultiWOZ 2.2
python convert_multiwoz_data.py \
             --input_dir /path/to/downloaded/dataset \
             --output_dir data/dst/multiwoz_22_intent

# SGD
python convert_sgd_data.py \
             --input_dir /path/to/downloaded/dataset \
             --output_dir data/dst/sgd_intent2             
```
* Fine-tuning

```
# Create symlink at d3st_src to point to dst/constant
ln -s absolute_path_to/dst/constant d3st_src

# Run the following scripts (look at the script for for details)
../scripts/d3st/finetune_d3st.sh

# Merge checkpoints
python response_generation/utils/merge_checkpoint.py \
                   --saved_checkpoint /path/to/saved/ckpt/folder \
                   --output_path /path/to/output_dir/best_checkpoint.pt
```

* Generate: Look at script/d3st/generate_d3st.sh for more details

### Train on CookDial / ChattyChef

* Download the CookDial dataset from <a href="https://github.com/YiweiJiang2015/CookDial"> link </a>
* Preprocess CookDial

```
cd dst

python convert_cookdial_data.py CookDialConverter /path/to/cookdial/dialog_directory
```

* Fine-tuning: 
  - Fine-tune from the scratch: See scripts/dst/finetune_t5dst.sh for more details
  - Fine-tune from a previous checkpoint (X -> ChattyChef): See scripts/dst/finetune_from_x.sh for more details  

```
# Merge checkpoints
python response_generation/utils/merge_checkpoint.py \
                   --saved_checkpoint /path/to/saved/ckpt/folder \
                   --output_path /path/to/output_dir/best_checkpoint.pt
```

* Predicting: To predict intents of ChattyChef test set, see scripts/dst/generate_dst.sh for more details

## Instruction State Tracking

Please take a look at `instruction_state_tracking/align.py` for the `WordMatch` and `SentEmb` algorithms.

## Response generation

```
cd response_generation
```

### Fine-tuning
Edit path in finetune_gpt.sh before running
```
# Fine-tune gpt-j model
../scripts/rg/finetune_gpt.sh

# Merge checkpoints
python utils/merge_checkpoint.py \
                   --saved_checkpoint /path/to/saved/ckpt/folder \
                   --output_path /path/to/output_dir/best_checkpoint.pt
```

### Generation
Generate examples from the ChattyChef test set

Edit path in generate.sh before running
```
# Generate (support multi-gpus)
../scripts/rg/generate.sh

# Merge predictions
python utils/merge_predictions.py \
            --input_dir predictions
```

### Evaluation
Download the BLEURT-20 checkpoint from <a href="https://github.com/google-research/bleurt#using-bleurt---tldr-version"> link </a>
```
cd evaluation
python evaluate_cooking.py \
                --input_file ../../data/cooking_v4/cooking_test.jsonl \
                --prediction_file predictions/merged_predictions.json \
                --bleurt_checkpoint /path/to/BLEURT-20
```

### Citation
If you use this codebase in your work, please consider citing our paper:

```
@inproceedings{le-etal-2023-improved,
    title = "Improved Instruction Ordering in Recipe-Grounded Conversation",
    author = "Le, Duong  and
      Guo, Ruohao  and
      Xu, Wei  and
      Ritter, Alan",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.561",
    pages = "10086--10104",
    abstract = "In this paper, we study the task of instructional dialogue and focus on the cooking domain. Analyzing the generated output of the GPT-J model, we reveal that the primary challenge for a recipe-grounded dialog system is how to provide the instructions in the correct order. We hypothesize that this is due to the model{'}s lack of understanding of user intent and inability to track the instruction state (i.e., which step was last instructed). Therefore, we propose to explore two auxiliary subtasks, namely User Intent Detection and Instruction State Tracking, to support Response Generation with improved instruction grounding. Experimenting with our newly collected dataset, ChattyChef, shows that incorporating user intent and instruction state information helps the response generation model mitigate the incorrect order issue. Furthermore, to investigate whether ChatGPT has completely solved this task, we analyze its outputs and find that it also makes mistakes (10.7{\%} of the responses), about half of which are out-of-order instructions. We will release ChattyChef to facilitate further research in this area at: https://github.com/octaviaguo/ChattyChef.",
}
```