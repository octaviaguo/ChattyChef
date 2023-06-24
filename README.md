# Improved Instruction Ordering in Recipe-Grounded Conversation
This repo contains the code and the new dataset <img src="images/chattychef.png" alt="Icon" width="30px"> ChattyChef for our ACL 2023 paper: <a href="https://arxiv.org/abs/2305.17280">Improved Instruction Ordering in Recipe-Grounded Conversation</a>.


## Installation
This project uses python 3.7.13
```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -r requirements.txt
```

Install BLEURT by following this instruction <a href="https://github.com/google-research/bleurt#installation"> link </a>


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
# Run the following scripts (look at the script for for details)
../scripts/d3st/finetune_d3st.sh

# Merge checkpoints
python response_generation/utils/merge_checkpoint.py \
                   --saved_checkpoint \path\to\saved\ckpt\folder \
                   --output_path \path\to\output_dir\best_checkpoint.pt
```

* Generate: Look at script/d3st/generate_d3st.sh for more details

### Train on CookDial / ChattyChef

* Download the CookDial dataset from <a href="https://github.com/YiweiJiang2015/CookDial"> link </a>
* Preprocess CookDial

```
cd dst

python convert_cookdial_data.py /path/to/cookdial/dialog_directory
```

* Fine-tuning: 
  - Fine-tune from the scratch: See scripts/dst/finetune_t5dst.sh for more details
  - Fine-tune from a previous checkpoint (X -> ChattyChef): See scripts/dst/finetune_from_x.sh for more details  

```
# Merge checkpoints
python response_generation/utils/merge_checkpoint.py \
                   --saved_checkpoint \path\to\saved\ckpt\folder \
                   --output_path \path\to\output_dir\best_checkpoint.pt
```

* Predicting: To predict intents of ChattyChef test set, see scripts/dst/generate_dst.sh for more details

## Instruction State Tracking

Todos

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
                   --saved_checkpoint \path\to\saved\ckpt\folder \
                   --output_path \path\to\output_dir\best_checkpoint.pt
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