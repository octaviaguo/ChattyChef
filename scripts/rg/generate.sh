GROUNDED_CHECKPOINT="EleutherAI/gpt-j-6B"
TOKENIZER_PATH="EleutherAI/gpt-j-6B"  # <-- path to tokenizer if add new token
TASK="combined_cooking"
OUTPUT_DIR="predictions"
BEST_CHECKPOINT=""  # <-- path to best_checkpoint.pt
MODE=0  # 0: GPT-J, 1: GPT-J+cut, 2: GPT-J+ctr

mkdir -p "${OUTPUT_DIR}"


# Use --use_intent to use the User Intent information
python main.py \
                --model_name_or_path ${GROUNDED_CHECKPOINT} \
                --best_checkpoint ${BEST_CHECKPOINT} \
                --tokenizer_path ${TOKENIZER_PATH} \
                --dataset_name ./datasets_loader/${TASK}_dataset.py \
                --output_dir ${OUTPUT_DIR} \
                --train_batch_size=1 \
                --eval_batch_size=1 \
                --max_source_length 1024 \
                --max_target_length 256 \
                --max_recipe_length 512 \
                --max_history_length 756 \
                --align_mode ${MODE} \
                --num_train_epochs 15 \
                --preprocessing_num_workers 1 \
                --num_beams 5 \
                --learning_rate 3e-5 \
                --gradient_accumulation_steps 1 \
                --eval_patience 5 \
                --do_test \
                --prefix test_ \
                --strategy deepspeed_stage_2_offload \
                --precision 16
