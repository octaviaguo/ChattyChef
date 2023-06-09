TASK="combined_cooking"
GROUNDED_CHECKPOINT="EleutherAI/gpt-j-6B"
MODE=0  # 0: GPT-J, 1: GPT-J+cut, 2: GPT-J+ctr
OUTPUT_DIR=""  # <-- add the output path

# Use --use_intent to use the User Intent information
python main.py \
                --model_name_or_path ${GROUNDED_CHECKPOINT} \
                --dataset_name ./datasets_loader/${TASK}_dataset.py \
                --output_dir ${OUTPUT_DIR} \
                --strategy deepspeed_stage_2_offload \
                --precision 16 \
                --train_batch_size=1 \
                --eval_batch_size=2 \
                --max_source_length 1024 \
                --max_target_length 256 \
                --max_recipe_length 512 \
                --max_history_length 768 \
                --align_mode ${MODE} \
                --num_train_epochs 3 \
                --val_check_interval 1. \
                --preprocessing_num_workers 24 \
                --num_beams 5 \
                --learning_rate 1e-5 \
                --gradient_accumulation_steps 4 \
                --eval_patience 1 \
                --do_train