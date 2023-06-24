WORK_DIR="/path/to/work/directory"  # <-- edit here
MODEL="t5-3b"
TASK="cooking_intent"
# timestamp=$(date +%d-%m-%Y_%H-%M-%S)

GROUNDED_CHECKPOINT="t5-3b"
OUTPUT_DIR="${WORK_DIR}/predictions/${TASK}/${MODEL}/"
BEST_CHECKPOINT="path/to/best_checkpoint.pt"
TOKENIZER_PATH="${GROUNDED_CHECKPOINT}"

mkdir -p "${OUTPUT_DIR}"

python main.py \
                --model_name_or_path ${GROUNDED_CHECKPOINT} \
                --best_checkpoint ${BEST_CHECKPOINT} \
                --tokenizer_path ${TOKENIZER_PATH} \
                --dataset_name ../data/dst/converted_data/converted_test_data.jsonl \
                --output_dir ${OUTPUT_DIR} \
                --train_batch_size=1 \
                --eval_batch_size=4 \
                --max_source_length 1024 \
                --max_target_length 32 \
                --num_train_epochs 15 \
                --description cookdial \
                --context_window 4 \
                --preprocessing_num_workers 7 \
                --num_beams 5 \
                --learning_rate 3e-5 \
                --gradient_accumulation_steps 1 \
                --eval_patience 5 \
                --do_predict \
                --prefix test_ \
                --strategy deepspeed_stage_2_offload \
                --precision 32