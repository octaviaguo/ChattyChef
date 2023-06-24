WORK_DIR="/path/to/work/directory"  # <-- edit here
MODEL="t5-3b"
TASK="multiwoz_22_intent"  # "sgd_intent2"
timestamp=$(date +%d-%m-%Y_%H-%M-%S)

GROUNDED_CHECKPOINT="t5-3b"
OUTPUT_DIR="${WORK_DIR}/checkpoints/${TASK}/${MODEL}/${timestamp}"

mkdir -p "${OUTPUT_DIR}"

python main.py \
                --model_name_or_path ${GROUNDED_CHECKPOINT} \
                --dataset_name ./dataset_loaders/${TASK}_dataset.py \
                --output_dir ${OUTPUT_DIR} \
                --train_batch_size=2 \
                --eval_batch_size=8 \
                --max_source_length 1024 \
                --max_target_length 32 \
                --num_train_epochs 30 \
                --strategy deepspeed_stage_2_offload \
                --val_check_interval 1. \
                --context_window 4 \
                --preprocessing_num_workers 24 \
                --num_beams 5 \
                --precision 32 \
                --learning_rate 1e-4 \
                --gradient_accumulation_steps 4 \
                --eval_patience 2 \
                --do_train