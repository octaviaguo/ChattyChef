WORK_DIR="/path/to/work/directory"  # <-- edit here
MODEL="t5-3b"

# TASK:
# - newdata_16shot_intent : ChattyChef 16 shot
# - newdata_128shot_intent : ChattyChef 128 shot
TASK="newdata_128shot_intent"

timestamp=$(date +%d-%m-%Y_%H-%M-%S)

EXP_NAME="cross_cookdial_ours_16shot"
GROUNDED_CHECKPOINT="t5-3b"
PREV_CHECKPOINT="/path/to/best_checkpoint.pt"
OUTPUT_DIR="${WORK_DIR}/checkpoints/${TASK}/${MODEL}/${EXP_NAME}_${timestamp}"

mkdir -p "${OUTPUT_DIR}"

python main.py \
                --model_name_or_path ${GROUNDED_CHECKPOINT} \
                --dataset_name ./dataset_loader/${TASK}_dataset.py  \
                --output_dir ${OUTPUT_DIR} \
                --strategy deepspeed_stage_2_offload \
                --train_batch_size=4 \
                --eval_batch_size=8 \
                --max_source_length 1024 \
                --max_target_length 32 \
                --num_train_epochs 30 \
                --val_check_interval 1. \
                --description newdataset \
                --context_window 4 \
                --preprocessing_num_workers 24 \
                --num_beams 5 \
                --precision 32 \
                --learning_rate 1e-5 \
                --gradient_accumulation_steps 1 \
                --eval_patience 5 \
                --do_train \
                --finetune_from_checkpoint ${PREV_CHECKPOINT}