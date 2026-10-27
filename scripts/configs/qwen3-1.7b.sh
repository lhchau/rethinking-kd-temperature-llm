# Qwen3-4B -> Qwen3-1.7B, LoRA student.
MODEL_TYPE=qwen3
MODEL=checkpoints/Qwen3-1.7B
TEACHER=checkpoints/Qwen3-4B
TEACHER_PEFT=results/qwen3-4b/sft-seed10/best_rougeL
LR=1e-4
EPOCHS=10
BATCH_SIZE=${BATCH_SIZE:-8}
EVAL_BATCH_SIZE=8
MAX_LENGTH=1024
DEV_NUM=200
DS_CONFIG=configs/deepspeed/ds_config_zero2_bf16.json
TRAIN_OPTS="--peft lora --gradient-checkpointing --eval-max-new-tokens 256"
EVAL_CAP=256
