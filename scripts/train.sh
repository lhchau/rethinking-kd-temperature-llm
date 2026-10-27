#!/usr/bin/env bash
# Train one model on Dolly.
#
#   bash scripts/train.sh <config> <loss> [tau] [seed] [extra finetune.py args]
#
#   bash scripts/train.sh gpt2-base fkl 3.0            # GPT-2 XL -> GPT-2 Base with FKL at tau = 3
#   bash scripts/train.sh gpt2-base rkl 1.0 10 --on-policy
#   bash scripts/train.sh llama3-3b sft                 # LoRA SFT of the Llama-3.2-3B teacher
#
# <config> is a file in scripts/configs, <loss> one of
# sft fkl rkl symkl js sfkl srkl akl ab drkl. All visible GPUs are used and the
# global batch is kept at 32 by gradient accumulation. The run is saved to
# results/<config>/<loss>-tau<tau>-seed<seed> (results/<config>/sft-seed<seed> for SFT).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -lt 2 ]; then sed -n '2,13p' "$0"; exit 1; fi
CONFIG=$1 LOSS=$2 TAU=${3:-1.0} SEED=${4:-10}
shift $(( $# < 4 ? $# : 4 ))
source "scripts/configs/$CONFIG.sh"

NGPU=${NGPU:-$(python3 -c "import torch; print(torch.cuda.device_count())")}
if (( NGPU < 1 )); then echo "no CUDA GPU visible" >&2; exit 1; fi
GLOBAL_BATCH=32
GRAD_ACC=$(( GLOBAL_BATCH / (BATCH_SIZE * NGPU) ))
if (( GRAD_ACC < 1 || GRAD_ACC * BATCH_SIZE * NGPU != GLOBAL_BATCH )); then
  echo "global batch $GLOBAL_BATCH is not a multiple of BATCH_SIZE=$BATCH_SIZE x $NGPU GPUs" >&2
  exit 1
fi

if [ "$LOSS" = sft ]; then RUN=sft-seed$SEED; else RUN=$LOSS-tau$TAU-seed$SEED; fi
if [[ " $* " == *" --on-policy "* ]]; then RUN=$RUN-onpolicy; fi
SAVE=results/$CONFIG/$RUN

export TOKENIZERS_PARALLELISM=false
torchrun --standalone --nproc_per_node "$NGPU" finetune.py \
  --model-type "$MODEL_TYPE" --model-path "$MODEL" \
  ${TEACHER:+--teacher-model-path "$TEACHER"} ${TEACHER_PEFT:+--teacher-peft-path "$TEACHER_PEFT"} \
  --data-dir "processed_data/dolly/full/$MODEL_TYPE" --dev-num "$DEV_NUM" \
  --max-length "$MAX_LENGTH" --max-prompt-length 256 --num-workers 4 \
  --loss "$LOSS" --kd-temperature "$TAU" --seed "$SEED" \
  --epochs "$EPOCHS" --lr "$LR" --batch-size "$BATCH_SIZE" --gradient-accumulation-steps "$GRAD_ACC" \
  --eval-batch-size "$EVAL_BATCH_SIZE" --save "$SAVE" \
  --deepspeed --deepspeed_config "$DS_CONFIG" \
  $TRAIN_OPTS "$@"
