# Rethinking the Role of Temperature in Large Language Model Distillation (EMNLP 2026)

Official PyTorch implementation of our paper:

**Rethinking the Role of Temperature in Large Language Model Distillation** \
*[Hoang-Chau Luong](https://scholar.google.com/citations?hl=en&user=j4ck-L4AAAAJ), [Lingwei Chen](https://scholar.google.com/citations?user=CnKs0mEAAAAJ&hl=en)* \
Rochester Institute of Technology

Reverse KL (RKL) is usually preferred over forward KL (FKL) for LLM distillation, but that
conclusion comes from comparisons run at KD temperature τ = 1. Temperature softens the teacher
distribution and changes *what* FKL learns, while for RKL it mainly rescales the gradient. Once τ is
tuned, FKL consistently outperforms RKL, and temperature improves a whole family of KL-based
objectives (Sym-KL, JS, SFKL, SRKL, AKL).

Every objective takes the teacher and student logits z and compares p<sup>τ</sup> = softmax(z<sup>t</sup>/τ)
with q<sup>τ</sup> = softmax(z<sup>s</sup>/τ), scaled by τ² (Hinton et al., 2015). See
[`distillm/losses.py`](distillm/losses.py).

## Setup

Experiments ran on 4 × NVIDIA A100 (40 GB) GPUs with CUDA 12. Python 3.9 or newer is required.

```bash
pip install -r requirements.txt
```

Download the benchmarks and checkpoints, and tokenize Dolly, for each model family you need:

```bash
bash scripts/prepare_data.sh gpt2      # also: opt, llama3, qwen3
```

This fetches the instruction-following benchmarks released with MiniLLM into `data/`: Dolly,
Self-Instruct, Vicuna, Super-NaturalInstructions and Unnatural Instructions. It also downloads the
student and teacher checkpoints into `checkpoints/`, and writes the tokenized Dolly training and
validation splits to `processed_data/`. The GPT-2 XL and OPT-6.7B teachers are the Dolly-fine-tuned
checkpoints released by MiniLLM. Llama-3.2 is gated, so accept its license on the Hugging Face Hub
and run `hf auth login` first.

## Training and evaluation

```bash
bash scripts/train.sh <config> <loss> [tau] [seed] [extra args]
bash scripts/eval.sh  <config> <checkpoint>
```

| config | teacher → student | fine-tuning |
|---|---|---|
| `gpt2-base` | GPT-2 XL (1.5B) → GPT-2 Base (120M) | full |
| `gpt2-medium` | GPT-2 XL (1.5B) → GPT-2 Medium (340M) | full |
| `opt-1.3b` | OPT 6.7B → OPT 1.3B | full |
| `llama3-1b` | Llama-3.2-3B → Llama-3.2-1B | LoRA |
| `qwen3-1.7b` | Qwen3-4B → Qwen3-1.7B | LoRA |
| `llama3-3b`, `qwen3-4b` | SFT of the Llama / Qwen teachers | LoRA |

| `--loss` | objective | notes |
|---|---|---|
| `sft` | cross-entropy on the references | no teacher |
| `fkl` | forward KL, KL(p‖q) | |
| `rkl` | reverse KL, KL(q‖p) | |
| `symkl` | Sym-KL, 0.5·FKL + 0.5·RKL | |
| `js` | generalized JS divergence | mixture (1 − λ)p + λq, λ = 0.9 |
| `sfkl` / `srkl` | skewed FKL / RKL (Ko et al., 2024) | skew α = 0.1 (`--skew-alpha`) |
| `akl` | adaptive KL (Wu et al., 2025) | head mass μ = 0.5 |
| `ab` | α-β divergence (Wang et al., 2025) | α = 0.2, β = 0.7, used at τ = 1 |
| `drkl` | diversity-aware RKL (Luong et al., 2026) | `--drkl-alpha 1`, `--drkl-beta` 4 (GPT-2) or 2, used at τ = 1 |

Training follows Appendix A of the paper: a global batch of 32 (gradient accumulation adapts to the
number of visible GPUs) and a maximum length of 512 tokens (1024 for Llama and Qwen). GPT-2 and
OPT train for 20 epochs, with learning rates of 5e-4 for GPT-2 and 5e-5 for OPT. After every epoch
the student is scored on 1,000 held-out Dolly examples (200 for the LoRA models), and the
checkpoint with the best validation ROUGE-L is kept at `results/<config>/<run>/best_rougeL`.

Evaluation samples responses at decoding temperature 1 on the five benchmarks with seeds
10, 20, 30, 40 and 50, and prints the ROUGE-L mean ± std. To run a subset, set
`BENCHMARKS="dolly vicuna"` or `SEEDS=10`.

### Example: GPT-2 XL → GPT-2 Base with FKL at τ = 3

```bash
bash scripts/prepare_data.sh gpt2
bash scripts/train.sh gpt2-base fkl 3.0
bash scripts/eval.sh gpt2-base results/gpt2-base/fkl-tau3.0-seed10/best_rougeL
```

## Reproducing the paper

The temperature grid of Figure 1 and Tables 2 and 5:

```bash
CONFIG=gpt2-base   # gpt2-medium, opt-1.3b, llama3-1b, qwen3-1.7b
for loss in fkl rkl symkl js sfkl srkl akl; do
  for tau in 1.0 1.5 2.0 2.5 3.0 3.5 4.0; do
    bash scripts/train.sh $CONFIG $loss $tau
    bash scripts/eval.sh $CONFIG results/$CONFIG/$loss-tau$tau-seed10/best_rougeL
  done
done
```

The best τ of each objective is selected from {1.5, …, 4.0}. Everything else is unchanged.

| experiment | command |
|---|---|
| Figure 1 (FKL vs. RKL across τ) | the grid above with `fkl rkl` for `gpt2-base`, `gpt2-medium`, `opt-1.3b` |
| Table 1 (on-policy) | `bash scripts/train.sh gpt2-base fkl <tau> 10 --on-policy` (and `rkl`) |
| Tables 2 and 5, Figure 2 | the grid above for `gpt2-base` and `opt-1.3b` |
| baselines AB and DRKL | `bash scripts/train.sh <config> ab` and `bash scripts/train.sh <config> drkl` |
| Tables 6 and 7 | first `bash scripts/train.sh llama3-3b sft` (or `qwen3-4b`), then the grid for `llama3-1b` (`qwen3-1.7b`) |
| teacher rows | `bash scripts/eval.sh gpt2-base checkpoints/gpt2-xl`, `bash scripts/eval.sh opt-1.3b checkpoints/opt-6.7b` |

On-policy training uses DistiLLM's adaptive student-generated outputs: student samples from a
replay buffer are used with an initial probability of 0.5 (`--init-threshold`), and that
probability decays over training.

## Repository layout

```
finetune.py            SFT / distillation training loop with per-epoch validation
evaluate.py            sampling + ROUGE-L on one benchmark
arguments.py           command-line arguments
distillm/losses.py     the distillation objectives
distillm/sampler.py    student sampling for on-policy distillation
distillm/buffer.py     replay buffer of student samples
data_utils/            tokenized training data and evaluation prompts
tools/                 Dolly tokenization, evaluation summary
scripts/               prepare_data.sh, train.sh, eval.sh and one config per teacher-student pair
configs/deepspeed/     DeepSpeed configs (ZeRO-1 fp16 for GPT-2/OPT, ZeRO-2 bf16 for Llama/Qwen)
tests/                 CPU tests: python -m pytest tests
```

## Acknowledgements

This code builds on [MiniLLM](https://github.com/microsoft/LMOps/tree/main/minillm) (data, evaluation
and training infrastructure) and [DistiLLM](https://github.com/jongwooko/distillm) (skewed KL
objectives and on-policy sampling). The work is partially supported by the NSF under grant
CNS-2549995 and by Research Computing at the Rochester Institute of Technology.

## Citation

```bibtex
@inproceedings{luong2026rethinking,
  title     = {Rethinking the Role of Temperature in Large Language Model Distillation},
  author    = {Luong, Hoang-Chau and Chen, Lingwei},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```
