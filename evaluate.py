"""Sample responses for one benchmark and score them with ROUGE-L.

Writes <save>/answers.jsonl and <save>/metrics.json.
"""
import json
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import GenerationConfig

from arguments import get_args
from data_utils.prompt_datasets import PromptDataset
from rouge_metric import compute_metrics
from utils import gather_list, get_dtype, get_model, get_tokenizer, initialize, load_ds_config, print_rank

torch.set_num_threads(4)


@torch.no_grad()
def main():
    args = get_args()
    initialize(args)
    device = torch.cuda.current_device()
    tokenizer = get_tokenizer(args)
    dataset = PromptDataset(args, tokenizer, args.data_dir)
    dtype = get_dtype(load_ds_config(args))
    # cast LoRA adapters too, as the DeepSpeed engine does during training
    model = get_model(args, device, dtype, train=False).to(dtype).eval()

    loader = DataLoader(dataset, sampler=DistributedSampler(dataset, shuffle=False),
                        batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=dataset.collate)
    # temperature-1 sampling (Appendix A), with MiniLLM's no-repeat-6-gram constraint
    generation_config = GenerationConfig(do_sample=True, top_p=1.0, top_k=0, temperature=1.0, no_repeat_ngram_size=6,
                                         eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    max_new_tokens = args.max_length - args.max_prompt_length
    if args.eval_max_new_tokens is not None:
        max_new_tokens = min(max_new_tokens, args.eval_max_new_tokens)

    responses = []
    for batch in tqdm(loader, desc=args.data_dir, disable=dist.get_rank() != 0):
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model.generate(**batch, generation_config=generation_config, max_new_tokens=max_new_tokens)
        texts = tokenizer.batch_decode(output[:, batch["input_ids"].size(1):], skip_special_tokens=True)
        responses += [t.split("<|endoftext|>")[0].replace("<n>", "\n").strip() for t in texts]

    responses = gather_list(responses)[:len(dataset)]
    if dist.get_rank() == 0:
        metrics = compute_metrics(responses, dataset.answers)
        print_rank(f"{args.data_dir} | {metrics}")
        with open(os.path.join(args.save, "answers.jsonl"), "w") as f:
            f.writelines(json.dumps({"text": r}) + "\n" for r in responses)
        with open(os.path.join(args.save, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
