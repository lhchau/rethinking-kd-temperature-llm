import json
import os

import torch
from torch.utils.data import Dataset


class PromptDataset(Dataset):
    """Evaluation prompts and reference answers of one benchmark.

    Reads `<data_dir>/valid_<model_type>.jsonl` when it exists (prompts rendered
    with a chat template by tools/process_data_dolly.py --eval-sets), otherwise
    the benchmark's own `valid.jsonl`.
    """

    def __init__(self, args, tokenizer, data_dir):
        path = os.path.join(data_dir, f"valid_{args.model_type}.jsonl")
        if not os.path.exists(path):
            path = os.path.join(data_dir, "valid.jsonl")
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        # chat-template prompts already contain their special tokens
        add_special_tokens = args.model_type != "qwen3"
        self.prompts = [
            tokenizer.encode(r["prompt"].replace("<n>", "\n"), add_special_tokens=add_special_tokens)[:args.max_prompt_length]
            for r in rows
        ]
        self.answers = [r["output"] if isinstance(r["output"], list) else [r["output"]] for r in rows]
        self.pad_id = tokenizer.eos_token_id
        self.max_prompt_length = args.max_prompt_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return self.prompts[index]

    def collate(self, samples):
        bs = len(samples)
        batch = {
            "input_ids": torch.full((bs, self.max_prompt_length), self.pad_id, dtype=torch.long),
            "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
        }
        for i, prompt in enumerate(samples):  # left padding
            batch["input_ids"][i, -len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            batch["attention_mask"][i, -len(prompt):] = 1
        return batch
