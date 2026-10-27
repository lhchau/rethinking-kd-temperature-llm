"""CPU checks of the losses, data pipeline and on-policy sampling: python -m pytest tests"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import LOSSES  # noqa: E402
from data_utils.lm_datasets import LMTrainDataset  # noqa: E402
from data_utils.mmap_dataset import MMapDatasetBuilder  # noqa: E402
from distillm import ReplayBuffer, SampleGenerator, distillation_loss  # noqa: E402

KD_LOSSES = [loss for loss in LOSSES if loss != "sft"]
ARGS = SimpleNamespace(kd_temperature=2.0, skew_alpha=0.1, drkl_alpha=1.0, drkl_beta=2.0)


def make_logits(seed=0, shape=(2, 6, 11)):
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, shape[-1], shape[:2], generator=g)
    labels[0, :2] = -100
    return torch.randn(shape, generator=g, dtype=torch.float64) * 2, labels


@pytest.mark.parametrize("loss", KD_LOSSES)
def test_gradient_vanishes_when_student_matches_teacher(loss):
    teacher, labels = make_logits()
    # a tiny offset: AKL's head/tail ratio is 0/0 at exact equality
    student = (teacher + 1e-6 * make_logits(1)[0]).requires_grad_()
    distillation_loss(SimpleNamespace(loss=loss, **vars(ARGS)), student, teacher, labels).backward()
    assert student.grad.abs().max() < 1e-5


@pytest.mark.parametrize("loss", KD_LOSSES)
def test_ignored_positions_do_not_matter(loss):
    student, labels = make_logits(0)
    teacher, _ = make_logits(1)
    args = SimpleNamespace(loss=loss, **vars(ARGS))
    perturbed = student.clone()
    perturbed[labels == -100] += 5.0
    assert torch.allclose(distillation_loss(args, student, teacher, labels),
                          distillation_loss(args, perturbed, teacher, labels))


def test_dataset_masks_prompt_and_left_pads_generation_prompt(tmp_path):
    examples = [([1, 2, 3], [4, 5]), ([6], [7, 8, 9, 10])]
    builder = MMapDatasetBuilder(str(tmp_path / "train_0"), np.uint16)
    for prompt, response in examples:
        builder.add_item(prompt + [-1] + response)
    builder.finalize()
    (tmp_path / "train.jsonl").write_text("".join(json.dumps({"output": "x"}) + "\n" for _ in examples))

    args = SimpleNamespace(data_dir=str(tmp_path), max_length=16, max_prompt_length=4)
    data = LMTrainDataset(args, SimpleNamespace(eos_token_id=0), "train")
    model_batch, labels, gen_batch = data.collate([data[0], data[1]])
    assert model_batch["input_ids"][0, :4].tolist() == [1, 2, 3, 4]
    assert labels[0, :4].tolist() == [-100, -100, 4, 5]
    assert labels[1, :4].tolist() == [7, 8, 9, 10]
    assert gen_batch["input_ids"].tolist() == [[0, 1, 2, 3], [0, 0, 0, 6]]
    assert gen_batch["attention_mask"].tolist() == [[0, 1, 1, 1], [0, 0, 0, 1]]


def test_student_samples_are_labelled_with_the_next_token():
    pad = 0

    class StubModel:
        def eval(self):
            pass

        def generate(self, input_ids, attention_mask, **kwargs):
            responses = torch.tensor([[11, 12, pad], [13, pad, pad]])
            return torch.cat([input_ids, responses], dim=1)

    args = SimpleNamespace(max_length=8, max_prompt_length=3, student_gen_max_new_tokens=None)
    generator = SampleGenerator(args, SimpleNamespace(eos_token_id=pad, pad_token_id=pad))
    gen_batch = {"input_ids": torch.tensor([[pad, 1, 2], [3, 4, 5]]), "attention_mask": torch.ones(2, 3)}
    model_batch, labels = generator.run_sample(StubModel(), gen_batch)
    assert model_batch["input_ids"].tolist() == [[1, 2, 11, 12], [3, 4, 5, 13]]
    assert labels.tolist() == [[-100, 11, 12, -100], [-100, -100, 13, -100]]

    buffer = ReplayBuffer(capacity=4, batch_size=2, pad_id=pad)
    buffer.add({"input_ids": model_batch["input_ids"][:1, :3], "attention_mask": model_batch["attention_mask"][:1, :3]},
               labels[:1, :3])
    buffer.add(model_batch, labels)
    replay_batch, replay_labels = buffer.sample("cpu")
    assert replay_batch["input_ids"].shape == replay_labels.shape == (2, 4)
