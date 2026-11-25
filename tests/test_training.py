"""Tests for the dataloader, LR schedule and an end-to-end mini training run."""
import json
import math
import os

import numpy as np
import pytest
import torch

from gpt2.data import EOT, ShardedDataLoader, shuffle_documents
from gpt2.train_gpt2 import get_lr


def make_shards(root, n_train=3, size=5000, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)
    for i in range(n_train + 1):
        toks = rng.integers(0, 50000, size=size).astype(np.uint16)
        toks[rng.choice(size, size // 100, replace=False)] = EOT  # ~100-token documents
        split = "val" if i == 0 else "train"
        np.save(os.path.join(root, f"edufineweb_{split}_{i:06d}.npy"), toks)


# ----------------------------------------------------------------------------- LR
def test_lr_schedule_matches_video():
    assert get_lr(0) == pytest.approx(6e-4 / 715)
    assert get_lr(714) == pytest.approx(6e-4)  # end of warmup = max_lr
    assert get_lr(715) == pytest.approx(6e-4)
    assert get_lr(19073) == pytest.approx(6e-5)  # end = min_lr = 0.1 * max_lr
    assert get_lr(10 ** 6) == pytest.approx(6e-5)
    mid = (715 + 19073) / 2
    assert get_lr(int(mid)) == pytest.approx(6e-5 + 0.5 * (6e-4 - 6e-5), rel=1e-3)
    lrs = [get_lr(i) for i in range(715, 19074, 97)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))  # monotone decay


# ----------------------------------------------------------------------------- data
def test_loader_targets_are_shifted_inputs(tmp_path):
    make_shards(tmp_path)
    dl = ShardedDataLoader(4, 32, 0, 1, "train", str(tmp_path), verbose=False)
    raw = np.load(os.path.join(tmp_path, "edufineweb_train_000001.npy")).astype(np.int64)
    x, y = dl.next_batch()
    assert x.shape == (4, 32) and y.shape == (4, 32)
    assert torch.equal(x.flatten(), torch.from_numpy(raw[:128]))  # no shuffle == video behaviour
    assert torch.equal(y.flatten(), torch.from_numpy(raw[1:129]))


def test_ddp_ranks_get_disjoint_strided_slices(tmp_path):
    make_shards(tmp_path)
    r0 = ShardedDataLoader(2, 16, 0, 2, "train", str(tmp_path), verbose=False)
    r1 = ShardedDataLoader(2, 16, 1, 2, "train", str(tmp_path), verbose=False)
    raw = np.load(os.path.join(tmp_path, "edufineweb_train_000001.npy")).astype(np.int64)
    x0, _ = r0.next_batch()
    x1, _ = r1.next_batch()
    assert torch.equal(x0.flatten(), torch.from_numpy(raw[0:32]))
    assert torch.equal(x1.flatten(), torch.from_numpy(raw[32:64]))
    x0b, _ = r0.next_batch()
    assert torch.equal(x0b.flatten(), torch.from_numpy(raw[64:96]))


def test_shuffle_documents_is_a_permutation_of_documents():
    rng = np.random.default_rng(1)
    toks = np.array([7, 8, EOT, 1, 2, EOT, 3, EOT, 4, 5, 6], dtype=np.int32)
    out = shuffle_documents(toks, rng)
    assert sorted(out.tolist()) == sorted(toks.tolist())
    docs = lambda a: sorted(tuple(d) for d in np.split(a, np.flatnonzero(a == EOT)) if len(d))
    assert docs(out) == docs(toks)  # same multiset of documents, each kept intact


def test_shuffle_identical_across_ranks_and_val_unshuffled(tmp_path):
    make_shards(tmp_path)
    a = ShardedDataLoader(2, 16, 0, 2, "train", str(tmp_path), shuffle=True, seed=5, verbose=False)
    b = ShardedDataLoader(2, 16, 0, 2, "train", str(tmp_path), shuffle=True, seed=5, verbose=False)
    assert torch.equal(a.tokens, b.tokens)
    v = ShardedDataLoader(2, 16, 0, 1, "val", str(tmp_path), shuffle=True, verbose=False)
    raw = np.load(os.path.join(tmp_path, "edufineweb_val_000000.npy")).astype(np.int64)
    assert torch.equal(v.tokens, torch.from_numpy(raw))


def test_loader_epoch_wrap_and_state_roundtrip(tmp_path):
    make_shards(tmp_path, n_train=2, size=1000)
    dl = ShardedDataLoader(2, 50, 0, 1, "train", str(tmp_path), shuffle=True, verbose=False)
    for _ in range(25):  # > 2 shards worth -> wraps into epoch 1
        dl.next_batch()
    assert dl.epoch >= 1
    state = dl.state_dict()
    expected = [dl.next_batch()[0] for _ in range(5)]
    dl2 = ShardedDataLoader(2, 50, 0, 1, "train", str(tmp_path), shuffle=True, verbose=False)
    dl2.load_state_dict(state)
    got = [dl2.next_batch()[0] for _ in range(5)]
    assert all(torch.equal(e, g) for e, g in zip(expected, got))


# ----------------------------------------------------------------------------- e2e
def test_mini_training_run_and_resume(tmp_path):
    from gpt2.train_gpt2 import main
    data = tmp_path / "data"
    make_shards(data, n_train=2, size=20000)
    common = ["--data_root", str(data), "--log_dir", str(tmp_path / "log"), "--total_batch_size", "512",
              "--micro_batch", "2", "--seq_len", "128", "--n_layer", "2", "--n_head", "2", "--n_embd", "64",
              "--warmup_steps", "2", "--eval_interval", "3", "--val_steps", "2", "--hellaswag", "0",
              "--sample_interval", "3", "--ckpt_interval", "3", "--latest_interval", "2",
              "--device", "cpu", "--dtype", "fp32", "--shuffle", "1"]
    main(common + ["--max_steps", "4"])
    lines = [json.loads(l) for l in open(tmp_path / "log" / "metrics.jsonl")]
    assert sum(1 for l in lines if l["kind"] == "train") == 4
    assert os.path.exists(tmp_path / "log" / "latest.pt")
    assert os.path.exists(tmp_path / "log" / "model_00003.pt")
    ck = torch.load(tmp_path / "log" / "latest.pt", weights_only=False)
    assert ck["step"] == 3 and "optimizer" in ck and "train_loader" in ck
    # resume and extend to 6 steps: should run steps 4 and 5 only
    main(common + ["--max_steps", "6", "--resume", "1"])
    lines = [json.loads(l) for l in open(tmp_path / "log" / "metrics.jsonl")]
    steps = [l["step"] for l in lines if l["kind"] == "train"]
    assert steps == [0, 1, 2, 3, 4, 5]


def test_overfit_single_batch():
    """Video [1:00:55]: a correct model+optimizer can drive loss on one batch to ~0."""
    from gpt2.model import GPT, GPTConfig
    torch.manual_seed(0)
    m = GPT(GPTConfig(block_size=32, vocab_size=100, n_layer=2, n_head=2, n_embd=64))
    x = torch.randint(0, 100, (4, 32))
    y = torch.randint(0, 100, (4, 32))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(300):
        opt.zero_grad()
        _, loss = m(x, y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
