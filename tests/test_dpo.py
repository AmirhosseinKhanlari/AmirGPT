"""Tests for the DPO data preparation and trainer."""
import json
import math
import os

import numpy as np
import torch

from sft.chat_format import ASSISTANT, END, enc, encode_conversation


def pair(prompt, chosen, rejected, sc=5.0, sr=3.0):
    u = {"role": "user", "content": prompt}
    return {"prompt_id": prompt, "chosen": [u, {"role": "assistant", "content": chosen}],
            "rejected": [u, {"role": "assistant", "content": rejected}], "score_chosen": sc, "score_rejected": sr}


def test_dpo_loss_hand_computed():
    from sft.train_dpo import dpo_loss
    # policy prefers chosen by 1 nat more than the reference, rejected by 1 nat less:
    # rewards = 0.1 * (+1) = 0.1 and 0.1 * (-1) = -0.1, margin 0.2, loss = log(1 + e^-0.2)
    loss, cr, rr = dpo_loss(torch.tensor([-10.0]), torch.tensor([-12.0]), torch.tensor([-11.0]), torch.tensor([-11.0]), 0.1)
    assert math.isclose(cr.item(), 0.1, rel_tol=1e-6) and math.isclose(rr.item(), -0.1, rel_tol=1e-6)
    assert math.isclose(loss.item(), math.log(1 + math.exp(-0.2)), rel_tol=1e-6)  # 0.598139
    # policy == reference -> loss log 2, zero rewards
    loss, cr, rr = dpo_loss(torch.tensor([-3.0]), torch.tensor([-4.0]), torch.tensor([-3.0]), torch.tensor([-4.0]), 0.5)
    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6) and cr.item() == 0 and rr.item() == 0


def test_sequence_logps_sums_only_masked_positions():
    from sft.train_dpo import sequence_logps
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 7)
    y = torch.tensor([[1, 2, 3, 4], [0, 5, 6, 0]])
    m = torch.tensor([[0, 1, 1, 0], [0, 0, 1, 1]])
    lp = torch.log_softmax(logits, -1)
    want = torch.stack([lp[0, 1, 2] + lp[0, 2, 3], lp[1, 2, 6] + lp[1, 3, 0]])
    assert torch.allclose(sequence_logps(logits, y, m), want, atol=1e-6)


def test_build_split_masks_only_final_reply_and_filters():
    from sft.prepare_dpo import build_split, identity_hit
    rows = [pair("What is 2+2?", "It is 4.", "It is 5."),
            pair("Same?", "identical", "identical"),  # dropped: identical
            pair("Tie?", "a", "b", 4.0, 4.0),  # dropped: tie
            pair("Who?", "As an AI developed by OpenAI, I can't.", "Hello"),  # dropped: other identity
            pair("Who2?", "Sure", "I am ChatGPT"),  # dropped: other identity (rejected side)
            pair("word " * 1000, "x", "y"),  # dropped: prompt leaves < 64 tokens
            pair("Long?", "yes " * 2000, "no")]  # kept, chosen truncated
    arrays, st = build_split(rows, "t")
    assert st["kept"] == 2 and st["dropped_identical"] == 1 and st["dropped_tie"] == 1
    assert st["dropped_other_identity"] == 2 and st["dropped_prompt_too_long"] == 1
    assert st["truncated_pairs"] == 1 and st["truncated_chosen"] == 1 and st["truncated_rejected"] == 0
    toks, mask, offs = arrays["chosen"]
    t0, k0 = toks[offs[0]:offs[1]].tolist(), mask[offs[0]:offs[1]].tolist()
    assert (t0, k0) == tuple(encode_conversation(rows[0]["chosen"]))  # one turn: same as the SFT encoding
    learned = [t for t, k in zip(t0, k0) if k]
    assert learned == enc.encode_ordinary("It is 4.") + [END]
    assert offs[2] - offs[1] == 1024 and mask[offs[2] - 1] == 1  # truncated reply, still masked
    assert arrays["scores"].tolist() == [[5.0, 3.0], [5.0, 3.0]]
    assert identity_hit("A llama is an animal.") is None and identity_hit("Meta's LLaMA model") == "llama"
    # ties are kept on request
    assert build_split(rows[2:3], "t", keep_ties=True)[1]["kept"] == 1


def test_multi_turn_prompt_masks_only_last_assistant():
    from sft.prepare_dpo import encode_pair_side
    prompt = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"},
              {"role": "user", "content": "Bye?"}]
    toks, mask, truncated = encode_pair_side(prompt, "Goodbye.")
    assert not truncated and toks.count(ASSISTANT) == 2
    assert [t for t, k in zip(toks, mask) if k] == enc.encode_ordinary("Goodbye.") + [END]


def test_repeated_ngrams():
    from sft.train_dpo import repeated_ngrams, quality_summary
    assert repeated_ngrams("a b c d e") == (0, 2)
    # "visit the Notre-Dame Cathedral" twice: words visit the notre dame cathedral (5) x 2
    rep, tot = repeated_ngrams("Visit the Notre-Dame Cathedral. Visit the Notre-Dame Cathedral.")
    assert (rep, tot) == (2, 7)  # 10 words -> 7 4-grams; the last 2 repeat the first 2
    q = quality_summary([{"reply": "a b c a b c a", "n_tokens": 7, "stopped": True},
                         {"reply": "hi", "n_tokens": 1, "stopped": False}])
    assert math.isclose(q["rep4"], 1 / 4) and q["no_end_frac"] == 0.5 and q["mean_tokens"] == 4.0


def _write(tmp, name, arrays):
    for side in ("chosen", "rejected"):
        for k, a in zip(("tokens", "mask", "offsets"), arrays[side]):
            np.save(tmp / f"{name}_{side}_{k}.npy", a)


def test_dpo_smoke_run(tmp_path):
    """Tiny random model: reward accuracy/margin on the (here identical) test pairs must rise."""
    from gpt2.model import GPT, GPTConfig, config_to_dict, load_checkpoint
    from sft.prepare_dpo import build_split
    from sft.prepare_sft import build_split as build_sft, identity_conversations
    from sft.train_dpo import PreferenceData, main
    rows = [pair(f"Question number {i}: what is the colour of the sky?", f"The sky is blue, answer {i}.",
                 f"Bananas bananas bananas {i} bananas.") for i in range(32)]
    arrays, _ = build_split(rows, "t")
    _write(tmp_path, "train", arrays)
    _write(tmp_path, "test", arrays)
    d = PreferenceData(str(tmp_path), "test")
    x, y, m = d.batch(np.array([0, 1]))
    assert x.shape[0] == 4 and m[:2].sum() > 0 and m[2:].sum() > 0  # chosen rows then rejected rows
    sft = tmp_path / "sft"
    sft.mkdir()
    toks, mask, offs, _ = build_sft(identity_conversations(20, 0), "v")
    for k, a in (("tokens", toks), ("mask", mask), ("offsets", offs)):
        np.save(sft / f"val_{k}.npy", a)
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=50304, n_layer=2, n_head=2, n_embd=64))
    torch.save({"model": model.state_dict(), "config": config_to_dict(model.config), "step": 0}, tmp_path / "init.pt")
    out = tmp_path / "out"
    main(["--init", str(tmp_path / "init.pt"), "--data", str(tmp_path), "--sft_data", str(sft), "--out", str(out),
          "--max_steps", "20", "--micro_batch", "8", "--max_lr", "3e-3", "--eval_interval", "10",
          "--sft_eval_batches", "2", "--sft_eval_micro", "4", "--quality_prompts", "3", "--quality_max_tokens", "12",
          "--device", "cpu", "--dtype", "fp32"])
    lines = [json.loads(l) for l in open(out / "metrics.jsonl")]
    vals = [l for l in lines if l["kind"] == "val"]
    assert math.isclose(vals[0]["value"], math.log(2), rel_tol=1e-5) and vals[0]["reward_acc"] == 0  # pi == ref
    assert vals[-1]["reward_acc"] > 0.9 and vals[-1]["reward_margin"] > 0 and vals[-1]["value"] < math.log(2)
    assert {"reward_acc", "reward_margin", "chosen_reward", "rejected_reward"} <= set(
        next(l for l in lines if l["kind"] == "train"))
    assert len([l for l in lines if l["kind"] == "quality"]) == 2 and len([l for l in lines if l["kind"] == "sft_val"]) == 2
    summary = json.load(open(out / "summary.json"))
    assert summary["quality_start"]["n"] == 3 and "sft_val_end" in summary
    for name in ("amirgpt_dpo_best.pt", "amirgpt_dpo_last.pt"):
        m2, ck = load_checkpoint(str(out / name))
        assert ck["chat_format"] == "amirgpt-v1" and ck["config"]["vocab_size"] == 50304
    # training changed the weights
    m0, _ = load_checkpoint(str(tmp_path / "init.pt"))
    assert not torch.equal(m0.lm_head.weight, m2.lm_head.weight)


def test_preference_data_max_len_and_max_pairs(tmp_path):
    from sft.prepare_dpo import build_split
    from sft.train_dpo import PreferenceData
    rows = [pair(f"q{i}", "short " * (5 if i % 2 else 200), "no") for i in range(10)]
    arrays, _ = build_split(rows, "t")
    _write(tmp_path, "train", arrays)
    full = PreferenceData(str(tmp_path), "train")
    short = PreferenceData(str(tmp_path), "train", max_len=50)
    assert len(full) == 10 and len(short) == 5 and short.idx.tolist() == [1, 3, 5, 7, 9]
    assert short.lengths.max() <= 50
    two = PreferenceData(str(tmp_path), "train", max_pairs=2, max_len=50)
    x, y, m = two.batch([1])
    xf, yf, mf = full.batch([3])  # local pair 1 of the filtered view is stored pair 3
    assert torch.equal(x, xf) and torch.equal(m, mf)
