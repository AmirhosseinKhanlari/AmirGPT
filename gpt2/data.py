"""
Data loading for pretraining (video [1:02:01] "DataLoaderLite", [3:16:52] FineWeb shards).

Same contract as Karpathy's DataLoaderLite: each call to next_batch() returns
(x, y) of shape (B, T) where y is x shifted by one token, and in DDP every rank
reads a disjoint, strided slice of the same token stream.

Improvements -- these are the extensions Karpathy himself asks for in the video
([3:46:58] the loss has "weird periodicity" because "we are ... iterating all
them from scratch without doing any permutation"; [3:50:05] "you would want to
look into extensions where you actually permute the data randomly, you permute
the documents around in every single shard on every single new epoch and
possibly even permute the shards"). Enabled with shuffle=True; shuffle=False
reproduces the video's loader exactly:
  * shard order is re-shuffled every epoch (train split only);
  * inside each train shard, the DOCUMENTS (delimited by <|endoftext|>) are
    shuffled, so batches mix topics/sources instead of reading the crawl in order;
  * all permutations are derived from (seed, epoch, shard) so every DDP rank sees
    the identical stream and runs are reproducible;
  * state_dict()/load_state_dict() allow exact resumption from a checkpoint.
The validation split is never shuffled, so val loss is comparable across runs.
"""

import os

import numpy as np
import torch

EOT = 50256


def load_tokens(filename):
    npt = np.load(filename)
    return npt.astype(np.int32)  # uint16 -> int32 first (README errata), torch.long later


def shuffle_documents(tokens, rng):
    """Permute the order of <|endoftext|>-delimited documents in a 1D token array."""
    starts = np.flatnonzero(tokens == EOT)
    if len(starts) < 2:
        return tokens
    # boundaries of segments; the shard may begin mid-document (segment before first EOT)
    bounds = np.concatenate(([0], starts[starts > 0], [len(tokens)]))
    lengths = np.diff(bounds)
    perm = rng.permutation(len(lengths))
    # vectorised gather: for each output segment, indices bounds[p] ... bounds[p]+len-1
    seg_lengths = lengths[perm]
    seg_starts = bounds[:-1][perm]
    offsets = np.repeat(seg_starts - np.concatenate(([0], np.cumsum(seg_lengths)[:-1])), seg_lengths)
    idx = np.arange(len(tokens)) + offsets
    return tokens[idx]


class ShardedDataLoader:

    def __init__(self, B, T, process_rank, num_processes, split, data_root="edu_fineweb10B",
                 shuffle=False, seed=1337, verbose=True):
        assert split in {"train", "val"}
        self.B, self.T = B, T
        self.process_rank, self.num_processes = process_rank, num_processes
        self.split = split
        self.shuffle = bool(shuffle) and split == "train"  # val is never shuffled
        self.seed = seed
        shards = sorted(s for s in os.listdir(data_root) if split in s and s.endswith(".npy") and ".tmp" not in s)
        self.shards = [os.path.join(data_root, s) for s in shards]
        assert len(self.shards) > 0, f"no shards found for split {split} in {data_root}"
        if verbose:
            print(f"found {len(self.shards)} shards for split {split} (shuffle={self.shuffle})")
        self.reset()

    # -- internal helpers ---------------------------------------------------------
    def _shard_order(self, epoch):
        if not self.shuffle:
            return np.arange(len(self.shards))
        return np.random.default_rng([self.seed, epoch]).permutation(len(self.shards))

    def _load(self):
        shard_id = int(self.order[self.shard_pos])
        tokens = load_tokens(self.shards[shard_id])
        if self.shuffle:
            tokens = shuffle_documents(tokens, np.random.default_rng([self.seed, self.epoch, shard_id]))
        self.tokens = torch.from_numpy(tokens).long()
        assert len(self.tokens) >= self.B * self.T * self.num_processes + 1, \
            f"shard {self.shards[shard_id]} too small for B*T*world"

    # -- public API -----------------------------------------------------------------
    def reset(self):
        self.epoch = 0
        self.order = self._shard_order(0)
        self.shard_pos = 0
        self._load()
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)  # targets
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to the next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.shard_pos += 1
            if self.shard_pos >= len(self.shards):
                self.epoch += 1
                self.order = self._shard_order(self.epoch)
                self.shard_pos = 0
            self._load()
            self.current_position = B * T * self.process_rank
        return x, y

    def state_dict(self):
        return {"epoch": self.epoch, "shard_pos": self.shard_pos, "current_position": self.current_position}

    def load_state_dict(self, state):
        self.epoch = state["epoch"]
        self.order = self._shard_order(self.epoch)
        self.shard_pos = state["shard_pos"]
        self._load()
        self.current_position = state["current_position"]


# Karpathy's name, kept for readers following the video
DataLoaderLite = ShardedDataLoader
