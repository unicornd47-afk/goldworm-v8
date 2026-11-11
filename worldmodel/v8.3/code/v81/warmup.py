"""ERRATUM_v8.3_E1.1/E2.1 warm-up: a separately logged throwaway fit of WARMUP_UPDATES updates before a
measured segment. Stricter reading (IMPL_NOTES): the throwaway model has the SAME architecture (arm, structure) as
the fit that follows. Train windows only (the caller passes the replicate's train ctx/fut/tgt). No RNG coupling:
batch rows from numpy default_rng([WARMUP_SEED, 7]); the torch global RNG is forked and restored around the
build (MT.build seeds the global torch RNG); weights are discarded. Fit results with and without warm-up are
bit-identical (tested)."""
import time

import numpy as np
import torch

from . import models_torch as MT
from . import spec as S


def warmup(arm, structure, norm, ctx, fut, tgt, n_updates=S.WARMUP_UPDATES, seed=S.WARMUP_SEED,
           clock=time.process_time, check=None, check_every=S.FIT_CHECK_EVERY):
    """Returns dict(arm, structure, updates, cpu_s, seed). `check` is called every `check_every` updates (caps)."""
    t0 = clock()
    if arm == "LIN" or n_updates <= 0:
        return dict(arm=arm, structure=structure, updates=0, cpu_s=clock() - t0, seed=seed, block_updates=S.PROBE_BLOCK_UPDATES,
                    block_per_update_cpu_s=[])
    with torch.random.fork_rng(devices=[]):
        mdl = MT.build(arm, structure, norm, seed)
        bidx = np.random.default_rng([int(seed), 7]).integers(0, len(ctx), size=(n_updates, S.BATCH))
        lr = lambda t: MT.lr_cosine(t, n_updates, S.LR0, S.LR_MIN)
        blocks, last = [], [clock()]

        def cb(n):                                    # ERRATUM E2.2: CPU-s per update in blocks of 50 (warm-up too)
            if n % S.PROBE_BLOCK_UPDATES == 0:
                now = clock()
                blocks.append((now - last[0]) / S.PROBE_BLOCK_UPDATES)
                last[0] = now
        mdl, info = MT.train(mdl, ctx, fut, tgt, bidx, n_updates, lr, budget_check=check, check_every=check_every,
                             log_every=0, progress_cb=cb)
        done = info["updates_done"]
        del mdl
    return dict(arm=arm, structure=structure, updates=int(done), cpu_s=clock() - t0, seed=seed,
                block_updates=S.PROBE_BLOCK_UPDATES, block_per_update_cpu_s=blocks)
