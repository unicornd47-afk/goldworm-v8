"""Seed lists, seed guard, initial-state rule (PLAN §5 fix 3), episode ids and RNG derivation.

RNG derivation (IMPL_NOTES D1): the episode draw for plan seed s uses np.random.default_rng(s)
(SeedSequence(entropy=s)); training-data excitation uses default_rng([s, 1]). All episodes from
one seed share the draw (weather, schedule, noise) and the initial state; interventions differ only."""
import numpy as np

from . import spec as S

# slot numbering for episode ids (candidate-library child key, Option1 child(10, episode_id, k))
SLOT = {"N0": 0}
SLOT.update({iid: 1 + i for i, iid in enumerate(S.TRAIN_IDS)})
SLOT.update({iid: 7 + i for i, iid in enumerate(S.HELDOUT_IDS)})
N_SLOTS = 16


class SeedGuardError(RuntimeError):
    pass


class SeedGuard:
    """mode 'production': test seeds need sealed selection; adaptation seeds need sealed primary.
    mode 'preflight'/'smoke': only seeds in PREFLIGHT_SEEDS may be drawn."""

    def __init__(self, mode):
        if mode not in ("production", "preflight", "smoke"):
            raise SeedGuardError(mode)
        self.mode = mode
        self.selection_sealed = False
        self.primary_sealed = False
        self.drawn = []          # (seed, purpose)

    def check(self, seed, purpose):
        seed = int(seed)
        # ordering rules apply in every mode (purpose-based)
        if purpose.startswith("test") and not self.selection_sealed:
            raise SeedGuardError("test-purpose episode before COMPARATOR_SELECTION sealed")
        if purpose in ("adaptation_data", "adapt_eval") and not self.primary_sealed:
            raise SeedGuardError("adaptation before primary sealed")
        if self.mode == "preflight" and (purpose.startswith("test") or purpose in ("adaptation_data", "adapt_eval")):
            raise SeedGuardError("preflight may not create test/adaptation episodes")
        if self.mode in ("preflight", "smoke"):
            if seed not in S.PREFLIGHT_SEEDS:
                raise SeedGuardError("seed %d not allowed in %s mode" % (seed, self.mode))
        else:
            if seed in S.BURNED_SEEDS or seed in S.PREFLIGHT_SEEDS:
                raise SeedGuardError("burned/preflight seed %d in production" % seed)
            if seed in S.TEST_SEEDS and not self.selection_sealed:
                raise SeedGuardError("test seed %d before COMPARATOR_SELECTION sealed" % seed)
            if seed in S.TEST_SEEDS and not (purpose.startswith("test") or purpose == "adapt_eval"):
                raise SeedGuardError("test seed %d used for %s" % (seed, purpose))
            if purpose == "adapt_eval" and seed not in S.TEST_SEEDS[:S.ADAPT_EVAL_SEEDS_N]:
                raise SeedGuardError("adapt_eval seed %d outside 9100-9109" % seed)
            allowed = {"train": sum((list(v) for v in S.TRAIN_SEEDS.values()), []),
                       "val": sum((list(v) for v in S.VAL_SEEDS.values()), []),
                       "tuning": S.TUNING_SEEDS, "test_heldout": S.TEST_SEEDS, "test_id": S.TEST_SEEDS,
                       "adapt_eval": S.TEST_SEEDS[:S.ADAPT_EVAL_SEEDS_N], "adaptation_data": S.ADAPT_SEEDS}
            if purpose not in allowed or seed not in allowed[purpose]:
                raise SeedGuardError("seed %d not in the %s list" % (seed, purpose))
            if purpose == "adaptation_data" and not self.primary_sealed:
                raise SeedGuardError("adaptation data before primary sealed")
            if purpose == "adaptation_data" and seed not in S.ADAPT_SEEDS:
                raise SeedGuardError("adaptation data from non-adaptation seed %d" % seed)
        self.drawn.append((seed, purpose))


def initial_state_index(position):
    """Seed at 0-based position i in its list -> S[i mod 4]."""
    return int(position) % 4


def initial_state(position):
    return S.INITIAL_STATES[initial_state_index(position)]


def episode_id(seed, iv_id):
    return int(seed) * N_SLOTS + SLOT[iv_id]


def draw_rng(seed):
    return np.random.default_rng(int(seed))


def excitation_rng(seed):
    return np.random.default_rng([int(seed), 1])


def tuning_entry(position):
    """Tuning seed at position i: TRAIN entry index i mod 6."""
    return S.TRAIN_IDS[int(position) % 6]


def train_composition(n_ep=None):
    """Training episode e (0-based position in the replicate's seed list) -> (iv_id, onset).
    8 nominal first, then 4 per TRAIN entry in file order; onsets cycle 72,120,168,216 (IMPL_NOTES D3)."""
    n_ep = S.N_TRAIN_EP if n_ep is None else n_ep
    comp = [("N0", None)] * 8
    for iid in S.TRAIN_IDS:
        comp += [(iid, S.TRAIN_ONSETS[j]) for j in range(4)]
    assert len(comp) == 32
    return comp[:n_ep]


def val_composition(n_ep=None):
    """Validation: 2 nominal + 1 per TRAIN entry; onset cycles over entries (IMPL_NOTES D3)."""
    n_ep = S.N_VAL_EP if n_ep is None else n_ep
    comp = [("N0", None)] * 2 + [(iid, S.TRAIN_ONSETS[j % 4]) for j, iid in enumerate(S.TRAIN_IDS)]
    assert len(comp) == 8
    return comp[:n_ep]


def adaptation_composition(htype, K):
    """K adaptation episodes for HELDOUT type htype ('H1'..'H4'): seeds 8000.. in order,
    magnitudes alternating a, b; onsets cycle 72,120,168,216 like training data (IMPL_NOTES D5)."""
    out = []
    for j in range(K):
        iid = htype + ("a" if j % 2 == 0 else "b")
        out.append((S.ADAPT_SEEDS[j], j, iid, S.TRAIN_ONSETS[j % 4]))
    return out
