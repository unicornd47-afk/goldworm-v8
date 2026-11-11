"""Frozen experiment constants for exp001. Do not change after code freeze without an erratum."""
from pathlib import Path

ROOT = Path("/workspace/growth")
CODE = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "upstream_v8_c0dc0f13" / "model.safetensors"

# Model
VOCAB = 256
D_MODEL = 768
N_HEAD = 12
FFN_MULT = 4
CTX = 2048
L_START = 12
L_RATIO_A = 24
L_RATIO_B = 36
N_PARAMS = {12: 85_169_664, 24: 170_141_184, 36: 255_112_704}

# Training
TRAIN_T = 256          # predicted positions; input length is TRAIN_T + 1
BATCH = 1
THREAD_COUNT = 4       # torch + OMP, frozen for CPU preflight and all exp001 arms
EXP001_DEVICE = "cpu"  # all arms including proxy; GPU is out of scope for exp001
EXP001_DTYPE = "fp32"  # bf16 is out of scope for exp001
SEEDS = (11, 22, 33)
BOOTSTRAP_SEED = 77701
BOOTSTRAP_B = 10_000
MARGIN_BPB = 0.01
KNOCKOUT_MIN_BPB = 1e-3
N_DIV_MAX = 200
N_DIV_MIN = 50
N_DIV_FRAC = 0.2
NOISE_SIGMA = 1e-3
# RMS test: rms(r_i(N_div) - r_i(0)) >= max(RMS_RATIO * rms(r_i(0)), RMS_FLOOR)
RMS_RATIO = 2.0
RMS_FLOOR = 1e-4
COSINE_TO_SELF0_MAX = 0.70  # optional descriptive; not required for productive
PRODUCTIVE_FRACTION = 0.50
LR_OLD = 1e-5          # C's lr; also old blocks of G, and embeddings / ln_final
LR_NEW_PEAK = 1.5e-4   # S / new-block pretrain schedule
LR_NEW_MIN_FRAC = 0.1
WARMUP_STEPS = 200
ADAMW_BETAS = (0.9, 0.95)
WD_OLD = 0.01
WD_NEW = 0.1

# Function preservation. exp001 is locked to box CPU / fp32; GPU and bf16 are out of scope.
FP32_TOL_MAX_ABS = 1e-5
FP_INPUT_CONCAT_SHA256 = "898d3770faeef1901f9d614a79c8491d3de9f644415d5d1e761ea39d48e833b7"

# Locked artifacts
HELDOUT_SHA256 = "fdaa73481604e76f71b7fca70c91d7369a90e4d396e8ebe80be882c8874dffb6"
HELDOUT_V1_SHA256 = "0c81811b2ef9e1061627180e6d24d49d50cbeed81bfa0b734c55a901702e9df4"
TRAIN_NPY_SHA256 = "3ba0ac5bbb8f5056e1110f22eeb0976865db982aad79dd0b2379739759e7d507"
DEV_NPY_SHA256 = "b995fb77c8bd2f29e9093d65cf1b57f8b31945da0d5e05d8e15b4011d18f9f51"
WEIGHTS_SHA256 = "8a7b5716ab340a672bc307772fd3e8387beee4793c62e1f256c31b8c70aea0a7"

# Knockout / divergence eval subset of DEV: first K windows, first 256 bytes
KNOCKOUT_DEV_WINDOWS = 16
KNOCKOUT_BYTES = 256

# Proxy (secondary, total-FLOP). Pretrain bytes fixed; S_proxy scaled.
PROXY_D = 64
PROXY_HEADS = 4
PROXY_L0 = 2
PROXY_L1 = 4
PROXY_PRETRAIN_BYTES = 2_000_000
PROXY_B1_BYTES = 2_000_000          # post-grow default; PLAN_PASS may change only this
# unique params: 256*d + L*(12*d*d + 4*d) + 2*d
PROXY_N = {
    2: 256 * 64 + 2 * (12 * 64 * 64 + 4 * 64) + 2 * 64,   # 115328
    4: 256 * 64 + 4 * (12 * 64 * 64 + 4 * 64) + 2 * 64,   # 214144
}
# S_proxy: largest multiple of TRAIN_T that does not exceed
# (PROXY_N[2]/PROXY_N[4])*PROXY_PRETRAIN_BYTES + PROXY_B1_BYTES
PROXY_S_BYTES = 3_076_864           # 12,019 steps; residual FLOP fraction < 2e-5

# Wall caps from measured CPU preflight (results/preflight_torch_cpu.json)
DEFAULT_PILOT_CAP_S = 3600
HARD_CAP_S = 21600
PREFLIGHT_ABORT_FRAC = 0.9
GROWTH_FP_RESERVE_S = 300           # growth + FP + noise, growth arms only
EVAL_RESERVE_S = 600                # eval + knockout + q3 hook, every arm
# P1: train + applicable reserves <= 0.9 * 3600 = 3240 s for every arm.
# C/S reserves = 600 s ⇒ C training abort line = 2640 s.
# Growth arms reserves = 900 s ⇒ G/S training abort line = 2340 s (S uses 600 s
# but is sized with G and finishes sooner than C). Planning uses 2520 s for C
# (~120 s slack under 2640 so a per-arm re-measure cannot trip K5).
C_TRAIN_PLAN_S = 2520
C_TRAIN_ABORT_S = 2640              # 0.9*3600 - EVAL_RESERVE_S

# Registered CPU-box pilot presented-byte budgets (box CPU, 4 threads, fp32).
# G and S bytes are multiples of TRAIN_T. C figures are the FLOP-match planning
# totals. C/S stop at G's logged FLOPs (bytes only an upper bound; never more).
# GPU / bf16 are out of scope for exp001 (see EXP001_DEVICE / EXP001_DTYPE).
CPU_BOX_BUDGETS = {
    "A": {"G": 463_616, "C": 926_153, "S": 463_616},   # L=24; G 1,811 steps; C wall ~2518.8 s
    "B": {"G": 309_248, "C": 926_305, "S": 309_248},   # L=36; G 1,208 steps; C wall ~2519.2 s
}

# Convert hook (existing exporter, not vendored)
GW_CONVERT = Path("/workspace/gwpub/code/convert.py")
GW_Q3_KIND = "q3g64mse"
