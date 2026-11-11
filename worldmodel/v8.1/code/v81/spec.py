"""Binding plan files (HASHES_r3) and frozen numbers. CONFIG governs numbers, PLAN governs logic (PLAN Status)."""
import hashlib
import json
import os

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN_DIR = os.path.dirname(CODE_DIR)                     # .../v8/worldmodel/v8.1
HASHES_FILE = "HASHES_r3.sha256"
HASHES_FILE_SHA256 = "b2fac5bf5e57e9e7013445aba7b57b7e192aa10e4c72e49247d00d0fc6d70ed7"
BOUND_FILES = {
    "PLAN_v8.1_r3.md": "dae8731c5cc0e7c41d2bea830234be34f6f27d68279179dbcb086e30b34c7e7f",
    "CONFIG_v8.1_r3.json": "580bba9255265a76cbc92b23645817e1e6fb622dfbf23a1f208b25eab575ee21",
    "FEASIBILITY_v8.1_r3.md": "f7c772ec707a98219b517057ec55594c5d4e3cbae1ecd3229cdedff2637f3464",
    "INTERVENTIONS_v8.1.json": "948b5e3d5c53c9d6c4862fba1c6a4203c4f0c665d63a97be5f44dcf0bef61d9c",
}
PLAN_PASS_FILE = "/workspace/audit_dahill_v8.1/PLAN_PASS_r3.json"
PLAN_PASS_SHA256 = "2cda3f4809b119339ec21b1b7fd2a28501d4a35cc803d71c45b506a41d198727"
OPTION1_DIR = "/workspace/nested_swarms/v7/worldmodel/option1_thermal_v5"   # read-only reference


class SpecError(RuntimeError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_plan_hashes(plan_dir=PLAN_DIR):
    """Returns dict of verified hashes or raises SpecError (STOP condition)."""
    hf = os.path.join(plan_dir, HASHES_FILE)
    got = sha256_file(hf)
    if got != HASHES_FILE_SHA256:
        raise SpecError("HASHES_r3.sha256 mismatch: %s" % got)
    listed = {}
    with open(hf) as f:
        for line in f:
            if line.strip():
                h, name = line.split()
                listed[name] = h
    if listed != BOUND_FILES:
        raise SpecError("HASHES_r3 content differs from bound list")
    for name, h in BOUND_FILES.items():
        g = sha256_file(os.path.join(plan_dir, name))
        if g != h:
            raise SpecError("%s mismatch: %s" % (name, g))
    return dict(BOUND_FILES, **{HASHES_FILE: HASHES_FILE_SHA256})


def load_config(plan_dir=PLAN_DIR):
    with open(os.path.join(plan_dir, "CONFIG_v8.1_r3.json")) as f:
        return json.load(f)


def load_interventions(plan_dir=PLAN_DIR):
    with open(os.path.join(plan_dir, "INTERVENTIONS_v8.1.json")) as f:
        return json.load(f)


CFG = load_config()
IV = load_interventions()

# ---- numbers from CONFIG (governing) ----
DT = CFG["time"]["DT_h"]
EP_TRANSITIONS_SCEN = CFG["time"]["episode_transitions"]          # 192
EP_TRANSITIONS_DATA = CFG["training"]["episode_transitions"]      # 288
WARMUP = 96
CONTROL_FIRST, CONTROL_LAST = 96, 191
N_CONTROL = 96
WIN_PRIMARY = (96, 191)       # inclusive decision indices k
WIN_SECONDARY = (120, 191)
CONTEXT = CFG["context_steps"]   # 48
H = CFG["horizon_steps"]         # 12
PI_CFG = (float(CFG["PI"]["Kp"]), float(CFG["PI"]["Ti_h"]), int(CFG["PI"]["ell"]))
N_CAND = CFG["planner"]["N_CAND"]
LAMBDAS = tuple(CFG["planner"]["lambda_grid"])
GRU_HIDDEN = tuple(CFG["arms"]["GRU"]["structure_grid_hidden"])
GRU_PARAMS = tuple(CFG["arms"]["GRU"]["params"])
WM_HIDDEN = tuple(CFG["arms"]["WM"]["structure_grid_hidden"])
WM_PARAMS = tuple(CFG["arms"]["WM"]["params"])
LIN_ORDERS = tuple(CFG["arms"]["LIN"]["structure_grid_order"])
RC_STRUCTS = (1, 2, 3)        # CONFIG structure_grid: 1-state, 2-state, 3-state
RC_PARAMS = tuple(CFG["arms"]["RC_only"]["params"])
N_UPDATES = CFG["training"]["updates"]
BATCH = CFG["training"]["batch"]
LR0, LR_MIN = CFG["training"]["lr"]
GRAD_CLIP = 1.0
REPLICATES = CFG["training"]["replicates"]
N_TRAIN_EP = CFG["training"]["episodes_per_replicate"]["train"]
N_VAL_EP = CFG["training"]["episodes_per_replicate"]["val"]
TRAIN_ONSETS = (72, 120, 168, 216)
IV_ONSET_TEST = 120
IV_DURATION = 48
N_BOOT = CFG["endpoints"]["n_boot"]
BOOT_LO_IDX, BOOT_HI_IDX = 249, 9750       # 0-based (PLAN §9 fix 4)
P1_MIN_SAVING = 0.05
P3_MARGIN = 0.25
FALLBACK_MAX_FRAC = 0.10
ADAPT_K = tuple(CFG["adaptation"]["K"])
ADAPT_UPDATES = 100
ADAPT_LR = 1e-4
ADAPT_EVAL_SEEDS_N = 10
CAPS = dict(CFG["caps_cpu_s"]["per_phase"])
GLOBAL_CAP = float(CFG["caps_cpu_s"]["global_cap"])
PREFLIGHT_CAP = float(CFG["preflight"]["cap_cpu_s"])
PREFLIGHT_SEEDS = tuple(CFG["preflight"]["seeds"])
GATE_FRAC = 0.9
TOTAL_GATE = 3240.0
LADDER_PHASES = ("adaptation", "in_distribution", "tuning")

SEEDS = CFG["seeds"]
TUNING_SEEDS = tuple(SEEDS["tuning"])
TEST_SEEDS = tuple(SEEDS["test"])
ADAPT_SEEDS = tuple(SEEDS["adaptation"])


def seed_range(pair):
    return tuple(range(pair[0], pair[1] + 1))


TRAIN_SEEDS = {r: seed_range(SEEDS["train_data"]["r%d" % (r + 1)]) for r in range(REPLICATES)}
VAL_SEEDS = {r: seed_range(SEEDS["val_data"]["r%d" % (r + 1)]) for r in range(REPLICATES)}
INIT_SEEDS = {r: SEEDS["model_init"]["r%d" % (r + 1)] for r in range(REPLICATES)}
BOOT_SEEDS = dict(SEEDS["bootstrap"])
BURNED_SEEDS = tuple(range(5000, 5040))
INITIAL_STATES = tuple((s["T_r0"], s["T_w0"], s["q_h0"]) for s in CFG["initial_states"])

TRAIN_IVS = tuple(IV["TRAIN_INTERVENTIONS"])
HELDOUT_IVS = tuple(IV["HELDOUT_INTERVENTIONS"])
TRAIN_IDS = tuple(e["id"] for e in TRAIN_IVS)
HELDOUT_IDS = tuple(e["id"] for e in HELDOUT_IVS)
IV_BY_ID = {e["id"]: e for e in TRAIN_IVS + HELDOUT_IVS}
HELDOUT_TYPES = ("H1", "H2", "H3", "H4")

ARMS = ("WM", "GRU", "LIN", "RC_only")
COMPARATOR_ARMS_TIEBREAK = ("RC_only", "LIN", "GRU")


def structures(arm):
    return {"WM": WM_HIDDEN, "GRU": GRU_HIDDEN, "LIN": LIN_ORDERS, "RC_only": RC_STRUCTS}[arm]


def configs(arm):
    return [(s, lam) for s in structures(arm) for lam in LAMBDAS]


assert len(TUNING_SEEDS) == 20 and len(TEST_SEEDS) == 40 and len(ADAPT_SEEDS) == 10
assert all(len(TRAIN_SEEDS[r]) == N_TRAIN_EP and len(VAL_SEEDS[r]) == N_VAL_EP for r in range(REPLICATES))
assert len(TRAIN_IDS) == 6 and len(HELDOUT_IDS) == 8
assert abs(sum(CAPS.values()) - GLOBAL_CAP) < 1e-9
