"""Binding plan files (HASHES_v8.2) and frozen numbers. CONFIG governs numbers, PLAN governs logic (PLAN Status).

v8.2: ALL plan-bundle identity lives in the single BUNDLE dict below. Swapping the bundle (e.g. a v8.2 r2)
means editing BUNDLE only (hashes file + its sha, listed files, CONFIG name, plan-PASS file + sha)."""
import hashlib
import json
import os
import re

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN_DIR = os.path.dirname(CODE_DIR)                     # .../v8/worldmodel/v8.2

TODO_AFTER_PASS = "TODO_AFTER_PASS"                      # placeholder: launch is impossible while this is set
BUNDLE = dict(
    version="v8.2",
    hashes_file="HASHES_v8.2_E1.sha256",                 # v8.2 bundle + ERRATUM_v8.2_E1 (erratum wins on conflict)
    hashes_file_sha256="304bc28de338c92ee72817e7ed4ad498c367165e7f3ca47eed38e1732c636ff3",
    files={                                              # names exactly as listed in hashes_file (relative to PLAN_DIR)
        "PLAN_v8.2.md": "a1e5b1cd883ae1bff52d4eb86c18c9e3dda98681df7b0700c966249f9a7726c3",
        "CONFIG_v8.2.json": "7f31dc4e43e7899016e0c049c2e3ed251d278235f1fbc059f250ff5bc419ef63",
        "FEASIBILITY_v8.2.md": "6ce29d105e1f248a10c5a38e38f74db7173dc0e6eb721858870e6c4608ee3100",
        "../v8.1/INTERVENTIONS_v8.1.json": "948b5e3d5c53c9d6c4862fba1c6a4203c4f0c665d63a97be5f44dcf0bef61d9c",
        "HASHES_v8.2.sha256": "733cc25c235d6717f42223ce8868753bbccbfa80a234a4a88a8064deddb9b1d1",
        "ERRATUM_v8.2_E1.md": "67b4ecdb9bd5a31a91ef2e603fe15350b7048fb98a05f8e3168e99f152938c47",
    },
    # numbers the erratum changes vs CONFIG (CONFIG itself is unchanged); {} = CONFIG as is
    erratum_overrides={"lambda_grid": [0.5, 0.625, 0.75, 0.875, 1.0]},     # ERRATUM_v8.2_E1 §E1.1
    config_file="CONFIG_v8.2.json",
    interventions_file="../v8.1/INTERVENTIONS_v8.1.json",
    plan_pass_file="/workspace/audit_dahill_v8.2/PLAN_PASS_v8.2.json",   # dahill PLAN_PASS on bundle 304bc28d
    plan_pass_sha256="8ad13f896e2677498048f864ee3abed6e6534c4dcb75c889b59643de34bd49fe",
    plan_pass_hashes_key="hashes_v8.2_sha256",           # key in the PASS JSON that must equal hashes_file_sha256
    plan_pass_reviewer="dahill",                         # 'reviewer' in the PASS JSON
    release_file="RELEASE_v8.2.json",
    # implementation errata (outside code/, bound by sha) and the preserved failed first preflight
    impl_errata={"IMPL_ERRATUM_v8.2_W1.md": "08f62233dadb4d009dfe5d7ce3bd3529337886211c87b48c1db5743c1668bdf8"},
    preflight_attempt1=dict(file="PREFLIGHT_PROJECTION_attempt1_FAILED.json",
                            sha256="f2a0e6b2a608d71bd0f1d554a2aca481ac97621f2be08985a565bdf73c4b17f7"),
    code_manifest="CODE_HASHES_v8.2.sha256",
)
HASHES_FILE = BUNDLE["hashes_file"]
HASHES_FILE_SHA256 = BUNDLE["hashes_file_sha256"]
BOUND_FILES = dict(BUNDLE["files"])
PLAN_PASS_FILE = BUNDLE["plan_pass_file"]
PLAN_PASS_SHA256 = BUNDLE["plan_pass_sha256"]
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
        raise SpecError("%s mismatch: %s" % (HASHES_FILE, got))
    listed = {}
    with open(hf) as f:
        for line in f:
            if line.strip():
                h, name = line.split()
                listed[name] = h
    if listed != BOUND_FILES:
        raise SpecError("%s content differs from bound list" % HASHES_FILE)
    for name, h in BOUND_FILES.items():
        g = sha256_file(os.path.join(plan_dir, name))
        if g != h:
            raise SpecError("%s mismatch: %s" % (name, g))
    return dict(BOUND_FILES, **{HASHES_FILE: HASHES_FILE_SHA256})


def load_config(plan_dir=PLAN_DIR):
    with open(os.path.join(plan_dir, BUNDLE["config_file"])) as f:
        return json.load(f)


def load_interventions(plan_dir=PLAN_DIR):
    with open(os.path.join(plan_dir, BUNDLE["interventions_file"])) as f:
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
LAMBDAS = tuple(float(x) for x in BUNDLE["erratum_overrides"].get("lambda_grid", CFG["planner"]["lambda_grid"]))
# Machine-readable record of every number the erratum overrides over CONFIG (RELEASE_v8.2 must carry it verbatim).
GRID_OVERRIDE = dict(
    parameter="planner.lambda_grid",
    config_file=BUNDLE["config_file"], config_sha256=BUNDLE["files"][BUNDLE["config_file"]],
    config_value=[float(x) for x in CFG["planner"]["lambda_grid"]],
    effective_value=list(LAMBDAS),
    source="ERRATUM_v8.2_E1.md §E1.1", source_sha256=BUNDLE["files"]["ERRATUM_v8.2_E1.md"],
    applies_to=["WM", "GRU", "LIN", "RC_only"],
)
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
W1_ALLOWED_LADDER = ("a",)    # IMPL_ERRATUM_v8.2_W1 §W1.3(3): only step (a), only if adaptation fails

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


def parse_seed_ranges(entries):
    """CONFIG 'excluded_never_used_in_production' strings like '7000-7019' or '1-4 (preflight only)' -> set."""
    out = set()
    for e in entries:
        m = re.match(r"\s*(\d+)\s*-\s*(\d+)", str(e))
        if not m:
            raise SpecError("unparseable excluded seed range %r" % e)
        out.update(range(int(m.group(1)), int(m.group(2)) + 1))
    return out


EXCLUDED_SEEDS = frozenset(parse_seed_ranges(SEEDS["excluded_never_used_in_production"]))
# v8.2: every excluded seed (v5 burned 5000-5039 + all v8.1 tuning/test/adaptation/bootstrap seeds), minus the
# preflight seeds, which have their own SeedGuard mode.
BURNED_SEEDS = tuple(sorted(EXCLUDED_SEEDS - set(CFG["preflight"]["seeds"])))
INITIAL_STATES = tuple((s["T_r0"], s["T_w0"], s["q_h0"]) for s in CFG["initial_states"])

TRAIN_IVS = tuple(IV["TRAIN_INTERVENTIONS"])
HELDOUT_IVS = tuple(IV["HELDOUT_INTERVENTIONS"])
TRAIN_IDS = tuple(e["id"] for e in TRAIN_IVS)
HELDOUT_IDS = tuple(e["id"] for e in HELDOUT_IVS)
IV_BY_ID = {e["id"]: e for e in TRAIN_IVS + HELDOUT_IVS}
HELDOUT_TYPES = ("H1", "H2", "H3", "H4")

ARMS = ("WM", "GRU", "LIN", "RC_only")
COMPARATOR_ARMS_TIEBREAK = ("RC_only", "LIN", "GRU")


D_MAX_EXPECTED = 0.3269812989590207        # PLAN §8 / CONFIG selection.D_max_K_h_per_day (v8.1 tuning rows only)


def parse_comfort_bound(sel):
    """v8.2 selection comfort constraint from CONFIG['selection']: ONLY the absolute form
    'D_max_K_h_per_day' (feasible iff mean tuning D <= D_max, inclusive). Any other key/form raises SpecError."""
    if not isinstance(sel, dict) or "D_max_K_h_per_day" not in sel:
        raise SpecError("CONFIG selection must define D_max_K_h_per_day (absolute bound only)")
    other = [k for k in sel if k.startswith("D_") and k not in ("D_max_K_h_per_day", "D_max_source",
                                                                  "D_max_rejected_alternative")]
    if other:
        raise SpecError("unsupported comfort-bound keys in CONFIG selection: %r" % other)
    return dict(form="absolute", D_max=float(sel["D_max_K_h_per_day"]))


COMFORT_BOUND = parse_comfort_bound(CFG["selection"])
if COMFORT_BOUND["D_max"] != D_MAX_EXPECTED:
    raise SpecError("CONFIG D_max %r != expected %r" % (COMFORT_BOUND["D_max"], D_MAX_EXPECTED))



def structures(arm):
    return {"WM": WM_HIDDEN, "GRU": GRU_HIDDEN, "LIN": LIN_ORDERS, "RC_only": RC_STRUCTS}[arm]


def configs(arm):
    return [(s, lam) for s in structures(arm) for lam in LAMBDAS]


assert len(TUNING_SEEDS) == 20 and len(TEST_SEEDS) == 40 and len(ADAPT_SEEDS) == 10
assert all(len(TRAIN_SEEDS[r]) == N_TRAIN_EP and len(VAL_SEEDS[r]) == N_VAL_EP for r in range(REPLICATES))
assert len(TRAIN_IDS) == 6 and len(HELDOUT_IDS) == 8
assert abs(sum(CAPS.values()) - GLOBAL_CAP) < 1e-9
assert len(LAMBDAS) == 5 and CFG["arms"]["configs_per_arm"] == 3 * len(LAMBDAS)
_PROD_SEEDS = (set(TUNING_SEEDS) | set(TEST_SEEDS) | set(ADAPT_SEEDS) | set(sum((list(v) for v in TRAIN_SEEDS.values()), []))
               | set(sum((list(v) for v in VAL_SEEDS.values()), [])))
assert not (_PROD_SEEDS & EXCLUDED_SEEDS), "production seed list overlaps excluded seeds"
assert not (set(BOOT_SEEDS.values()) & EXCLUDED_SEEDS), "bootstrap seed reuses an excluded (v8.1) seed"
assert len(set(TUNING_SEEDS) | set(TEST_SEEDS) | set(ADAPT_SEEDS)) == 70
