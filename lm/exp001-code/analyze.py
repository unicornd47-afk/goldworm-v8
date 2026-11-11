#!/usr/bin/env python3
"""Frozen exp001 decision rule. Consumes raw per_chunk_bpb JSONs.

  python analyze.py --synthetic-tests
  python analyze.py --layout results/smoke_proxy --ratio proxy --seeds 7 \\
      --smoke --out results/analyze_smoke.json
  python analyze.py --layout results/arms --ratio A --out results/decision_A.json

Decision (load-bearing G vs C) [L]:
  1. For each seed and file, mean bits/byte over that file's 8 non-overlapping
     256-byte chunks, then the paired difference G − C (negative = grown better).
  2. The mean over files of that difference must be negative in ALL official
     seeds (11, 22, 33).
  3. Per file, average the paired difference over those seeds; cluster-bootstrap
     the 32 file-level means (B=10,000, seed 77701), 95% percentile CI.
  4. ADVANCE iff (2) and the CI upper bound ≤ −0.01 bpb, AND the productive-block
     and FP gates pass for every official seed. For grown arms (G, G_zeroall,
     G_signflip, and the proxy grown arm) a missing or None fp_pass fails closed.
     "If present" is allowed only for C and S.
  Otherwise NO_ADVANCE.

Official runs use exactly OFFICIAL_SEEDS. Any other seed in the inputs or the
layout raises AnalyzeError. A separate --smoke / --non-official path is required
for format checks (e.g. seed 7) and always yields NO_ADVANCE with reason
'non-official'.

G vs S is reported with the same machinery but tagged [H] and never used to ADVANCE.
The proxy rule is the same stats on grown vs S_proxy (total-FLOP); it has its own
NO_ADVANCE and also fails if the growth gate fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from constants import (
    BOOTSTRAP_B, BOOTSTRAP_SEED, MARGIN_BPB, PRODUCTIVE_FRACTION, RESULTS, SEEDS,
)

CHUNKS_PER_FILE = 8
OFFICIAL_SEEDS = tuple(SEEDS)  # (11, 22, 33)
CI_LEVEL = 0.95
# Arms that claim function preservation, or that must still record an FP gate
# (G_signflip is expected to fail; a missing sidecar is not a pass).
GROWN_ARMS = frozenset({"G", "G_zeroall", "G_signflip"})
DECISION_RULE = "exp001_v2.4"


class AnalyzeError(RuntimeError):
    """Fail-closed analysis error."""


def _load_json(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise AnalyzeError(f"missing JSON: {path}")
    return json.loads(path.read_text())


def load_eval_heldout(path: Path) -> dict:
    """Read eval_heldout.json (or a dict already containing primary_256.per_chunk_bpb)."""
    rec = _load_json(path) if not isinstance(path, dict) else path
    if "primary_256" not in rec or "per_chunk_bpb" not in rec["primary_256"]:
        raise AnalyzeError(f"not an eval_heldout.json with primary_256.per_chunk_bpb: {path}")
    chunks = [float(x) for x in rec["primary_256"]["per_chunk_bpb"]]
    if any(not np.isfinite(x) for x in chunks):
        raise AnalyzeError(f"non-finite per_chunk_bpb in {path} (fail-closed)")
    tags = list(rec.get("file_tags") or [])
    n_windows = int(rec.get("n_windows") or (len(tags) if tags else 0))
    if n_windows == 0:
        if len(chunks) % CHUNKS_PER_FILE != 0:
            raise AnalyzeError(f"{path}: chunk count {len(chunks)} not divisible by {CHUNKS_PER_FILE}")
        n_windows = len(chunks) // CHUNKS_PER_FILE
    if len(chunks) != n_windows * CHUNKS_PER_FILE:
        raise AnalyzeError(
            f"{path}: expected {n_windows}×{CHUNKS_PER_FILE}={n_windows * CHUNKS_PER_FILE} "
            f"chunks, got {len(chunks)}"
        )
    if tags and len(tags) != n_windows:
        raise AnalyzeError(f"{path}: file_tags {len(tags)} != n_windows {n_windows}")
    if not tags:
        tags = [f"file_{i}" for i in range(n_windows)]
    return {
        "per_chunk_bpb": chunks,
        "per_window_bpb": rec["primary_256"].get("per_window_bpb"),
        "bpb": rec["primary_256"].get("bpb"),
        "file_tags": tags,
        "n_files": n_windows,
        "path": str(path) if not isinstance(path, dict) else None,
    }


def file_means_from_chunks(chunks: list[float], n_files: int) -> np.ndarray:
    arr = np.asarray(chunks, dtype=np.float64).reshape(n_files, CHUNKS_PER_FILE)
    return arr.mean(axis=1)


def growth_gate_pass(arm_pass, fp_pass, *, require_fp: bool) -> bool:
    """Fail-closed productive + FP combination.

    For grown arms, fp_pass must be exactly True (missing or None fails).
    For C/S, a missing FP sidecar is allowed; a present False still fails.
    """
    if not arm_pass:
        return False
    if require_fp:
        return fp_pass is True
    return fp_pass is None or bool(fp_pass)


def load_gate(path: Path | None, *, require_fp: bool = True) -> dict:
    """Productive-block gate, optionally merged with fp_gate.json in the same directory."""
    if path is None:
        return {
            "present": False, "arm_pass": None, "fp_pass": None,
            "growth_gate_pass": False,
            "reason": "gate missing (fail-closed)",
            "require_fp": require_fp,
        }
    path = Path(path)
    rec = _load_json(path)
    arm_pass = rec.get("arm_pass")
    if arm_pass is None:
        # allow a wrapper {"productive": {...}}
        arm_pass = (rec.get("productive") or {}).get("arm_pass")
    fp_pass = rec.get("fp_pass")
    fp_path = path.parent / "fp_gate.json"
    if fp_path.is_file():
        fp = _load_json(fp_path)
        fp_pass = fp.get("pass")
    ok = growth_gate_pass(arm_pass, fp_pass, require_fp=require_fp)
    if ok:
        reason = None
    elif require_fp and fp_pass is not True:
        reason = "FP gate missing or failed (fail-closed for grown arms)"
        if not arm_pass:
            reason = "productive/FP gate failed or missing fields"
    else:
        reason = "productive/FP gate failed or missing fields"
    return {
        "present": True,
        "path": str(path),
        "arm_pass": arm_pass,
        "fp_pass": fp_pass,
        "fraction": rec.get("fraction"),
        "threshold": rec.get("threshold", PRODUCTIVE_FRACTION),
        "growth_gate_pass": ok,
        "require_fp": require_fp,
        "reason": reason,
    }


def cluster_bootstrap(file_means: np.ndarray, B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED):
    """Percentile CI of the mean, resampling clusters (files) with replacement."""
    x = np.asarray(file_means, dtype=np.float64)
    n = int(x.size)
    if n < 2:
        raise AnalyzeError(f"bootstrap needs ≥2 file clusters, got {n}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(B), n))
    boot = x[idx].mean(axis=1)
    alpha = (1.0 - CI_LEVEL) / 2.0
    lo = float(np.percentile(boot, 100.0 * alpha))
    hi = float(np.percentile(boot, 100.0 * (1.0 - alpha)))
    return {
        "point_mean": float(x.mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "B": int(B),
        "seed": int(seed),
        "n_clusters": n,
        "level": CI_LEVEL,
        "method": "percentile",
    }


def extra_seeds(*maps) -> list[int]:
    present = set()
    for m in maps:
        if not m:
            continue
        present.update(int(s) for s in m)
    return sorted(s for s in present if s not in OFFICIAL_SEEDS)


def assert_official_seeds(*maps, where: str = "inputs") -> None:
    extra = extra_seeds(*maps)
    if extra:
        raise AnalyzeError(
            f"non-official seeds in {where}: {extra}; official decision uses "
            f"exactly {list(OFFICIAL_SEEDS)} (fail-closed)"
        )


def _stack_by_seed(evals: dict[int, dict]) -> tuple[list[str], dict[int, np.ndarray]]:
    seeds = sorted(evals)
    tags = evals[seeds[0]]["file_tags"]
    n = evals[seeds[0]]["n_files"]
    out = {}
    for s in seeds:
        ev = evals[s]
        if ev["n_files"] != n or ev["file_tags"] != tags:
            raise AnalyzeError(f"file set mismatch at seed {s} (fail-closed)")
        out[s] = file_means_from_chunks(ev["per_chunk_bpb"], n)
    return tags, out


def paired_stats(left: dict[int, dict], right: dict[int, dict], label: str,
                 *, official: bool = True) -> dict:
    """left − right per file, per seed. left=G, right=C or S. Negative = left better.

    Official runs average only over OFFICIAL_SEEDS and raise AnalyzeError if any
    other seed is present in the inputs. The non-official path may use any seeds
    and is never an ADVANCE.
    """
    if official:
        assert_official_seeds(left, right, where=label)
        seeds = [s for s in OFFICIAL_SEEDS if s in left and s in right]
        if not seeds:
            raise AnalyzeError(f"{label}: no official seeds")
    else:
        seeds = sorted(set(left) & set(right))
        if not seeds:
            raise AnalyzeError(f"{label}: no overlapping seeds")
    tags, lmeans = _stack_by_seed({s: left[s] for s in seeds})
    _, rmeans = _stack_by_seed({s: right[s] for s in seeds})
    per_seed = {}
    diffs_by_seed = {}
    for s in seeds:
        d = lmeans[s] - rmeans[s]
        diffs_by_seed[s] = d
        per_seed[str(s)] = {
            "mean_diff": float(d.mean()),
            "negative": bool(d.mean() < 0.0),
            "n_files": int(d.size),
        }
    stacked = np.stack([diffs_by_seed[s] for s in seeds], axis=0)  # (S, F)
    file_means = stacked.mean(axis=0)
    boot = cluster_bootstrap(file_means)
    all_official = all(s in diffs_by_seed for s in OFFICIAL_SEEDS)
    official_neg = all(
        diffs_by_seed[s].mean() < 0.0 for s in OFFICIAL_SEEDS if s in diffs_by_seed
    ) and all_official
    return {
        "label": label,
        "seeds": seeds,
        "official": official,
        "file_tags": tags,
        "n_files": len(tags),
        "per_seed": per_seed,
        "file_means_over_seeds": [float(x) for x in file_means],
        "bootstrap": boot,
        "all_official_seeds_present": all_official,
        "all_official_seeds_negative": official_neg,
        "ci_upper_le_margin": boot["ci_hi"] <= -float(MARGIN_BPB),
        "margin_bpb": float(MARGIN_BPB),
    }


def apply_verdict(stats: dict, gates: dict[int, dict] | None, *, load_bearing: bool,
                  official: bool = True, require_fp: bool = True) -> dict:
    gates = gates or {}
    official_list = list(OFFICIAL_SEEDS)
    missing_seeds = [s for s in official_list if s not in stats["seeds"]]
    def _row(s):
        if s in gates:
            return gates[s]
        return load_gate(None, require_fp=require_fp)
    gate_rows = {str(s): _row(s) for s in official_list}
    # record any extra seeds only on the non-official path (official already raised)
    for s in stats["seeds"]:
        if s not in official_list:
            gate_rows[str(s)] = _row(s)
    official_gates_ok = (
        not missing_seeds
        and all(gate_rows[str(s)].get("growth_gate_pass") for s in official_list)
    )
    reasons = []
    if not official:
        reasons.append("non-official")
    if missing_seeds:
        reasons.append(f"missing official seeds {missing_seeds}")
    if not stats["all_official_seeds_negative"]:
        reasons.append("G−baseline mean is not negative in all official seeds")
    if not stats["ci_upper_le_margin"]:
        reasons.append(
            f"CI upper {stats['bootstrap']['ci_hi']:.6f} > {-MARGIN_BPB} (margin not met)"
        )
    if not official_gates_ok:
        reasons.append("growth gate failed or missing on an official seed")
    stats_pass = (
        official
        and not missing_seeds
        and stats["all_official_seeds_negative"]
        and stats["ci_upper_le_margin"]
    )
    if not official:
        verdict = "NO_ADVANCE"
    elif load_bearing:
        verdict = "ADVANCE" if (stats_pass and official_gates_ok) else "NO_ADVANCE"
    else:
        verdict = "HEURISTIC_PASS" if stats_pass else "HEURISTIC_FAIL"
    return {
        "verdict": verdict,
        "load_bearing": load_bearing,
        "tag": "[L]" if load_bearing else "[H]",
        "official": official,
        "stats_pass": stats_pass,
        "growth_gate_pass": official_gates_ok,
        "reasons": reasons,
        "gates": gate_rows,
        "stats": stats,
    }


def decide_growth(g, c, s=None, gates=None, *, official: bool = True,
                  require_fp: bool = True) -> dict:
    """Load-bearing G vs C plus heuristic G vs S."""
    if official:
        assert_official_seeds(g, c, s, gates, where="decide_growth")
    gc = apply_verdict(
        paired_stats(g, c, "G-C", official=official), gates,
        load_bearing=True, official=official, require_fp=require_fp,
    )
    out = {
        "decision_rule": DECISION_RULE,
        "official": official,
        "growth": gc,
        "growth_verdict": gc["verdict"],
    }
    if s is not None:
        gs = apply_verdict(
            paired_stats(g, s, "G-S", official=official), gates,
            load_bearing=False, official=official, require_fp=require_fp,
        )
        if not official:
            # heuristic path is also non-official; keep the forced tag visible
            gs["verdict"] = "HEURISTIC_FAIL"
            if "non-official" not in gs["reasons"]:
                gs["reasons"] = ["non-official"] + gs["reasons"]
        out["g_vs_s_heuristic"] = gs
    return out


def decide_proxy(g, s_proxy, gates=None, *, official: bool = True,
                 require_fp: bool = True) -> dict:
    """Own NO_ADVANCE for the total-FLOP / brute-force claim (grown vs S_proxy)."""
    if official:
        assert_official_seeds(g, s_proxy, gates, where="decide_proxy")
    rec = apply_verdict(
        paired_stats(g, s_proxy, "G-S_proxy", official=official), gates,
        load_bearing=True, official=official, require_fp=require_fp,
    )
    rec["claim"] = "growth_beats_brute_force"
    rec["official"] = official
    if official and rec["verdict"] == "ADVANCE":
        rec["proxy_verdict"] = "PROXY_ADVANCE"
    else:
        rec["proxy_verdict"] = "PROXY_NO_ADVANCE"
    rec["verdict"] = rec["proxy_verdict"]
    return rec


# ---------------------------------------------------------------------------
# Layout discovery
# ---------------------------------------------------------------------------

def discover_layout(root: Path, ratio: str, arms: tuple[str, ...] = ("G", "C", "S"),
                    seeds: tuple[int, ...] | None = None, *, official: bool = True) -> dict:
    """Find {arm}_{ratio}_seed{seed}/eval_heldout.json and productive.json.

    Official discovery uses exactly OFFICIAL_SEEDS and raises AnalyzeError if any
    other seed is present in the layout for this ratio. The seeds= filter is
    only applied on the non-official path.
    """
    root = Path(root)
    found = {a: {} for a in arms}
    gates = {}
    seed_set = set()
    layout_seeds = set()
    for p in sorted(root.glob("*_seed*/eval_heldout.json")):
        parent = p.parent.name  # e.g. G_A_seed11 or G_proxy_seed7
        # arm may contain underscores (G_zeroall); ratio is A|B|proxy
        if f"_{ratio}_seed" not in parent:
            continue
        arm, _, seed_s = parent.rpartition(f"_{ratio}_seed")
        try:
            seed = int(seed_s)
        except ValueError:
            continue
        layout_seeds.add(seed)
        if official:
            if seed not in OFFICIAL_SEEDS:
                continue  # collected in layout_seeds; raise below
        elif seeds is not None and seed not in seeds:
            continue
        if arm in found:
            found[arm][seed] = load_eval_heldout(p)
            seed_set.add(seed)
            prod = p.parent / "productive.json"
            if arm in GROWN_ARMS and prod.is_file():
                gates[seed] = load_gate(prod, require_fp=True)
    if official:
        extra = sorted(s for s in layout_seeds if s not in OFFICIAL_SEEDS)
        if extra:
            raise AnalyzeError(
                f"non-official seeds in layout {root} ratio={ratio}: {extra}; "
                f"official decision uses exactly {list(OFFICIAL_SEEDS)} (fail-closed)"
            )
        # official decision never averages a non-official seed even if passed in
        seed_set = {s for s in seed_set if s in OFFICIAL_SEEDS}
    return {
        "evals": found, "gates": gates, "seeds": sorted(seed_set),
        "layout_seeds": sorted(layout_seeds),
        "root": str(root), "ratio": ratio, "official": official,
    }


# ---------------------------------------------------------------------------
# Synthetic tests
# ---------------------------------------------------------------------------

def _synth_eval(file_means: np.ndarray, tags: list[str] | None = None) -> dict:
    """Build an eval_heldout-shaped dict whose per-file chunk means equal file_means."""
    n = int(file_means.size)
    tags = tags or [f"file_{i}" for i in range(n)]
    chunks = np.repeat(file_means.astype(np.float64), CHUNKS_PER_FILE)
    return {
        "primary_256": {
            "bpb": float(file_means.mean()),
            "n_pred_bytes": n * CHUNKS_PER_FILE * 255,
            "per_window_bpb": [float(x) for x in file_means],
            "per_chunk_bpb": [float(x) for x in chunks],
        },
        "n_windows": n,
        "file_tags": tags,
    }


def _passing_gate() -> dict:
    return {
        "arm_pass": True, "fraction": 1.0, "threshold": PRODUCTIVE_FRACTION,
        "n_new": 12, "n_productive": 12,
        "note": "synthetic passing gate",
        "fp_pass": True,
    }


def _failing_gate() -> dict:
    return {
        "arm_pass": False, "fraction": 0.0, "threshold": PRODUCTIVE_FRACTION,
        "n_new": 12, "n_productive": 0,
        "note": "synthetic failing gate",
        "fp_pass": True,
    }


def _write_eval(path: Path, ev: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ev) + "\n")


def _eval_map(means_by_seed: dict[int, np.ndarray], tags: list[str]) -> dict[int, dict]:
    return {s: load_eval_heldout(_synth_eval(m, tags)) for s, m in means_by_seed.items()}


def _gates_from(raw: dict, seeds=OFFICIAL_SEEDS, *, require_fp: bool = True) -> dict[int, dict]:
    out = {}
    for s in seeds:
        rec = raw[s]
        arm_pass = rec.get("arm_pass")
        fp_pass = rec["fp_pass"] if "fp_pass" in rec else True
        ok = growth_gate_pass(arm_pass, fp_pass, require_fp=require_fp)
        if ok:
            reason = None
        elif require_fp and fp_pass is not True:
            reason = "FP gate missing or failed (fail-closed for grown arms)"
            if not arm_pass:
                reason = "productive/FP gate failed or missing fields"
        else:
            reason = "synthetic gate fail"
        out[s] = {
            "present": True,
            "path": "synthetic",
            "arm_pass": arm_pass,
            "fp_pass": fp_pass,
            "fraction": rec.get("fraction"),
            "threshold": rec.get("threshold", PRODUCTIVE_FRACTION),
            "growth_gate_pass": ok,
            "require_fp": require_fp,
            "reason": reason,
        }
    return out


def run_synthetic_tests(out_dir: Path) -> dict:
    """Frozen cases with expected verdicts. Writes JSON + a tiny file tree."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 32
    tags = [f"pkg/file_{i}.py" for i in range(n)]
    rng = np.random.default_rng(20261111)
    # Shared C baseline around 1.20 bpb, small file effects
    base = 1.20 + 0.01 * rng.standard_normal(n)

    def pack(g_offset, c=None, flip_seed=None, gate_fail=False, s_offset=0.03,
             g_means=None, official=True, gate_raw=None, extra_seed=None):
        c_m = dict.fromkeys(OFFICIAL_SEEDS, c if c is not None else base)
        g_m = {}
        for s in OFFICIAL_SEEDS:
            if g_means is not None:
                g_m[s] = g_means[s]
            else:
                g_m[s] = (c_m[s] if isinstance(c_m[s], np.ndarray) else base) + g_offset
        if flip_seed is not None:
            g_m[flip_seed] = base - g_offset  # opposite sign
        s_m = {s: base + s_offset for s in OFFICIAL_SEEDS}  # S worse than C
        if extra_seed is not None:
            g_m[extra_seed] = base + g_offset
            c_m[extra_seed] = base
            s_m[extra_seed] = base + s_offset
        g = _eval_map(g_m, tags)
        c_ev = _eval_map(
            {s: (c_m[s] if isinstance(c_m[s], np.ndarray) else base) for s in c_m},
            tags,
        )
        s_ev = _eval_map(s_m, tags)
        if gate_raw is None:
            gate_raw = {s: (_failing_gate() if gate_fail else _passing_gate()) for s in g_m}
        return decide_growth(
            g, c_ev, s_ev, gates=_gates_from(gate_raw, seeds=tuple(g_m)),
            official=official,
        )

    cases = []

    def add(name, expected, rec, note, **extra):
        got = rec["growth_verdict"]
        ok = got == expected
        row = {
            "name": name,
            "expected": expected,
            "got": got,
            "ok": ok,
            "note": note,
            "reasons": rec["growth"]["reasons"],
            "bootstrap": rec["growth"]["stats"]["bootstrap"],
            "per_seed": rec["growth"]["stats"]["per_seed"],
            "g_vs_s_verdict": rec.get("g_vs_s_heuristic", {}).get("verdict"),
            "growth_gate_pass": rec["growth"]["growth_gate_pass"],
        }
        row.update(extra)
        cases.append(row)

    # (a) clear ADVANCE: G better by 0.05 in every seed, tight CI, gate passes
    rec_a = pack(g_offset=-0.05)
    add("a_clear_ADVANCE", "ADVANCE", rec_a,
        "G−C ≈ −0.05 in all 3 seeds; CI well below −0.01; gate passes")

    # (b) sign flip in one seed → NO_ADVANCE
    rec_b = pack(g_offset=-0.05, flip_seed=33)
    add("b_sign_flip_one_seed", "NO_ADVANCE", rec_b,
        "seeds 11 and 22 negative, seed 33 flipped positive")

    # (c) just misses margin: all seeds slightly negative (~−0.006), CI upper > −0.01
    rec_c = pack(g_offset=-0.006)
    add("c_just_misses_margin", "NO_ADVANCE", rec_c,
        "all seeds negative but mean ≈ −0.006, CI upper > −0.01")

    # (d) gate-fail: same as (a) but productive gate fails → not ADVANCE
    rec_d = pack(g_offset=-0.05, gate_fail=True)
    add("d_gate_fail", "NO_ADVANCE", rec_d,
        "stats would ADVANCE; productive-block gate fails")

    # (e) otherwise clear ADVANCE plus a stray seed 7 → AnalyzeError (official path)
    e_ok = False
    e_got = None
    e_err = None
    try:
        pack(g_offset=-0.05, extra_seed=7, official=True)
        e_got = "no_error"
    except AnalyzeError as err:
        e_got = "AnalyzeError"
        e_err = str(err)
        e_ok = True
    cases.append({
        "name": "e_stray_seed_raises",
        "expected": "AnalyzeError",
        "got": e_got,
        "ok": e_ok,
        "note": "clear ADVANCE plus stray seed 7 on the official path raises AnalyzeError (fail-closed; not averaged into the CI)",
        "error": e_err,
    })

    # (f) otherwise clear ADVANCE where G's fp_pass is None → NO_ADVANCE
    raw_f = {s: {**_passing_gate(), "fp_pass": None} for s in OFFICIAL_SEEDS}
    rec_f = pack(g_offset=-0.05, gate_raw=raw_f)
    add("f_missing_fp_pass", "NO_ADVANCE", rec_f,
        "stats would ADVANCE; G fp_pass is None, grown-arm FP gate fails closed")

    # (g) realistic noise around the margin (true mean −0.012); no fixed verdict
    rng_g = np.random.default_rng(424242)
    noise = 0.004 * rng_g.standard_normal((len(OFFICIAL_SEEDS), n))
    g_means_margin = {
        s: base - 0.012 + noise[i] for i, s in enumerate(OFFICIAL_SEEDS)
    }
    rec_g = pack(g_offset=0.0, g_means=g_means_margin)
    rec_g2 = pack(g_offset=0.0, g_means=g_means_margin)
    b_g = rec_g["growth"]["stats"]["bootstrap"]
    b_g2 = rec_g2["growth"]["stats"]["bootstrap"]
    g_consistent = (
        rec_g["growth_verdict"] == rec_g2["growth_verdict"]
        and b_g["ci_lo"] == b_g2["ci_lo"]
        and b_g["ci_hi"] == b_g2["ci_hi"]
        and b_g["point_mean"] == b_g2["point_mean"]
    )
    # Rule applied consistently: recompute from the recorded clauses.
    rule_ok = (
        (rec_g["growth_verdict"] == "ADVANCE")
        is (
            rec_g["growth"]["stats"]["all_official_seeds_negative"]
            and rec_g["growth"]["stats"]["ci_upper_le_margin"]
            and rec_g["growth"]["growth_gate_pass"]
        )
    )
    cases.append({
        "name": "g_noisy_around_margin",
        "expected": "rule_consistent",
        "got": rec_g["growth_verdict"],
        "ok": g_consistent and rule_ok,
        "note": "true mean −0.012 plus per-seed/per-file noise, RNG seed 424242; verdict is whatever the frozen rule returns, twice identically",
        "reasons": rec_g["growth"]["reasons"],
        "bootstrap": b_g,
        "per_seed": rec_g["growth"]["stats"]["per_seed"],
        "growth_gate_pass": rec_g["growth"]["growth_gate_pass"],
        "repeat_verdict": rec_g2["growth_verdict"],
        "consistent": g_consistent,
        "rule_applied": rule_ok,
    })

    # clearly-advance noisy case: mean −0.05 plus modest noise must still ADVANCE
    rng_h = np.random.default_rng(434343)
    noise_h = 0.005 * rng_h.standard_normal((len(OFFICIAL_SEEDS), n))
    g_means_clear = {
        s: base - 0.05 + noise_h[i] for i, s in enumerate(OFFICIAL_SEEDS)
    }
    rec_h = pack(g_offset=0.0, g_means=g_means_clear)
    add("g_noisy_clear_ADVANCE", "ADVANCE", rec_h,
        "true mean −0.05 plus modest per-seed/per-file noise must still ADVANCE; RNG seed 434343")

    # also a proxy variant of (a): grown vs S_proxy should PROXY_ADVANCE
    g = _eval_map({s: base - 0.05 for s in OFFICIAL_SEEDS}, tags)
    s_p = _eval_map({s: base + 0.03 for s in OFFICIAL_SEEDS}, tags)
    proxy_a = decide_proxy(g, s_p, gates=_gates_from({s: _passing_gate() for s in OFFICIAL_SEEDS}))
    proxy_ok = proxy_a["proxy_verdict"] == "PROXY_ADVANCE"
    cases.append({
        "name": "proxy_clear_PROXY_ADVANCE",
        "expected": "PROXY_ADVANCE",
        "got": proxy_a["proxy_verdict"],
        "ok": proxy_ok,
        "note": "proxy rule on grown vs S_proxy, same stats, gate passes",
        "reasons": proxy_a["reasons"],
        "bootstrap": proxy_a["stats"]["bootstrap"],
        "per_seed": proxy_a["stats"]["per_seed"],
        "growth_gate_pass": proxy_a["growth_gate_pass"],
    })

    report = {
        "n_cases": len(cases),
        "PASS": all(c["ok"] for c in cases),
        "official_seeds": list(OFFICIAL_SEEDS),
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "margin_bpb": MARGIN_BPB,
        "chunks_per_file": CHUNKS_PER_FILE,
        "decision_rule": DECISION_RULE,
        "cases": cases,
    }
    dest = out_dir / "analyze_synthetic_tests.json" if out_dir.name != "analyze_synthetic_tests.json" else out_dir
    if dest.is_dir() or str(out_dir).endswith("analyze_tests") or out_dir.name == "analyze_tests":
        dest = out_dir / "analyze_synthetic_tests.json"
    # always also write the canonical results path
    canon = RESULTS / "analyze_synthetic_tests.json"
    canon.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2) + "\n"
    canon.write_text(text)
    if dest != canon:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _summarize_growth(rec: dict) -> dict:
    g = rec["growth"]
    b = g["stats"]["bootstrap"]
    return {
        "verdict": rec["growth_verdict"],
        "official": rec.get("official"),
        "point_mean_G_minus_C": b["point_mean"],
        "ci": [b["ci_lo"], b["ci_hi"]],
        "all_official_seeds_negative": g["stats"]["all_official_seeds_negative"],
        "all_official_seeds_present": g["stats"]["all_official_seeds_present"],
        "growth_gate_pass": g["growth_gate_pass"],
        "reasons": g["reasons"],
        "g_vs_s": (rec.get("g_vs_s_heuristic") or {}).get("verdict"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="exp001 frozen decision rule")
    ap.add_argument("--synthetic-tests", action="store_true")
    ap.add_argument("--layout", type=Path, help="Directory containing {arm}_{ratio}_seedN/ folders")
    ap.add_argument("--ratio", default="A", choices=("A", "B", "proxy"))
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="Non-official seed filter. Official decision always uses 11/22/33.")
    ap.add_argument("--proxy", action="store_true",
                    help="Use the proxy decision rule (grown vs S, total-FLOP claim).")
    ap.add_argument("--smoke", action="store_true",
                    help="Non-official path (format checks). Always NO_ADVANCE, reason 'non-official'.")
    ap.add_argument("--non-official", action="store_true",
                    help="Same as --smoke.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tests-dir", type=Path, default=RESULTS)
    args = ap.parse_args()

    if args.synthetic_tests:
        report = run_synthetic_tests(args.tests_dir)
        print(json.dumps({
            "PASS": report["PASS"],
            "n_cases": report["n_cases"],
            "cases": [{k: c[k] for k in ("name", "expected", "got", "ok", "note") if k in c}
                      for c in report["cases"]],
        }, indent=2))
        return 0 if report["PASS"] else 1

    if not args.layout:
        print("Need --synthetic-tests or --layout", file=sys.stderr)
        return 2

    official = not (args.smoke or args.non_official)
    seeds = tuple(args.seeds) if args.seeds else None
    if official and seeds is not None and tuple(sorted(set(seeds))) != tuple(sorted(OFFICIAL_SEEDS)):
        raise AnalyzeError(
            f"--seeds {seeds} is not exactly OFFICIAL_SEEDS {list(OFFICIAL_SEEDS)}; "
            "use --smoke / --non-official for a format check (fail-closed)"
        )
    found = discover_layout(
        args.layout, args.ratio, arms=("G", "C", "S"), seeds=seeds, official=official,
    )
    g, c, s = found["evals"]["G"], found["evals"]["C"], found["evals"]["S"]
    if not g:
        raise AnalyzeError(f"no G evals under {args.layout} ratio={args.ratio}")
    if args.proxy:
        if not s:
            raise AnalyzeError("proxy rule needs S (S_proxy) evals")
        rec = {"proxy": decide_proxy(g, s, gates=found["gates"], official=official), "layout": found}
        rec["proxy_verdict"] = rec["proxy"]["proxy_verdict"]
        summary = {
            "proxy_verdict": rec["proxy_verdict"],
            "official": official,
            "reasons": rec["proxy"]["reasons"],
            "bootstrap": rec["proxy"]["stats"]["bootstrap"],
            "per_seed": rec["proxy"]["stats"]["per_seed"],
        }
    else:
        if not c:
            raise AnalyzeError("growth rule needs C evals")
        rec = decide_growth(g, c, s if s else None, gates=found["gates"], official=official)
        rec["layout"] = {k: found[k] for k in ("root", "ratio", "seeds", "official", "layout_seeds") if k in found}
        summary = _summarize_growth(rec)
    rec["summary"] = summary
    text = json.dumps(rec, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AnalyzeError as e:
        print(f"ANALYZE_FAIL {e}", file=sys.stderr)
        sys.exit(1)
