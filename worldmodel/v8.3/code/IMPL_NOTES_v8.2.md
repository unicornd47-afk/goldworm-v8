# IMPL_NOTES v8.2 — FROZEN CODE (v8.2/code), awaiting independent code review

Implementer: TMbot · Research (non-author). Status: **frozen in `v8.2/code` (manifest `code/CODE_HASHES_v8.2.sha256`;
its sha256 is recorded in `RELEASE_v8.2_DRAFT.json`, not here, to avoid a circular hash). Toy preflight done; no
production run, no run_output.** Base: frozen v8.1 code, manifest `CODE_HASHES_v8.1.sha256` =
`be232be6b42d534c7bfd8ee3f0d6e3c75bd1c1c3129f47ab88de8e174ff4d4e8` (re-verified 29/29 OK after this work; nothing
under v8.1/ was modified). Plan: `HASHES_v8.2.sha256` = `733cc25c…b1d1` (PLAN a1e5b1cd…26c3, CONFIG 7f31dc4e…ef63,
FEASIBILITY 6ce29d10…3100, INTERVENTIONS 948b5e3d…1d9c unchanged), PASSed by dahill: `/workspace/audit_dahill_v8.2/PLAN_PASS_v8.2.json` sha256 `8ad13f896e2677498048f864ee3abed6e6534c4dcb75c889b59643de34bd49fe` (verdict PLAN_PASS, reviewer dahill, hashes_v8.2_sha256 = 304bc28d…).
Full diff: `../DIFF_v8.1_to_v8.2.patch` = `diff -ruN -x __pycache__ v8.1/code v8.2/code` with the date/time part of
every `---`/`+++` header stripped (timestamp-free, reproducible; new files included). Code dir renamed code_draft → code.
The v8.1 `IMPL_NOTES.md` (Q1–Q17) is kept unchanged and still applies except where stated here.

## Bundle in use: v8.2 + ERRATUM_v8.2_E1
`spec.BUNDLE` points at `HASHES_v8.2_E1.sha256` = `304bc28de338c92ee72817e7ed4ad498c367165e7f3ca47eed38e1732c636ff3` (lists PLAN, CONFIG, FEASIBILITY, INTERVENTIONS,
HASHES_v8.2 733cc25c…, ERRATUM_v8.2_E1.md 67b4ecdb9bd5a31a91ef2e603fe15350b7048fb98a05f8e3168e99f152938c47, E1.6 flag-only). The `*_NOT_COMPARABLE_withdrawn.*` files are ignored. CONFIG_v8.2.json itself still says λ {0.5,1,2,4,8}; the
erratum wins, so `BUNDLE["erratum_overrides"]["lambda_grid"] = [0.5, 0.625, 0.75, 0.875, 1.0]` (§E1.1) and
`spec.LAMBDAS` reads it. The sealed selection records `lambda_grid`. E1.6 (flag-only fallback) = Q6 below. E1.4 (RC_only
s2 λ=1 sits on D_max) needs no code. E1.5 (fit reproduction vs v8.1 fit_info, report only): no code; RESULTS can
compare `run_output/fit_info.json` with v8.1's (873435b1…e3e5) — see Q11. dahill PASSed bundle 304bc28d (incl. E1). `spec.GRID_OVERRIDE` is the machine-readable override record
(parameter, CONFIG value [0.5,1,2,4,8], effective value [0.5,0.625,0.75,0.875,1.0], source + sha); `check_start`
requires `RELEASE.grid_override == spec.GRID_OVERRIDE`; it is also written to PREFLIGHT_PROJECTION.json and FINAL_STATUS.json.

## What changed (minimal diff; plant, models, features, planner, closed loop, budget, ledger, stage0 untouched)

| File | Change |
|---|---|
| `v81/spec.py` | Single `BUNDLE` dict holds every plan-bundle identity (hashes file + sha, listed files, CONFIG name, INTERVENTIONS path `../v8.1/…`, plan-PASS path/sha/JSON key, release + manifest names). Swapping to an r2 bundle = edit `BUNDLE` only. `PLAN_PASS_SHA256 = "TODO_AFTER_PASS"`. Excluded seeds parsed from CONFIG `excluded_never_used_in_production` (5000-5039, 7000-7019, 9100-9139, 8000-8009, 9901-9903; preflight 1-4 handled by its own mode) → `BURNED_SEEDS`. `parse_comfort_bound()` accepts ONLY the absolute `D_max_K_h_per_day` (any other D_* key raises SpecError); import fails unless it equals `D_MAX_EXPECTED` = 0.3269812989590207. Import-time asserts: 5 λ, 15 configs/arm, production seeds ∩ excluded = ∅, bootstrap seeds ∉ excluded. Seeds, λ grid, caps all come from CONFIG_v8.2 (no hard-coding). |
| `v81/stats.py` | `select_within_arm(arm, table, d_bound)` → (config, feasible): min mean E over F={D ≤ bound}; ties lower D, smaller structure, smaller λ; F=∅ → min D; ties lower E, smaller structure, smaller λ. `select_comparator(sel)` over GRU/LIN/RC_only: feasible before infeasible; feasible: min E (tie D); none: min D (tie E); then RC_only, LIN, GRU. `comfort_bound()` (absolute D_max only, SpecError otherwise). `fallback_flags()` (E1.6). `outcome()` unchanged from v8.1. `mse_contrast_ci()` for the 6 contrasts. |
| `v81/pipeline.py` | Tuning table now records per config: mean_E, mean_D, mean_J (over all 3×40 rows), rep_mean_E, rep_mean_D, n_rows, safety_exits, fallbacks, feasible. New pure `select_from_table()` (used for sealing **and** recompute). New validity gate `selection_recomputed_from_sealed_table`: re-reads sealed COMPARATOR_SELECTION.json, recomputes bound/selection/comparator/feasibility flags, and checks sealed means == means recomputed from the per-episode tuning rows. `_heldout_mse` keeps per-(rep, episode) errors; new `_heldout_mse_ci()` saves `BOOT_INDEX_mse.npy` (seed 9924, 10000×40, sha256 in ledger before stats) and `heldout_mse_rows.npz`. New descriptive `wm_vs_each_arm_descriptive` (S, dD of WM vs GRU/LIN/RC_only, point estimates). |
| `v81/release.py` | Manifest name `CODE_HASHES_v8.2.sha256`; release `RELEASE_v8.2.json`; binds `HASHES_v8.2`; plan PASS: placeholder sha → explicit NOT_LAUNCHED reason; PASS JSON key `hashes_v8.2_sha256`. New: diff hash must be in `rel["diff"]["sha256"]` and match `v8.2/DIFF_v8.1_to_v8.2.patch`; `PREFLIGHT_PROJECTION.json` must be in `bound_files` and its own `all_gates_pass` must be true (dahill v8.1 CODE_PASS condition 1, now enforced in code). |
| `v81/preflight.py` | **Bug-for-v8.2 fixed:** tuning projection hard-coded `3 * 3` (reps × 3 λ) → `3 * len(LAMBDAS)`; HELDOUT-scan record count `4*9*3*40` → `4*15*3*40`; MSE bootstrap matrix + 6 contrasts added to the measured bootstrap_stats cost; dummy scan records use tuning seed base from spec. |
| `v81/seeds.py` | Error message / docstring no longer name v8.1 seeds (logic unchanged; lists come from spec). |
| `run_production.py`, `run_preflight.py`, `v81/__init__.py` | Release/version names from `BUNDLE`; bound paths normalised (INTERVENTIONS is `../v8.1/…`). |
| tests | Existing tests moved to v8.2 seeds (7300/9500/4100) and bootstrap seed from spec; v8.1 `test_selection_ties` replaced; release test asserts TODO_AFTER_PASS, diff and preflight-binding reasons; smoke test checks 5 λ effort, recompute gate, table fields, comfort bound, MSE contrasts, MSE matrix ledger order. New `tests/test_v82_selection.py` (29 tests incl. parametrised): bundle/CONFIG numbers, unchanged-vs-v8.1 CONFIG blocks, seed disjointness (vs v8.1 CONFIG, pairwise, bootstrap), SeedGuard rejects all v8.1 seeds and accepts v8.2 lists in order, adaptation composition 4100.., ladder (c) = 7300-7314, feasible min-E, inclusive boundary, all tie-breaks, RC/LIN structure order, infeasible fallback + ties, comparator (feasible first, none feasible, ties, ignores WM), 20 random full tables vs brute force + recompute gate detects a tampered comparator, table-vs-rows check, absolute-only comfort bound + error cases, MSE CI vs explicit loop, NaN/missing handling. |
| E1.6 flag-only (Q6) | `stats.fallback_flags` → sealed `fallback_flags` {WM_fallback, comparator_fallback, arms_fallback}, part of the recompute gate, copied to FINAL_STATUS `selection_fallback_flags`. No new outcome; precedence NOT_LAUNCHED > INVALID > ADVANCE/NO_ADVANCE (v8.1). |
| removed | `CODE_HASHES_v8.1.sha256(.sha256)` (v8.1 manifest; a v8.2 manifest is created only at freeze). |

Package directory name kept as `v81` to keep the diff minimal (Q3).

## Verification done (draft only)
- `/workspace/venv_wm/bin/python -m pytest tests` (CPU, `CUDA_VISIBLE_DEVICES=""`): **79 passed** (36 carried over from v8.1 + 43 new).
- Smoke pipeline (preflight seeds 1-4, toy design, tmp dir only): outcome field set, all validity flags true incl.
  the new recompute gate; it exercised both feasible and infeasible (fallback) arms. Numbers are meaningless.
- D_max re-derived read-only from `v8.1/run_output/tuning_rows.npz` (sha 864ac87a…dfce4): RC_only s=2, λ=1, 120 rows →
  0.3269812989590207 exactly; λ=0.5 → 0.16305200579291218 (the rejected alternative). Matches PLAN/CONFIG.
- No preflight, no production design, nothing written under v8.1/ or v7/, no RELEASE, no manifest.

## Open questions (uni / dahill)

**Q1 Comfort hypothesis HC — RESOLVED by ERRATUM_v8.2_E1 §E1.2:** no HC; dD stays descriptive. Nothing implemented.

**Q2 Plan-PASS identity — RESOLVED.** `/workspace/audit_dahill_v8.2/PLAN_PASS_v8.2.json`, sha 8ad13f89…49fe;
`check_start` verifies file sha, `verdict == PLAN_PASS`, `hashes_v8.2_sha256 == 304bc28d…`, `reviewer == dahill`.

**Q3 Package name `v81`.** Kept to minimise the diff (renaming touches every import). OK, or rename to `v82`?

**Q4 Feasibility boundary — RESOLVED (confirmed 12:44):** inclusive `mean_D <= D_max`, exact float comparison.
Means are over all 3×40 rows (= mean of replicate means, equal counts); under ladder (c) over 3×30.

**Q5 λ tie-break changed.** v8.1 preferred λ = 1.0 on ties; v8.2 PLAN/CONFIG say "smaller λ". Implemented v8.2 rule.

**Q6 Infeasible selections — RESOLVED by ERRATUM_v8.2_E1 §E1.6 (67b4ecdb…, flag-only).** A fallback (infeasible)
selection of WM or of the comparator is flagged and reported only; the run stays valid; the outcome is ADVANCE /
NO_ADVANCE via P1–P4 as predeclared; no re-selection, no rerun. Implemented as `fallback_flags` in the sealed
COMPARATOR_SELECTION.json (checked by the recompute gate) and in FINAL_STATUS.json. The briefly implemented
NOT_COMPARABLE outcome was withdrawn by the coordinator and removed completely (code, policy constant, tests).

**Q7 Comparator with mixed feasibility.** Implemented: any feasible arm beats every infeasible arm even if the
infeasible one has lower E (CONFIG wording). Confirm.

**Q8 MSE CI definition details** (CONFIG `secondary_mse_ci`): per-seed value = nanmean over that seed's
(3 replicates × up to 8 HELDOUT episodes) errors; contrast a−b per seed; point = mean over 40 seeds; bootstrap
value = mean of the differences over the resampled seeds (matrix seed 9924, 10000×40, same [249]/[9750] rule).
Episodes where the PI reference had a safety exit are dropped for all arms (v8.1 behaviour). If any seed has NaN for
an arm (failed fit / non-finite prediction), the contrast gets `ci: null` and `n_nan_seeds` — no ranking claim.
The v8.1 point MSE per arm (mean over all errors) is kept unchanged for continuity.

**Q9 "S and ΔD of WM against every non-WM arm (descriptive)".** Implemented as point estimates only (primary
window, HELDOUT test). CIs wanted? If yes, which matrix (primary 9921?).

**Q10 Excluded seeds.** Parsed from the CONFIG strings; SeedGuard now rejects 7000-7019, 9100-9139, 8000-8009 (and
5000-5039) in production. 9901-9903 are bootstrap seeds, not episode seeds; asserted not reused. Train/val/init
seeds deliberately unchanged (CONFIG `unchanged_note`).

**Q11 Fit reproducibility — E1.5: report only, no gate.** Not coded; done at RESULTS time (max abs difference of
logged fit metrics per arm × structure vs v8.1 fit_info.json). Say if you want it computed inside the run instead.

**Q12 Start-rule additions in code.** `check_start` now additionally requires (a) the diff hash and (b) the
preflight file in `bound_files` with its own `all_gates_pass`. Both follow PLAN start rule 2/4 and dahill's v8.1
condition 1; please confirm.

**Q13 Diff hash stability — RESOLVED.** Headers are made timestamp-free (run from v8/worldmodel):
`diff -ruN -x __pycache__ v8.1/code v8.2/code | sed -E 's/^((---|\+\+\+) [^\t]+)\t.*$/\1/'`.
Re-running it reproduces the same sha256. Preflight output now goes to `v8.2/PREFLIGHT_PROJECTION.json`
(`release.PREFLIGHT_FILE`).

**Q14 — RESOLVED / REMOVED.** The 'D_ref + margin' mode was removed completely (no D_ref function, no parsing,
no tests). `comfort_bound` accepts only the absolute D_max from CONFIG (0.3269812989590207, verified at import)
and raises SpecError on any other form.

## Under-specified / inconsistent points noticed in the bundle
1. CONFIG_v8.2.json still carries the old λ grid {0.5,1,2,4,8} and the old note "lambda=1.0 reproduces Option1"
   (still true); E1 overrides the grid only in prose. A reviewer must read CONFIG + E1 together; a CONFIG r2 would
   remove the override from the code.
2. CONFIG `seeds.excluded_never_used_in_production` mixes episode seeds and bootstrap seeds; harmless, handled.
3. PLAN §9 "Other secondaries" adds "S and ΔD of WM against every non-WM arm" but §13's difference list does not
   mention it (Q9).
4. Plan-PASS file path/format was not specified in the plan (resolved, Q2).
5. The v8.1 preflight formula would have under-projected v8.2 tuning (3 λ hard-coded); fixed. FEASIBILITY_v8.2's
   15/9 scaling is consistent with the fixed formula.
6. Otherwise PLAN and CONFIG agree on every number I checked: seeds, λ grid, D_max (re-derived), bootstrap seeds,
   caps (sum 3600), ladder (7300-7314, 9500-9504), adaptation eval 9500-9509, P1–P4 unchanged.

## IMPL_ERRATUM_v8.2_W1 (preflight warm-up fix; sha256 08f62233dadb4d009dfe5d7ce3bd3529337886211c87b48c1db5743c1668bdf8)
Attempt 1 of the toy preflight failed (fits 815.4 > 720, adaptation 537.4 > 468; WM_32 0.0805 s/update vs WM_64
0.0133) because the warm-up only ran a GRU. Changes, preflight only: untimed warm-up of every arm x structure
before timed measurements; `os.getloadavg()` logged before/after; plausibility gate WM_32 per-update <= WM_64
(`preflight.plausibility`, re-checked from unit costs in `check_start`); ladder restricted to step (a) and only when
adaptation fails (`spec.W1_ALLOWED_LADDER`). `spec.BUNDLE` binds W1 (`impl_errata`) and the preserved, read-only
`PREFLIGHT_PROJECTION_attempt1_FAILED.json` (f2a0e6b2…b17f7); `check_start` requires both in the RELEASE.
Tests: `tests/test_w1_preflight.py`. Exactly one fresh preflight follows the re-freeze; no repeat.

## Freeze order (done up to RELEASE DRAFT)
dahill plan PASS ✔ → PASS sha in spec ✔ → tests ✔ → code_draft → code ✔ → CODE_HASHES_v8.2.sha256 ✔ →
timestamp-free diff ✔ → toy preflight ✔ → **independent code review (open)** → RELEASE FINAL + start authority (open).
