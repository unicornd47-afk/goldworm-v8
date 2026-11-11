# PLAN v8.2: Thermal world model, energy-targeted selection under a comfort budget

**Status.** This is a new preregistration in the thermal strand.
- **Base:** v8.1 r3 (HASHES_r3.sha256 b2fac5bf5e57e9e7013445aba7b57b7e192aa10e4c72e49247d00d0fc6d70ed7) plus ERRATUM_r3_IMPL_v1.md (86a0d2f26703f7c6d9e77953b8e38ae39528cdbd80747f7097a349316de65373, E1/E2). The v8.1 files are untouched.
- **v8.1 outcome:** NO_ADVANCE. S = −1.91 % against RC_only, and dD = −1.15 K·h/day, so WM traded energy for comfort. Every arm's config and λ had been selected on the combined J_score. Audit: /workspace/audit_dahill_v8.1/RESULT_AUDIT_v8.1.json, RESULT_CONFIRMED_NO_ADVANCE.
- **The only intended experimental change:** selection targets energy under a fixed absolute comfort budget (§8). The λ grid, fresh seeds, the RESULTS wording rules and the reallocated caps follow from that change or from the request; see §13 for the full list.
- **Binding files:**
  - CONFIG_v8.2.json (7f31dc4e43e7899016e0c049c2e3ed251d278235f1fbc059f250ff5bc419ef63)
  - FEASIBILITY_v8.2.md (6ce29d105e1f248a10c5a38e38f74db7173dc0e6eb721858870e6c4608ee3100)
  - ../v8.1/INTERVENTIONS_v8.1.json (948b5e3d5c53c9d6c4862fba1c6a4203c4f0c665d63a97be5f44dcf0bef61d9c), reused unchanged
  - All of them are hashed in HASHES_v8.2.sha256.
- **Start rule.** Launch requires **all** of the following:
  1. dahill's plan PASS on the exact hashes in HASHES_v8.2.sha256;
  2. a non-author implementation (TMbot) that reuses the v8.1 frozen code (manifest CODE_HASHES_v8.1.sha256 be232be6b42d534c7bfd8ee3f0d6e3c75bd1c1c3129f47ab88de8e174ff4d4e8) with a minimal diff (selection rule, λ grid, seed lists, bootstrap seeds, MSE CI, caps), with the diff hashed;
  3. an independent code-review PASS on the frozen v8.2 code hashes;
  4. a RELEASE_v8.2.json that binds HASHES_v8.2, the code hashes, the diff hash, both PASS files and the measured toy-preflight per-structure costs, per-phase and total projections and any ladder steps, with every gate in §10 passing.
- If any start condition is missing, the outcome is NOT_LAUNCHED (§11).
- **Start authority:** Dark's standing autonomy rule of 2026-11-10 12:24 Europe/Berlin, as relayed by mixxy. It is relayed and cannot be verified independently by the plan author or the reviewer.
- One production run only: no retry, no resume, no publication without Dark's OK.
- Where this file and CONFIG disagree, CONFIG governs the numbers and this file governs the logic.

## 0. Strand
This is the thermal world-model strand: the v7 Option1 plant plus the v8.1 time-windowed intervention overrides, unchanged.

## 1. Hypothesis H1 (single)
On intervention types never seen in training, tuning or selection, MPC with the GreyBox world model (WM) uses at least 5 % less heating energy than the same MPC driven by the comparator arm, with comfort non-inferior by an absolute margin.
- Every arm, the comparator included, is selected on tuning seeds by **minimum energy subject to the same absolute comfort budget D_max**.

## 2. Arms
The arms are unchanged from v8.1 r3, and LIN is as corrected in E1. All run inside the identical Option1 MPC: 64 candidates, H = 12, identical candidate seeds.
- **WM GreyBox:** RC core with Kalman filter plus a GRU encoder-decoder residual, h ∈ {32, 64, 96}.
- **GRU:** h ∈ {32, 64, 96}.
- **LIN:** thermal5/model_lin.py subspace identification plus a finite-history Kalman predictor, ridge rel 1e-6, order ∈ {3, 5, 8}.
- **RC_only:** structure ∈ {1, 2, 3}-state.
- **PI:** frozen (0.4, 8 h, 4), used as reference and fallback.
- **Oracle-MPC:** descriptive only.
- **λ grid (changed):** {0.5, 1, 2, 4, 8}, extended toward energy, where λ multiplies the planner's energy term.
  - 0.5 is kept because in v8.1 tuning it was the only λ at which most configs reached the comfort level of the v8.1 selections.
  - 4 and 8 extend the grid toward the energy end.
- **Equal search effort:** 3 structures x 5 λ = 15 configs per arm, each with 3 replicates x 40 tuning episodes. That is 1800 episodes and 172,800 planner decisions per arm, logged and gated.
- **Training:** unchanged (2000 Adam updates, batch 16, same loss; LIN closed form).

## 3. Inputs and indicator channels
These are unchanged from v8.1 r3 §3. Every arm sees the same 6 observables, no intervention has an indicator channel, and the Stage-0 import check applies.

## 4. Interventions
These are unchanged: the INTERVENTIONS_v8.1.json file is reused by hash. The window is k ∈ [120, 168).
- **TRAIN:** T1, T2 and T3, two magnitudes each.
- **HELDOUT:** H1 heater derating, H2 envelope loss, H3 wall mass and H4 dead time, two magnitudes each.
- **Where HELDOUT types appear:** only in test episodes (9500-9539) and in adaptation data (4100-4109, generated after the primary result is sealed). The HELDOUT-id scan over training, tuning and selection logs remains a gate.

## 5. Seeds, initial states, episodes (fresh tuning, test and adaptation seeds)
- **Tuning:** seeds 7300-7319. Each seed gives 1 nominal episode and 1 TRAIN episode (entry = seed position mod 6), so 40 episodes per config.
- **Test:** seeds 9500-9539, each run with all 8 HELDOUT entries, so 320 episodes per instance, with common random numbers across arms.
- **Adaptation data:** seeds 4100-4109. Adaptation evaluation uses test seeds 9500-9509.
- **Bootstrap seeds (new):** 9921 primary, 9922 adaptation, 9923 secondary, 9924 MSE.
- **Never used in production:** 5000-5039, 7000-7019, 9100-9139, 8000-8009, the v8.1 bootstrap seeds 9901-9903, and 1-4 (preflight only).
- **Unchanged on purpose:** the train and validation data seeds (6100-6131 / 6200-6231 / 6300-6331 and 6140-6147 / 6240-6247 / 6340-6347) and the model-init seeds (3101-3103).
  - The fits are then expected to reproduce the v8.1 fits, which isolates the selection change.
  - These seeds are neither tuning nor test data.
- **Initial states:** the seed-index rule is unchanged. The seed at 0-based position i in any list uses S[i mod 4], and all episodes from that seed share it.
  - Examples: tuning 7300 → S0, 7303 → S3, 7304 → S0; test 9500 → S0 … 9539 → S3, which gives 10 test seeds per state.
  - The states are S0 (15, 14.5, 0), S1 (17.5, 17, 2.5), S2 (20, 19, 5) and S3 (22.5, 21, 0).

## 6. Scoring window
This is unchanged. Warm-up runs under PI for steps 0-95. The primary window is steps 96-191 and the secondary window steps 120-191. Settling time is logged only.

## 7. Metrics
These are unchanged.
- **E:** the sum of 2.5·u_k [kWh].
- **D (C1):** the sum of 0.25·min(v, 11.5) [K·h/day].
- **J_score:** the sum of 2.5u + 0.25(4v + 8v²). It is reported, but no longer used for selection.
- **Planner cost:** λ·2.5u + 0.25(4v + 8v²).
- **Fallback and safety-exit charging:** as in v8.1 r3.

## 8. Selection on tuning seeds (CHANGED: energy-targeted)
- **D_max = 0.3269812989590207 K·h/day**, fixed before any v8.2 data.
  - **Source:** this is the mean tuning D of the v8.1 comparator structure (RC_only, 2-state) at λ = 1.0, over its 120 v8.1 **tuning** rows (3 replicates x 40 episodes, seeds 7000-7019). They come from v8.1/run_output/tuning_rows.npz, sha256 864ac87a67addcc727d1b1f8a655431600c866b8f9badfe6d3c10f567fffdce4.
  - λ = 1.0 is the neutral weighting, where the planner cost equals J_score.
  - No v8.1 test, in-distribution, adaptation or secondary data was used.
  - **Rejected alternative:** 0.16305200579291218, the same structure at its J-selected λ = 0.5. It would reproduce v8.1's J-driven comfort level: in v8.1 tuning no λ ≥ 1 config of any arm met it, so selection could not lean toward energy.
- **Within each arm (identical for WM, GRU, LIN and RC_only):**
  - F is the set of the 15 configs whose mean tuning D (3 replicates x 40 episodes) is at or below D_max.
  - If F is non-empty, the arm takes the config in F with the lowest mean tuning E. Ties go to lower D, then the smaller structure, then the smaller λ.
  - If F is empty, the arm takes its lowest-mean-D config (predeclared fallback). Ties go to lower E, then the smaller structure, then the smaller λ.
- **Comparator:** chosen among the selected GRU, LIN and RC_only configs.
  - Arms whose selection is feasible rank before infeasible ones.
  - Among feasible arms, the lowest mean tuning E wins. If no arm is feasible, the lowest mean tuning D wins.
  - Ties go to the other metric, then the order RC_only, LIN, GRU.
- **Sealing:** COMPARATOR_SELECTION.json holds the full per-config table (E, D, J_score, feasibility). Its sha256 is written to the ledger before any test-seed episode is generated. Test data never enters selection.

## 9. Endpoints (HELDOUT, test seeds, primary window)
These are unchanged. Episode values are the mean over the 3 replicates of the selected config.
- **P1:** S = 1 − ΣE_WM / ΣE_comp ≥ 0.05.
- **P2:** paired cluster bootstrap (40 test-seed clusters). np.sort(S_boot)[249] must be > 0.
- **P3:** np.sort(ΔD_boot)[9750] of mean(D_WM − D_comp) must be < 0.25 K·h/day (absolute).
- **P4:** if WM's safety-exit count on test (out of 320 x 3) is greater than the comparator's, there is no ADVANCE. All arms are reported.
- **Bootstrap rule (everywhere):** B = 10,000, lower = np.sort(x)[249], upper = np.sort(x)[9750] (0-based).
  - Primary, secondary and MSE matrices are 10000 x 40; the adaptation matrix is 10000 x 10 (E2).
  - All matrices are saved with sha256 before the statistics are computed.
- **Validity gates:**
  - as in v8.1 r3, with equal search effort now 15 configs x 3 x 40 per arm;
  - plus a new gate: the sealed selection must follow the §8 rule exactly. A reviewer recomputes it from the sealed table.
- **Key secondary, few-shot adaptation:** unchanged except for seeds. K ∈ {1, 3, 10}, 100 updates (LIN: one refit), adaptation data from 4100-4109, evaluation on 9500-9509 x magnitude b.
- **New secondary, held-out MSE with CI:**
  - The metric is the 12-step y_r MSE (v8.1 definition).
  - A paired cluster-bootstrap CI (seed 9924) is computed for the per-seed mean differences WM−GRU, WM−RC_only, WM−LIN, GRU−RC_only, GRU−LIN and RC_only−LIN.
- **Other secondaries:**
  - secondary window, in-distribution, per-type breakdown, settling time, Oracle headroom;
  - the full tuning E-D table per arm, structure and λ;
  - S and ΔD of WM against every non-WM arm (descriptive).

## 10. Budget (FEASIBILITY_v8.2.md, measured v8.1 costs)
- **Projected cost:** 1950.3 CPU-s (32.5 min); 2437.9 s (40.6 min) with x1.25 contingency.
- **Hard cap:** 3600 CPU-s (60 min), CPU only.
- **Per-phase caps:**

| Phase | Cap (s) |
|---|---|
| Overhead | 60 |
| Data | 30 |
| Fits | 800 |
| Tuning | 1100 |
| Test | 800 |
| In-distribution | 120 |
| PI/Oracle | 30 |
| Adaptation | 520 |
| Bootstrap/stats | 60 |
| Audit | 80 |

- **Preflight:** as in v8.1 r3, on seeds 1-4 with a 120 s cap. Each of the 12 arm x structure combinations is measured separately. Every phase projection must be at or below 0.9 x its cap, and the total at or below 3240 s.
- **Shrink ladder:** it applies only to adaptation, in-distribution and tuning (adaptation 10 → 5 seeds, then drop the in-distribution secondary, then tuning 40 → 30 episodes).
- **Never shrunk:** the λ grid, the structure grid, fits and the primary test. Any of these above 0.9 x its cap means no launch.

## 11. Outcomes
- **NOT_LAUNCHED:** any start condition is missing, or a preflight gate fails after the ladder.
- **ADVANCE:** P1, P2, P3, P4 and all validity gates pass.
- **NO_ADVANCE:** the run is valid but P1, P2, P3 or P4 fails.
- **INVALID:** the run was launched but a validity gate fails or a cap is exceeded.

## 12. Rules for RESULTS_v8.2.md (dahill wording points)
1. **MSE rankings:** stated only with the paired bootstrap CI from §9. Without a CI there is no ranking claim ("practically level" at most, with the numbers).
2. **"Refuted" / "widerlegt":** used only relative to this protocol (this plant, these interventions, energy-under-D_max selection, D_max = 0.32698 K·h/day), never in general.
3. **Literature citations:** a source may be cited only after it has been checked, and only for what it states.
   - **SPARTAN (Lei, Schölkopf, Posner, arXiv 2411.06890)** was checked by the plan author on 2026-11-10 against the arXiv abstract and paper.
     - It presents a sparse Transformer world model that learns local causal interaction graphs between object tokens.
     - It reports prediction error similar to a Transformer baseline (Table 1: Pong 8.60 vs 8.83, Traffic 0.52 vs 0.51).
     - It reports better graph accuracy (SHD), robustness to removed non-causal entities, and **better few-shot adaptation** (5 trajectories) to dynamics changes, including unseen environments.
     - **It makes no claim about planning or control performance.** It must not be cited for "better prediction does not translate into better planning" (the v8.1 RESULTS usage). At most it can be cited for "structure helped few-shot adaptation in their object-centric benchmarks, at similar prediction error".
   - Any other citation needs the same check, recorded in RESULTS, or it is dropped.

## 13. Scope limits (unchanged) and differences vs v8.1 r3
**Scope:**
- The result covers this plant class only, where the RC core matches the plant structure.
- There is no initial-state robustness claim after the 96-step warm-up.
- The known v8.1 confounder is unchanged and identical for all arms: in the training data, onset 120, S1 and PI excitation coincide.
- The v3 GreyBox code (/workspace/experiment_weltmodell/ew/models.py, 017b6362…) remains reference only.

**Differences vs v8.1 r3 (+E1/E2):**
1. **Selection:** minimum mean tuning E subject to mean tuning D ≤ D_max = 0.3269812989590207 (from v8.1 tuning rows only), with a lowest-D fallback. The comparator is the lowest E among feasible GRU, LIN and RC_only. This replaces selection on min J_score.
2. **λ grid:** {0.5, 1, 2} → {0.5, 1, 2, 4, 8}, so configs per arm go from 9 to 15 (1800 tuning episodes and 172,800 decisions per arm).
3. **Seeds:** tuning 7000-7019 → 7300-7319; test 9100-9139 → 9500-9539; adaptation 8000-8009 → 4100-4109; adaptation evaluation 9100-9109 → 9500-9509; bootstrap seeds 9901-9903 → 9921-9923, plus 9924 (MSE). Train, validation and init seeds are unchanged.
4. **New secondary:** paired bootstrap CI for held-out MSE contrasts.
5. **New validity gate:** recompute the sealed selection from its table.
6. **New RESULTS wording rules** (§12), including the checked SPARTAN scope.
7. **Caps reallocated within 3600 s:** tuning 1000 → 1100, test 900 → 800. Feasibility is recomputed from the measured v8.1 costs.
8. **Start rule:** names TMbot as the implementer, requires reuse of the v8.1 frozen code with a hashed minimal diff, and names the 2026-11-10 12:24 standing autonomy rule (relayed by mixxy) as start authority. The RELEASE file is now RELEASE_v8.2.json.
9. **E1 and E2 folded in:** LIN described correctly; adaptation bootstrap matrix 10000 x 10.
