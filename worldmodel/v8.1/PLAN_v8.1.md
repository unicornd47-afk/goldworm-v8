# PLAN v8.1 (DRAFT): Thermal world model, planning under held-out interventions

Status: FROZEN v8.1 for independent review (dahill). No code, no data, no fits until (1) dahill PASS on this exact hash, (2) RELEASE_v8.1.json verified. Per Dark ("execte no more request", "gpu enabled", 2026-11-10 via mixxy), dahill PASS on this hash plus verified release = start. Planned on CPU; GPU allowed but not expected to be needed.

## 0. Which strand, and why
v8.1 continues the **thermal world-model strand** (v8 follows the v7 world-model numbering) (v7 Option1 / GreyBox line), not GoldWormV8.
- Reason: every "no world-model advantage" result we have (Pilots 2-4, V7.2, V7.3, Option1 NO_ADVANCE with 2.417888 % saving < 10 %, and the GreyBox v3 result as summarized by dahill: comfort win, energy-cost loss vs LIN and RC-only) comes from this strand. GoldWormV8 (85M) is a code model with no action/intervention interface; using it here would change two things at once.
- If Dark wants GoldWorm as the world model instead, this draft is void and a separate plan is written.

## 1. Single preregistered hypothesis (H1)
On interventions **never seen in training**, which change hidden plant state, MPC using the physics-structured world model (WM = GreyBox) reaches **lower energy cost at non-inferior comfort** than the same MPC driven by GRU, LIN or RC-only, under identical planner search effort.
Rationale: in-distribution, flexible black-box models (GRU, LIN) already match or beat WM on MSE (Option1 Stage 1: LIN 0.040, GRU 0.048, WM 0.11-0.12 K^2). The only place a structured model should win is extrapolation to new interventions. If it does not win there either, the strand closes.

## 2. Arms (same as v3/Option1, no new arms)
WM (GreyBox), GRU, LIN, RC-only, each inside the identical MPC; PI controller as fixed reference and fallback. Fallback rules unchanged from v3.
- **Identical inputs (dahill, mandatory):** every model arm receives exactly the same action vector and the same intervention-indicator channels at train and test time. No arm gets an extra channel.
- Parameter budgets matched within +-5 % between WM-family and GRU where architecture allows; LIN/RC-only keep their natural size (documented).

## 3. Interventions (frozen before any data, dahill mandatory)
- `INTERVENTIONS_v8.1.json`: two disjoint lists, TRAIN_INTERVENTIONS and HELDOUT_INTERVENTIONS, each entry with type, magnitude, onset step, duration. Candidate held-out types: heater power derating, window-open heat-loss step, thermal-mass change, ambient step outside the training range, actuator delay change.
- It is created and hashed after this plan is frozen and before any data is generated; its SHA256 (and the config with size grid, lambda grid, initial states) is bound together with this plan's hash in a separate RELEASE_v8.1.json that dahill checks before launch. This plan file itself is never edited; any change = new plan version.
- Held-out intervention types appear in **no** training, tuning or model-selection episode.

## 4. Seeds
- Tuning seeds: 7000-7019 (tuning and planner weight search only).
- Test seeds: 9100-9139 (fresh; 5000-5039 are burned and not reused).
- Model init seeds: 3 per arm per size (PhaseMirror).

## 5. Planner cost weighting (dahill)
If the energy weight lambda in the MPC cost is changed vs v3, it is chosen from one fixed grid (to be listed with the hash) on tuning seeds only, with **the same number of planner evaluations per arm** for WM, GRU, LIN and RC-only. The search budget per arm is logged and checked as a gate.

## 6. PhaseMirror heuristics (in plan from the start)
- 3 model sizes per learnable arm (small/medium/large; exact sizes in the hash-bound config).
- 3 init seeds per size.
- Multiple initial states: >= 4 fixed initial temperatures/states per test seed, listed in the frozen config.
- Settling: settling time logged per episode; scoring only after settling, settling defined before data (e.g. first step after which |T - setpoint| < 0.5 K for 12 consecutive steps, capped at 48 steps; episodes that never settle count as comfort failures, not excluded).

## 7. Endpoints and gates (all on HELDOUT, test seeds only)
- Primary (conjunction, all must hold):
  - P1: WM-MPC energy saving vs best of {GRU, LIN, RC-only}-MPC >= 5 %, point estimate.
  - P2: paired bootstrap 95 % CI of that saving lower bound > 0.
  - P3: comfort non-inferiority: WM comfort violation (K·h) not worse than best competitor by more than 5 %, CI upper bound.
- Validity gates: identical-input check, equal search budget, frozen intervention hash, no held-out type in training logs, fallback rate < 10 % per arm.
- Key secondary (preregistered, reported with CI, not decisive): few-shot adaptation: after K=1, 3, 10 observed held-out-intervention episodes per type (fixed, from tuning seeds), each arm may update parameters with the same fixed update budget; then planning cost on test seeds. (SPARTAN, arXiv 2411.06890: structure helps mainly in adaptation, not raw prediction error.)
- Secondary (reported, not decisive): in-distribution intervention results, per-size and per-type breakdown, MSE on held-out interventions.
- **Reproducibility (fixes Option1 blocker):** all bootstrap resample index matrices are saved to disk with hashes, so every CI is independently recomputable.

## 8. Budget and execution
- CPU only, hard cap 60 CPU-minutes for the production run, plus a separate toy preflight cap. Exactly one production run, no retry, no resume, no publication.
- Implementation by a non-author worker; independent technical review before launch, as in Option1.

## 9. Outcomes
- ADVANCE: all primary + validity gates pass.
- NO_ADVANCE: valid run, primary fails. Then the thermal WM strand is closed in this form.
- INVALID: any validity gate fails; no scientific claim.

## 10. Open items for review
- Exact held-out intervention list and magnitudes (TMbot literature input welcome).
- Thresholds 5 % (P1) and 5 % (P3): reviewer may tighten.
- Exact size grid and lambda grid.
