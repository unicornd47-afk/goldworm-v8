# IMPL_NOTES v8.1 (implementation by TMbot · Research, non-author, §11)

Bound to HASHES_r3.sha256 = b2fac5bf5e57e9e7013445aba7b57b7e192aa10e4c72e49247d00d0fc6d70ed7 and
PLAN_PASS_r3.json = 2cda3f4809b119339ec21b1b7fd2a28501d4a35cc803d71c45b506a41d198727. Plan files and the v7
Option1 directory were only read, never modified. CPU only; the production run was NOT started.

Where the plan is silent or ambiguous, nothing was invented beyond the minimum needed to make the code run;
each such point is listed below as an open question (Q) with the conservative choice taken (D).
uni/dahill: please answer or accept each Q in the code review. Any change means re-freeze + new preflight.

## Open questions with the choice implemented

**Q1 LIN = "ARX ridge" (CONFIG) vs Option1 LIN.** Option1's actual LIN (thermal5/model_lin.py) is
deterministic subspace identification + finite-history Kalman predictor with ridge rel 1e-6, order 5; it is not an
ARX. Implemented: a faithful port of Option1's LIN with the state order {3,5,8} as the structure grid
(order 5 reproduces Option1 bit-for-bit, see test_lin_order5_equals_option1). t_max = 276 (Option1 B1 erratum).

**Q2 (D7) RC parameterisation behind CONFIG's 7/11/13 counts.** The plan gives counts, not equations. Implemented:
state z = [physical..., b] (b disturbance bias, AR(rho)), input w_t = [u_t, u_{t-1}, T_o_t, 1], steady-state
Kalman gains L for every state including b:
- 3-state (T_r,T_w,q_h): a1 a2 a3 a4 b1 b2 c1 c2 rho + 4 gains = 13
- 2-state (T_r,T_w): u acts on T_r directly via a3 u_t + a5 u_{t-1} (lumped heater lag), + rho + 3 gains = 11
- 1-state (T_r): a2 a3 a4 a5 rho + 2 gains = 7
Raw, unconstrained scalars (as in the ew reference); init values from ew RC_INIT where they map
(a1 .05, a2 .02, a3 .1, a4 .05, b1 .01, b2 .01, c1 .5, c2 3.0), rho .9, gains (.5,.05,0,.05); for 1/2-state
a3 = a5 = 0.3. Filter starts at t = k-47 with z = [y, (y), 0, 0]. The 48-step filter is evaluated in closed form
(powers of M = (I - L e1ᵀ)A); tested against the sequential filter.

**Q3 Future T_o over the 12-step horizon (RC core).** No arm gets future T_o (Option1 decoder inputs are u and
clock only). The RC core holds T_o at the last measured T_o[k]. GRU/LIN unchanged from Option1.

**Q4 Warm-up controller.** PLAN/CONFIG say warm-up 0..95 under the *frozen* PI (Kp 0.4, Ti 8 h, ℓ 4).
Option1 Stage-2 warm-up used the *base* PI (0.4, 2 h, 0). Implemented the plan (frozen PI). Training/val/
adaptation data episodes keep Option1's data-generation excitation (random holds / base PI + noise).

**Q5 (D1) Seed -> random stream.** Episode draw for plan seed s: np.random.default_rng(s); data excitation:
default_rng([s, 1]). All episodes of one seed (tuning: nominal + TRAIN; test: 8 HELDOUT + in-distribution)
share the same draw (weather, schedule, noise) and the initial state, so interventions are the only difference
(common random numbers across arms AND across entries). Candidate library: Option1 child (10, episode_id, k)
under the Option1 root entropy, episode_id = 16*seed + slot (N0=0, T1a..T3b=1..6, H1a..H4b=7..14).

**Q6 (D3) Training/validation composition order.** Training (32 seeds per replicate, position i): positions
0-7 nominal, then 4 per TRAIN entry in file order with onsets 72,120,168,216. Validation (8): 2 nominal, then
one per TRAIN entry with onsets cycling 72,120,168,216,72,120. Even/odd excitation rule by position as in Option1.
Validation loss is logged only (final weights are used, no checkpoint selection).

**Q7 (D5) Adaptation data format.** Plan does not fix length/excitation/onset of adaptation episodes.
Implemented: Option1-style 288-transition data episodes from seeds 8000.. (position j -> S[j mod 4], even/odd
excitation), magnitude a,b alternating, onsets cycling 72,120,168,216. K-shot uses the first K episodes of a
type. WM/GRU/RC_only: 100 Adam updates, constant lr 1e-4, batch 16, on windows of the K adaptation episodes
only (origins 48..276), from a copy of the fitted weights; LIN: one refit on train + K adaptation episodes.
Evaluation: test seeds 9100-9109 x magnitude b, onset 120 (as plan).

**Q8 Adaptation bootstrap matrix size.** FEASIBILITY says "3 matrices of 10000 x 40"; adaptation has only
10 clusters (9100-9109). Implemented 10000 x 10 for adaptation, 10000 x 40 for primary and secondary.

**Q9 (D9) Bootstrap index generation.** idx = default_rng(seed).integers(0, n_clusters, (10000, n)).astype(int16)
with seeds 9901/9902/9903; saved (.npy) with sha256 in the ledger before statistics are computed. P2/P3 share
the primary matrix; in-distribution and secondary-window CIs use the secondary matrix.

**Q10 (D10) Held-out 12-step MSE (secondary).** Defined as y_r MSE over k+1..k+12 at origin k = 120 on the
PI-reference trajectories of the 320 HELDOUT test episodes, fed the logged commanded actions; selected config x
3 replicates. Charged to the bootstrap_stats phase (included in the preflight projection).

**Q11 Loss/heads.** "Multi-step H=12 MSE on y_r" for WM, GRU, RC_only (normalized y_r). The GRU keeps
Option1's 2-output increment head (needed for the CONFIG parameter counts); the T_o output is unused.
WM residual = identical GRUCore(h) whose head output channel 0 is added to the RC T_r prediction
(normalized units); the residual head is zero-initialised (as in the ew reference), so WM starts as RC_only.

**Q12 Safety-exit and fallback accounting.** Safety exit during control at step i charges E = 2.5, d = 2.875
(J = J_max) for steps i..191 (Option1 fill_failure convention, step i included). A warm-up exit (identical
for all arms) charges all 96 steps and counts as a safety exit for every arm. A failed fit means PI for all
decisions of that instance (flagged as fallback). Fallback gate: (episode, replicate) pairs with fallback
<= 10 % of 320 x 3 per arm.

**Q13 Oracle-MPC (descriptive).** Uses lambda = 1.0, true x_k, true weather means, and knows the
intervention schedule (incl. H4 delay). PI reference runs on test (320), in-distribution (40) and the 40
adaptation-eval episodes (charged to adaptation).

**Q14 T2 ambient offset** shifts the plant's T_o and the measured T_o equally (the offset is "added to the
T_o mean").

**Q15 CPU accounting.** time.process_time (process CPU), OMP/MKL/OPENBLAS threads = 1, torch threads = 1.
Production Budget t0 = process start (imports count as overhead). Overhead = total - named phases <= 60 s.

**Q16 Preflight projection rules** (beyond the plan text): overhead = measured setup CPU (imports + hash
verification) + 15 s reserve (Option1 PROD_SUPERVISOR_RESERVE_S); LIN fit cost from a 4-episode toy fit scaled
linearly to 32 episodes (refit: 42); scenario construction and PI reference episodes are added where they occur;
one throwaway torch forward/backward is done before the per-structure measurements so lazy torch initialisation
is not charged to the first structure (its CPU counts toward the 120 s cap); if a toy episode falls back to PI,
its cost is scaled up to 96 MPC decisions.

**Q17 Start rule format.** run_production.py launches only if RELEASE_v8.1.json has status "FINAL", binds
HASHES_r3 and the PLAN_PASS hash, names a non-uni implementer, has code_review_pass = {verdict: "CODE_PASS",
file, sha256} where that file's JSON carries `code_manifest_sha256` equal to the frozen manifest hash and a
reviewer that is neither uni nor TMbot, the frozen code manifest verifies, and preflight.all_gates_pass is true.
Otherwise it writes NOT_LAUNCHED_run_output.json and generates no data.

## Implementation facts worth checking in review
- Identical inputs: every learned arm consumes the same ctx (48,6) = [y_r, T_o, s, u_{t-1}, sin, cos] and
  fut (12,3) = [u, sin, cos]; LIN consumes the Option1 subset (y_r, T_o, u, clock) of the same ObservableView.
  No intervention flag/type/magnitude/onset exists anywhere on the model side (Stage-0 import scan + view slots).
- Selection: within-arm lowest mean tuning J (λ-independent step score) over 3 reps x 40 episodes; ties lower E,
  smaller structure, λ = 1.0. Comparator among GRU/LIN/RC_only; ties lower E, then RC_only, LIN, GRU.
  COMPARATOR_SELECTION.json is sealed (sha256 in LEDGER) before any test-purpose episode can be constructed
  (SeedGuard raises otherwise); adaptation data cannot be constructed before PRIMARY_RESULT.json is sealed.
- A bug found by the unit tests during implementation (off-by-one in the closed-form filter powers) was fixed
  before freezing; test_rc_closed_form_filter_equals_sequential covers it.
- Smoke test (tests/test_smoke_pipeline.py) runs the whole production pipeline code on preflight seeds 1-4
  with tiny counts; its numbers are meaningless and are not results.
