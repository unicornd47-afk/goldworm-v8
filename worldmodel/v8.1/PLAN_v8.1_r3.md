# PLAN v8.1 r3: Thermal world model, MPC planning under held-out interventions

**Status.** This is revision r3 of PLAN_v8.1_r2.md (sha256 d0422ea2e87bae73442478bfad7c16540219e69a08630ae75e870ee4f4070c79), which stays unchanged, as does the original PLAN_v8.1.md (7ae7809f…2f2d).
- r3 changes only dahill's five mandatory fixes, plus the scope and reference notes in §12.
- It is bound to three companion files:
  - INTERVENTIONS_v8.1.json (948b5e3d5c53c9d6c4862fba1c6a4203c4f0c665d63a97be5f44dcf0bef61d9c, unchanged from r2)
  - CONFIG_v8.1_r3.json (580bba9255265a76cbc92b23645817e1e6fb622dfbf23a1f208b25eab575ee21)
  - FEASIBILITY_v8.1_r3.md (f7c772ec707a98219b517057ec55594c5d4e3cbae1ecd3229cdedff2637f3464)
- All binding files are hashed in HASHES_r3.sha256.
- **Start rule (fix 1).** Launch requires **all** of the following:
  1. dahill's plan PASS on the exact hashes in HASHES_r3.sha256;
  2. an implementation written by a non-author (not uni);
  3. an independent code-review PASS on the frozen code hashes (CODE_FROZEN manifest);
  4. a RELEASE_v8.1.json that binds HASHES_r3, the code hashes, both PASS files and the **measured toy-preflight projection** (per-structure unit costs, per-phase and total projections, applied shrink steps), with every preflight gate in §10 passing.
- If any start condition is missing, the outcome is NOT_LAUNCHED (§11).
- Dark's "execte no more request" and "gpu enabled" (via mixxy) mean no separate GO from Dark is needed once all four hold.
- One production run only: no retry, no resume, no publication.
- Where this file and CONFIG disagree, CONFIG governs the numbers and this file governs the logic.

## 0. Strand
This is the thermal world-model strand, continuing v7 Option1 (protocol v5). The plant is the Option1 plant (thermal5/plant.py, constants.py), extended only by the time-windowed parameter overrides in INTERVENTIONS. It is not GoldWormV8, which has no action or intervention interface.

## 1. Hypothesis H1 (single)
On intervention types never seen in training, tuning or selection, MPC with the GreyBox world model (WM) uses at least 5 % less heating energy than the same MPC driven by the comparator arm. The comparator is fixed on tuning seeds. Comfort must be non-inferior by an absolute margin.

## 2. Arms
All arms run inside the identical Option1 MPC: 64-candidate library, H = 12, identical candidate seeds.
- **WM GreyBox:** a 3-state RC core (T_r, T_w, q_h) plus a disturbance bias and a steady-state Kalman filter over the 48-step context, with 13 learned physical scalars and a GRU encoder-decoder residual on T_r. The hidden-size grid is {32, 64, 96}.
- **GRU:** the Option1 GRU encoder-decoder, hidden-size grid {32, 64, 96}. Parameter counts are 7458 / 27202 / 59234; WM adds 13, so the arms are matched within 0.2 %.
- **LIN:** the Option1 ARX ridge model, order grid {3, 5, 8}.
- **RC_only:** the WM core without the residual, structure grid {1-state, 2-state, 3-state}.
- **PI:** reference and fallback, frozen from Option1 (Kp 0.4, Ti 8 h, ℓ 4), no re-tuning.
- **Oracle-MPC:** descriptive only.
- **Search effort:** every arm has 3 structures x 3 energy weights λ ∈ {0.5, 1.0, 2.0} = 9 configs. Each arm gets the same 40 tuning episodes x 3 replicates per config, which is 1080 episodes and 103,680 planner decisions per arm. This is logged and gated.
- **Training:** WM, GRU and RC_only share the same loss, data, 2000 Adam updates and batch size of 16. LIN uses a closed-form fit.

## 3. Inputs and indicator channels (dahill point 4)
- **Identical inputs:** every arm sees the same 6 observables at train, tuning, test and adaptation time: y_r, T_o, setpoint, commanded u, and clock sin and cos.
- **No indicators:** no intervention has an indicator channel, whether TRAIN or HELDOUT. No arm receives type, magnitude or onset.
- **How interventions are seen:** only through their effect on measured y_r and T_o. Under H4 the model sees the commanded u, not the delayed applied u.
- **Why TRAIN gets no indicator either:** this keeps the train and test interfaces identical. A HELDOUT indicator could not have been trained anyway.
- **Static check:** Stage-0 checks that no model module imports plant, intervention or scorer code (as in Option1).

## 4. Interventions (INTERVENTIONS_v8.1.json)
- **Window:** every override is active for k ∈ [120, 168) in test and tuning episodes, a 48-step (12 h) window, and is nominal outside it.
- **TRAIN (6 entries):** 3 types, never held out.
  - T1 internal-gain step: D_BAR 0.5 → 1.0 / 1.5.
  - T2 ambient offset within the training range: −2 / +2 K.
  - T3 heater time constant at constant steady-state gain: A_QH (0.60, 4.0) → (0.45, 5.5) / (0.75, 2.5).
  - In training data, onsets cycle through 72, 120, 168 and 216.
- **HELDOUT (8 entries):** 4 types that change hidden dynamics.
  - H1 heater derating: A_QH[1] 4.0 → 3.0 / 2.4. Energy is still charged on the commanded u.
  - H2 envelope loss, i.e. a window open: A_TR[2] 0.0125 → 0.025 / 0.0375, with A_TR[0] = 1 − A_TR[1] − A_TR[2].
  - H3 wall thermal mass: wall exchange coefficients b0 and b2 scaled by 0.5 / 2.0, with b1 = 1 − b0 − b2.
  - H4 actuator dead time: 2 / 4 steps.
- **Heating headroom check (u = 1, steady state):** nominal 39.4 K, H1b 24.4 K, H2b 22.5 K, so all held-out cases are controllable for typical T_o.
- **Where HELDOUT types appear:** only in test episodes (9100-9139) and in adaptation data (8000-8009). Adaptation data is generated after the primary result is sealed. A validity gate scans every training, tuning and selection log for HELDOUT ids.

## 5. Seeds, initial states, episodes
- **Tuning:** seeds 7000-7019. Each seed gives 1 nominal episode and 1 TRAIN episode (entry = seed index mod 6), so 40 episodes per config.
- **Test:** seeds 9100-9139. Each seed is run with all 8 HELDOUT entries, so 320 episodes per instance, with common random numbers across arms.
- **Adaptation data:** seeds 8000-8009.
- **Training and validation data:**
  - Training seeds are 6100-6131 for r1, 6200-6231 for r2 and 6300-6331 for r3.
  - Validation seeds are 6140-6147 for r1, 6240-6247 for r2 and 6340-6347 for r3.
  - Each replicate has 32 training episodes: 8 nominal plus 4 for each TRAIN entry.
- **Model initialisation:** seeds 3101 / 3102 / 3103, one per replicate r.
- **Initial states (fix 3, seed-index rule):** this rule is identical for every seed list (train, val, tuning, test, adaptation).
  - The seed at 0-based position i in its list uses S[i mod 4], and **all** episodes from that seed share it.
    - Tuning: both episodes of a seed share it.
    - Test: all 8 HELDOUT episodes and the in-distribution episode of a seed share it.
    - Adaptation: all episodes of a seed share it.
  - Examples: tuning 7000 → S0, 7001 → S1, 7003 → S3, 7004 → S0; test 9100 → S0 … 9139 → S3, which gives 10 test seeds per state.
  - The states (T_r0, T_w0, q_h0) are S0 (15, 14.5, 0), S1 (17.5, 17, 2.5), S2 (20, 19, 5) and S3 (22.5, 21, 0).
  - Weather, schedule, δ0 and e_o0 come from the seed, as in Option1 draw_episode.
- **Burned seeds:** 5000-5039 are not used.

## 6. Scoring window (dahill point 2)
- **Episode structure:** each episode has 192 transitions. Steps 0-95 are warm-up under the frozen PI, identical for every arm. MPC controls steps 96-191.
- **Primary window:** steps 96-191 (all 96 control steps), identical for all arms.
- **Secondary window:** steps 120-191 (intervention plus recovery), also identical for all arms.
- **Settling time:** the first k ≥ 120 with |T_r − setpoint| < 0.5 K for 8 consecutive steps. It is logged as a secondary measure only and never defines a window or excludes an episode.

## 7. Metrics
- **E [kWh]:** the sum over the window of 2.5·u_k (commanded u; P_max 10 kW x 0.25 h).
- **C1 comfort deficit D [K·h/day]:** the sum over the window of d_k = 0.25·min(v, 11.5), with v = max(0, L − T_r) + max(0, T_r − 24), as in v5 §7.1. The primary window is one day.
- **J_score:** the sum of 2.5u + 0.25(4v + 8v²), the Option1 step score, which is independent of λ.
- **Planner cost:** λ·2.5u + 0.25(4v + 8v²).
- **Fallback:**
  - A non-finite prediction hands control to PI for the rest of the episode. The episode is counted and flagged.
  - A safety exit (T_r outside [10, 30]) charges E = 2.5 and d = 2.875 for each remaining step.

## 8. Selection on tuning seeds (dahill point 1)
- **Within each arm:** the config with the lowest mean tuning J_score over 3 replicates x 40 episodes is selected. Ties go to lower E, then the smaller structure, then λ = 1.0.
- **Comparator:** among the selected GRU, LIN and RC_only configs, the arm with the lowest mean tuning J_score is the comparator. Ties go to lower E, then the order RC_only, LIN, GRU.
- **Sealing:** the selection is written to COMPARATOR_SELECTION.json, and its sha256 is appended to the ledger before any test-seed episode is generated. Test data never enters selection.

## 9. Endpoints (HELDOUT, test seeds, primary window)
Episode values are the mean over the 3 replicates of the selected config.
- **P1:** S = 1 − ΣE_WM / ΣE_comp ≥ 0.05.
- **Bootstrap rule (fix 4), used everywhere:** B = 10,000 resamples. Every 95 % percentile CI (primary, adaptation, secondary) has lower = np.sort(x)[249] and upper = np.sort(x)[9750]. These are 0-based indices with no interpolation, and the rule is symmetric: 249 values lie below the lower endpoint and 249 above the upper.
- **P2:** paired cluster bootstrap with cluster = test seed (40 clusters, all 8 entries of a seed together). np.sort(S_boot)[249] must be > 0.
- **P3 (dahill point 3):** np.sort(ΔD_boot)[9750] of mean(D_WM − D_comp) must be **< 0.25 K·h/day**. This is an absolute margin, half of Option1's 0.5. It allows WM to be worse by less than the margin.
- **P4 safety exits (fix 5):** safety exits are counted per arm on test as the number of (episode, replicate) pairs with a safety exit, out of 320 x 3. All arms (and PI) are reported, as are tuning, in-distribution and adaptation counts. If count_WM > count_comparator, there is **no ADVANCE**; this counts as a primary failure.
- **Bootstrap reproducibility:** one index matrix (10000 x 40, int16) is shared by P2 and P3. It and the adaptation and secondary matrices are saved with sha256 before the statistics are computed.
- **Validity gates:**
  - identical-input check
  - equal search effort (9 configs x 3 x 40 per arm)
  - frozen hashes match RELEASE
  - no HELDOUT id in training, tuning or selection logs
  - COMPARATOR_SELECTION sealed before test
  - fallback episodes ≤ 10 % per arm on test
  - all caps respected
- **Key secondary, few-shot adaptation:**
  - For each HELDOUT type and K ∈ {1, 3, 10}, the adaptation episodes come from seeds 8000+, alternating magnitude a and b.
  - WM, GRU and RC_only get 100 Adam updates (batch 16, lr 1e-4) from their fitted weights. LIN gets one ridge refit on training plus adaptation windows.
  - Evaluation uses test seeds 9100-9109 x magnitude b of that type, onset 120, for the selected config x 3 replicates.
  - Reported: S and ΔD per K with bootstrap CI. These results do not decide the outcome.
- **Other secondaries:** results on the secondary window, in-distribution results (40 test seeds x 1 TRAIN entry), per-type breakdown, per-size and per-λ tuning tables (PhaseMirror), settling time, held-out 12-step MSE, and Oracle-MPC headroom.

## 10. Budget (dahill budget point, FEASIBILITY_v8.1_r3.md)
- **Expected cost:** 2640 CPU-s (44.0 min); 3300 s (55.0 min) with x1.25 contingency.
- **Hard cap:** 3600 CPU-s (60 min) on CPU, with the per-phase caps in FEASIBILITY.
- **Toy preflight (fix 2):** capped at 120 s, on seeds 1-4 only.
  - **Measurement:** per-update and per-episode cost are measured separately for every arm x structure: WM and GRU h32/h64/h96 (h=96 included), LIN order 3/5/8 and RC_only 1/2/3-state.
  - **Projection:** fits and tuning use each structure's own cost. Test, in-distribution and adaptation use the per-arm maximum over structures.
  - **Gate:** every phase projection must be at or below 0.9 x its phase cap, and the total must be at or below 3240 s.
  - **Shrink ladder:** it applies only to the adaptation, in-distribution and tuning phases (adaptation 10 → 5 seeds, then drop the in-distribution secondary, then tuning 40 → 30 episodes), and the applied steps are recorded in RELEASE.
  - **The ladder never shrinks fits:** if projected fits exceed 0.9 x the fit cap (720 s), there is no launch. The same holds for the primary test and every other phase not on the ladder.
- **GPU:** allowed by Dark but not used in this revision.

## 11. Outcomes
- **NOT_LAUNCHED:** any start condition in the Status section is missing, or a preflight gate fails after the ladder. No run takes place and there is no scientific result.
- **ADVANCE:** P1, P2, P3, P4 and all validity gates pass.
- **NO_ADVANCE:** the run is valid but P1, P2, P3 or P4 fails. The thermal world-model strand closes in this form.
- **INVALID:** the run was launched but a validity gate fails or a cap is exceeded. No scientific claim is made.

Results go to RESULTS.md honestly, and the report goes to mixxy. Nothing is published without Dark's OK. Implementation is done by a non-author, with an independent technical review before launch.

## 12. Scope limits and reference (non-blocking)
- **Plant class only:** the result applies only to this plant class, where the RC core of WM and RC_only matches the plant's RC structure. It makes no claim about mismatched-structure plants or real buildings.
- **No initial-state robustness claim:** there is no claim about robustness to initial state after the 96-step warm-up. S0-S3 are a design spread, not a robustness test.
- **Reference code:** the v3 GreyBox code exists at /workspace/experiment_weltmodell/ew/models.py (gb_step, fit_rc, GreyBoxPlanner), frozen in /workspace/experiment_weltmodell/CODE_FROZEN.sha256 (models.py 017b6362a8977acf3fbf7c4dfc67bce94f785c95e870e432cdbfc065b677209a).
  - It was written for the ew plant, so it is reference only.
  - The v8.1 WM is a port to the Option1 plant and interface, reviewed under start condition 3.
