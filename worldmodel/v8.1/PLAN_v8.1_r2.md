# PLAN v8.1 r2: Thermal world model, MPC planning under held-out interventions

**Status.** This is the complete, freezable revision of PLAN_v8.1.md (sha256 7ae7809fe4663d308437861d547977e083cec67f540f0c8537fabf30b35b2f2d), which stays unchanged.
- It is bound to three companion files:
  - INTERVENTIONS_v8.1.json (948b5e3d5c53c9d6c4862fba1c6a4203c4f0c665d63a97be5f44dcf0bef61d9c)
  - CONFIG_v8.1.json (ab0076e414297e920b6e1394c6f6db067971b4910fb3768255a5c2449c71e1fd)
  - FEASIBILITY_v8.1.md (f5f95e65698e078a3536522522a60bdbaa7cd6a6c6ff7f5fb4105976c0bf0c93)
- All four files are hashed in HASHES_r2.sha256.
- Start rule (Dark via mixxy: "execte no more request", "gpu enabled"): dahill PASS on these exact four hashes, plus a verified RELEASE_v8.1.json, means start.
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
- **Initial states:** episode i in a seed list starts from S[i mod 4]. The states (T_r0, T_w0, q_h0) are S0 (15, 14.5, 0), S1 (17.5, 17, 2.5), S2 (20, 19, 5) and S3 (22.5, 21, 0). Weather, schedule, δ0 and e_o0 come from the seed, as in Option1 draw_episode.
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
- **P2:** paired cluster bootstrap with cluster = test seed (40 clusters, all 8 entries of a seed together) and 10,000 resamples. The lower endpoint of S, the 250th sorted value, must be > 0.
- **P3 (dahill point 3):** the upper endpoint (9750th) of mean(D_WM − D_comp) must be **< 0.25 K·h/day**. This is an absolute margin, half of Option1's 0.5. It allows WM to be worse by less than the margin.
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

## 10. Budget (dahill budget point, FEASIBILITY_v8.1.md)
- **Expected cost:** 2640 CPU-s (44.0 min); 3300 s (55.0 min) with x1.25 contingency.
- **Hard cap:** 3600 CPU-s (60 min) on CPU, with the per-phase caps in FEASIBILITY.
- **Toy preflight:** capped at 120 s on seeds 1-4. It projects the full cost. Above 3240 s, the predeclared shrink ladder applies (adaptation 10 → 5 seeds, then drop the in-distribution secondary, then tuning 40 → 30 episodes) and is recorded in RELEASE. If the projection is still above 3240 s, there is no launch.
- **GPU:** allowed by Dark but not used in this revision.

## 11. Outcomes
- **ADVANCE:** P1, P2, P3 and all validity gates pass.
- **NO_ADVANCE:** the run is valid but the primary fails. The thermal world-model strand closes in this form.
- **INVALID:** any gate fails or any cap is exceeded. No scientific claim is made.

Results go to RESULTS.md honestly, and the report goes to mixxy. Nothing is published without Dark's OK. Implementation is done by a non-author, with an independent technical review before launch.
