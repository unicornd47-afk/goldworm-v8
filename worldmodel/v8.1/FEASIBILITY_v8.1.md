# FEASIBILITY v8.1 (r2): CPU budget from measured Option1 timings

This file contains arithmetic only. No data was generated, nothing was fitted and nothing was simulated. Source: `v7/worldmodel/option1_thermal_v5/run_output/STATUS.json` (phase cpu_s, run ffda3713…) and `artifacts/RESULTS.md` §8.

## Measured unit costs (Option1, CPU, single process)
| Item | Measured | Derived unit |
|---|---|---|
| GRU fit, h=64, 2000 updates, batch 16 | 14.48 / 13.25 / 13.49 s | 13.74 s per fit |
| WM (RSSM) fit, 2000 updates | 51.08 / 50.11 / 49.70 s | 50.30 s per fit (25 ms/update, 48-step recurrent loop) |
| LIN fit | 0.037 / 0.025 / 0.024 s | 0.029 s |
| Stage-2 closed loop, 16 scen x 96 decisions x 64 candidates | GRU 2.07 s, LIN 1.99 s, RSSM 5.07 s per instance | 1.35 / 1.30 / 3.30 ms per decision, so 0.129 / 0.124 / 0.317 s per 96-step episode (plant step included) |
| PI, 16 scenarios | 0.030 s | 0.0019 s per episode |
| Oracle-MPC, 16 scenarios | 0.273 s | 0.017 s per episode |
| Overhead outside phases | 16.75 s (+15 s supervisor reserve) | 31.75 s |

## Conservative assumptions for new parts (not measured)
- **RC_only fit:** 20 s (torch RC + Kalman loop of about 10 ms/update x 2000). Episode cost = LIN x 1.15 = 0.143 s.
- **WM GreyBox fit:** RC fit + GRU fit of the same hidden size. Size factors for h = 32/64/96 are 0.6/1.0/1.6, giving 28.2 / 33.7 / 42.0 s.
- **WM GreyBox episode cost:** taken as the RSSM cost of 0.317 s, which is pessimistic (GRU 0.129 + RC).
- **Episode execution:** unbatched, one episode at a time, as in Option1. Lock-step batching across episodes is allowed and would only reduce cost.

## Full design (CONFIG_v8.1.json)
| Phase | Count | CPU-s |
|---|---|---|
| Overhead + supervisor | Option1 measured | 31.8 |
| Stage-0 checks + data generation | 3 reps x 40 eps x 288 steps | 1.5 |
| Fits | GRU 9, WM 9, LIN 9, RC_only 9 (λ needs no refit) | 624.0 |
| Tuning (structure x λ) | 4 arms x 9 configs x 3 reps x 40 eps = 4320 eps | 770.8 |
| Primary test (HELDOUT) | 4 arms x 3 reps x 320 eps = 3840 eps | 685.1 |
| In-distribution secondary | 4 x 3 x 40 eps | 85.6 |
| PI + Oracle-MPC | 400 + 360 eps | 6.9 |
| Adaptation updates | 36 per arm x 100 updates (largest size assumed) | 152.2 |
| Adaptation evaluation | 4 arms x 3 reps x 4 types x 3 K x 10 eps = 1440 eps | 256.9 |
| Bootstrap (3 matrices of 10000 x 40, saved) + stats | | 10.0 |
| Audit, hashing, serialization | | 15.0 |
| **Total expected** | | **2639.8 s = 44.0 CPU-min** |
| **With x1.25 contingency** | | **3299.7 s = 55.0 CPU-min** |

The un-shrunk design from PLAN_v8.1 would cost far more: 4 arms x 3 sizes x 3 seeds x 4 init states x 40 test seeds x 8 interventions = 46,080 test episodes, about 2.9 h of test alone. It was shrunk **before any data**, in four ways:
1. Each test seed is paired with all 8 held-out entries and **one** initial state (S[i mod 4], 10 seeds per state) instead of 4.
2. Size and structure selection happens on tuning seeds. Test uses only the selected config x 3 replicates, and per-size results are reported on tuning only.
3. LIN and RC_only have no width grid. Their structure grid (order or RC nodes) gives them the same 9 configs as WM and GRU.
4. Adaptation is evaluated on 10 test seeds x the larger magnitude.

## Cap
- **Production hard cap: 3600 CPU-s (60 CPU-min).** This is unchanged, and GPU is not used.
- **Per-phase caps:**

| Phase | Cap (s) |
|---|---|
| Overhead | 60 |
| Data | 30 |
| Fits | 800 |
| Tuning | 1000 |
| Test | 900 |
| In-distribution | 120 |
| PI/Oracle | 30 |
| Adaptation | 520 |
| Bootstrap/stats | 60 |
| Audit | 80 |
| **Sum** | **3600** |

  Exceeding any cap means INVALID: no resume and no rerun.
- **Toy preflight cap: 120 CPU-s.** It runs on non-production seeds 1-4 only. It measures per-update and per-decision cost for every arm and size and projects the full design.
  - If the projection is at or below 3240 s (54 min), launch is allowed.
  - Otherwise the predeclared shrink ladder applies, in order, until the projection is at or below 3240 s. Each applied step is recorded in RELEASE_v8.1.json before launch:
    - (a) adaptation evaluation 10 → 5 seeds;
    - (b) in-distribution secondary dropped;
    - (c) tuning episodes 40 → 30 per config (seeds 7000-7014 only).
  - If it still exceeds 3240 s, there is no launch, and a new plan revision is needed.
