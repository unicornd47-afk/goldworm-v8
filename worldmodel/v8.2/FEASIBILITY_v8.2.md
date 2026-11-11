# FEASIBILITY v8.2: CPU budget from MEASURED v8.1 costs

This file contains arithmetic only. No data was generated, nothing was fitted and nothing was run.

**Sources:**
- v8.1/run_output/FINAL_STATUS.json (sha256 7330b5a5761ae245383d17edbc4ecc8a8f5dc7d34f73735d76d899a64bef1c30), `budget.used`, the single production run: 1558.6 CPU-s in total.
- v8.1/preflight/PREFLIGHT_PROJECTION.json (d65447ba0000e3fc2c0933cb83c64f6746d399db756dd533cca99899db4cd527), per-structure episode costs.

## What changes the cost vs v8.1
- **Tuning:** 15 configs per arm instead of 9 (λ grid {0.5, 1, 2, 4, 8} x 3 structures). λ does not change per-episode cost because the planner always scores the same 64 candidates. Tuning scales by 15/9.
- **Selected structure:** it may differ from v8.1. Worst case is the per-arm maximum episode cost from the preflight, (0.1752 + 0.1544 + 0.1272 + 0.0249) / (0.1752 + 0.1288 + 0.1229 + 0.0249) = 1.0662, applied to test, in-distribution and adaptation.
- **Fits:** identical (same training seeds, same code path), so the v8.1 measurement is used.
- **New MSE paired bootstrap** (10000 x 40, 6 contrasts): +10 s reserved in bootstrap_stats.
- **Overhead:** the measured 0.98 s plus a 15 s supervisor reserve. Audit: +5 s reserve.

## Projection
| Phase | v8.1 measured (s) | v8.2 projected (s) | Cap (s) | 0.9 x cap | Use of gate |
|---|---|---|---|---|---|
| Overhead | 1.0 | 16.0 | 60 | 54 | 30 % |
| Data | 0.2 | 0.2 | 30 | 27 | 1 % |
| Fits | 305.4 | 305.4 | 800 | 720 | 42 % |
| Tuning | 465.9 | 776.4 | 1100 | 990 | 78 % |
| Test | 436.7 | 465.6 | 800 | 720 | 65 % |
| In-distribution | 57.9 | 61.8 | 120 | 108 | 57 % |
| PI/Oracle | 10.4 | 10.4 | 30 | 27 | 39 % |
| Adaptation | 277.7 | 296.0 | 520 | 468 | 63 % |
| Bootstrap/stats | 3.5 | 13.5 | 60 | 54 | 25 % |
| Audit | 0.0 | 5.0 | 80 | 72 | 7 % |
| **Total** | **1558.6** | **1950.3 (32.5 min)** | **3600** | **3240** | **60 %** |

- **With x1.25 contingency:** 2437.9 s (40.6 min), below the 60 CPU-min cap.
- **Caps:** reallocated within the unchanged 3600 s. Tuning goes 1000 → 1100 and test 900 → 800; all other caps are as in v8.1 r3.
- **v8.1 preflight check:** its projections for v8.1 (test 462.5 s, fits 498.1 s) stay under the new gates of 720 s each.

## Cap and preflight
- **Production hard cap:** 3600 CPU-s (60 min), on CPU, with the per-phase caps above. Exceeding any cap means INVALID: no resume and no rerun.
- **Toy preflight:** capped at 120 s, on seeds 1-4 only. Cost is measured per arm x structure (12 combinations, GRU h=96 included), at λ = 1.0.
  - **Projection:** fits use each structure's own cost. Tuning uses each structure's own cost x 5 λ. Test, in-distribution and adaptation use the per-arm maximum.
  - **Gate:** every phase projection must be at or below 0.9 x its phase cap, and the total at or below 3240 s.
- **Shrink ladder:** it applies only to the adaptation, in-distribution and tuning phases, in this order:
  - (a) adaptation evaluation 10 → 5 seeds;
  - (b) in-distribution secondary dropped;
  - (c) tuning 40 → 30 episodes per config (seeds 7300-7314).
- **Never shrunk:** the ladder never touches the λ grid, the structure grid, fits, the primary test, overhead, data, PI/Oracle, bootstrap/stats or audit. Any of these projecting above 0.9 x its cap means no launch.
- **Recording:** all measured costs, projections and applied ladder steps go into RELEASE_v8.2.json.
