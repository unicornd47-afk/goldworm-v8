# IMPL_ERRATUM_v8.2_W1 — preflight warm-up fix (implementation erratum)

Author: TMbot · Research (non-author implementer), 2026-11-10 Europe/Berlin. Approved as "option 1" by dahill
(reviewer) and uni (plan author) under the fixed conditions below, as relayed by the coordinator.
Applies to the frozen v8.2 code (previous manifest da529e8248d5c1f365b223c9da2e6853bf57ce2049d6dec4061553a843a0308f)
under bundle HASHES_v8.2_E1.sha256 = 304bc28de338c92ee72817e7ed4ad498c367165e7f3ca47eed38e1732c636ff3 and
PLAN_PASS 8ad13f896e2677498048f864ee3abed6e6534c4dcb75c889b59643de34bd49fe. PLAN, CONFIG, FEASIBILITY and ERRATUM
E1 are unchanged.

## W1.1 What changes (preflight measurement only)
- `v81/preflight.py::measure`: before any timed update or episode, one untimed warm-up for EVERY arm type and EVERY
  structure (WM h32/h64/h96, GRU h32/h64/h96, LIN order 3/5/8, RC_only 1/2/3-state): build + 2 throwaway Adam
  updates (LIN: one throwaway fit) on the same toy windows, with a separate model instance (init seed 0, so the
  measured models, built with seed 3101, are unaffected). Its CPU is logged (`warmup_all_structures_cpu_s`) and
  counts toward the 120 s preflight cap.
- `run_preflight.py`: logs `os.getloadavg()` before and after the measurement.
- Plausibility gate and the W1 ladder restriction (W1.3) in `preflight.decide` and `release.check_start`.
- `spec.BUNDLE` / `release.check_start` bind this file and the failed first attempt (W1.4).
- NOT changed: production path (pipeline, models, planner, closed loop, selection, statistics), gates, caps,
  seeds, λ grid, comfort bound, ladder steps' content.

## W1.2 Evidence
- v8.2 attempt 1 (PREFLIGHT_PROJECTION_attempt1_FAILED.json, sha256
  f2a0e6b2a608d71bd0f1d554a2aca481ac97621f2be08985a565bdf73c4b17f7, 2026-11-10 13:04:58 Europe/Berlin):
  WM h=32 0.0805 s/update vs WM h=64 0.0133 and WM h=96 0.0111 s/update. WM h=32 alone contributes
  ≈ 3 × 2000 × 0.0805 ≈ 483 s to the fits projection (815.4 s > 720 s gate). Machine load: load average
  1.42 / 4.04 / 5.58 (1/5/15 min) read with `uptime` at 13:05:17, right after the attempt (not logged by the
  preflight itself).
- v8.1 preflight (d65447ba…d527): WM_32 0.0397 s/update, an outlier vs WM_64 0.0081 / WM_96 0.0090; v8.1 fits
  projected 498.1 s, actually used 305.4 s in the production run (FINAL_STATUS 7330b5a5…1c30).
- Cause: the existing warm-up ran only a GRU forward/backward, so WM-specific lazy initialisation (closed-form
  RC/Kalman ops) was charged to the first WM structure measured (h=32).

## W1.3 Binding conditions
1. Exactly ONE fresh preflight after the re-freeze, and it is binding. No repeat under any circumstance (load,
   crash, or result). If it crashes: report and stop (NOT_LAUNCHED).
2. Fits projection > 0.9 × cap (720 s) → NOT_LAUNCHED, no retry.
3. Adaptation projection > 0.9 × cap (468 s) → only shrink-ladder step (a) (adaptation eval 10 → 5 seeds,
   9500-9504) may be applied; if any gate still fails → NOT_LAUNCHED. Ladder steps (b) and (c) are not used.
4. Plausibility gate: WM_32 per-update cost must be ≤ WM_64 per-update cost in the fresh preflight, else
   NOT_LAUNCHED. Implemented in `preflight.decide` (recorded as `plausibility`) and re-checked from the unit
   costs in `release.check_start`.
5. The attempt-1 file is kept read-only and listed in the RELEASE as the first failed attempt.
