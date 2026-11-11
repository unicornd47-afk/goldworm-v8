"""v8.3 (PLAN_v8.3.md r3: budget and measurement correction of v8.2; science = v8.2 bundle, unchanged), minimal diff on
the frozen v8.2 code (manifest a6544b6f…b643). New: hostenv (C1/C4/A2), probe (C2/C3/K1/K2/A1); sidecar outside code/.

v8.2 thermal world-model experiment (PLAN_v8.2.md), minimal diff on the frozen v8.1 code (manifest be232be6…d4e8;
package name kept as v81 to minimise the diff). Implementation by TMbot · Research (non-author).

Package layout:
  spec        binding plan files, hashes, frozen numbers (CONFIG governs numbers)
  seeds       seed lists, seed guard, initial-state rule, episode ids, RNG derivation
  plant_iv    scorer-side plant + time-windowed interventions + episode generation
  features    observable-only feature construction (model side)
  models_torch, model_lin   learned models (model side; no plant/intervention/scorer imports)
  predictors, planner, closedloop   MPC and closed-loop execution
  stats, budget, ledger, stage0     endpoints/gates, CPU caps, sealing ledger, static checks
  pipeline    full-run orchestration (production design or smoke design)
  preflight   toy preflight + projection
  release     start-rule verification (NOT_LAUNCHED vs launch); v8.3: probe/CONFIG/host binding
  hostenv     v8.3 thread readback, host fingerprint, C4 start rule, sidecar launcher
  probe       v8.3 full-size probe, projections, K2, caps, ladder, A1 decision
"""
