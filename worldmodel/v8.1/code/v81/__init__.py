"""v8.1 thermal world-model experiment (PLAN_v8.1_r3.md). Implementation by TMbot · Research (non-author, §11).

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
  release     start-rule verification (NOT_LAUNCHED vs launch)
"""
