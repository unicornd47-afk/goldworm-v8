# v8.3: budget and measurement correction of v8.2 (PREREG r2, for dahill's review; supersedes r0 3107d350…4f2f and r1 69c8f1cd…9311; answers PLAN_REVIEW_v8.3_r0 d8150e96 B1–B3)

Author: uni · KI-Forschung. Implementation: TMbot (non-author), as in v8.2. Status: DRAFT, nothing run. No publishing without Dark's OK.
Base: v8.2 frozen unchanged. PLAN_v8.2.md a1e5b1cd…26c3, CONFIG_v8.2.json 7f31dc4e…ef63, ERRATUM_v8.2_E1.md 67b4ecdb…8c47, PASS PLAN_PASS_v8.2.json, code manifest a6544b6f…b643.
Reason: v8.2 ended INVALID. Fits took 803.67 CPU-s against an 800 s cap, while the preflight had projected 336.8 s (factor 2.39). RESULT_AUDIT_v8.2.json 65563…644a confirmed it.

## 1. What stays identical
The science is unchanged: model, arms, structures, the λ grid {0.5, 0.625, 0.75, 0.875, 1.0}, D_max, selection rule, primary S, P1–P8, bootstrap, data and model-init seeds, and all statistics.
Tuning, test and adaptation seeds also stay the same (7300–7319, 9500–9539, 4100–4109). v8.2 never generated a tuning, test or adaptation episode (audit: "Tuning, Test und Selektion sind nie gelaufen"), so no outcome on these seeds has been seen. Condition: before launch, TMbot shows from `run_output/` and the ledger that no tuning, test or adaptation file exists. Otherwise the fresh seeds 7400–7419, 9600–9639 and 4200–4209 apply, and nothing else changes.

## 2. Changes (only these)
C1. Thread pinning stays as a safeguard. v8.2 was already single-threaded (RESULTS_v8.2 83c45978…b8e5), so the pinning is not a cause hypothesis. v8.3 additionally reads the values back into the ledger at start; a mismatch means NOT_LAUNCHED. The factor 2.39 is unexplained (host incident vs. systematic is open) and is used nowhere in the plan.

C2. Full-scale probe on the target machine before the freeze [L] (B3). It runs under the C4 start rule with 1 thread, with the frozen production code, and only on train/val data seeds and train-entry episodes. It never uses tuning, test or adaptation seeds.
 (a) Fits: WM h96 (the most expensive structure in the v8.2 preflight) as one complete fit at full size. The other 11 arm × structure fits run for the first 10% of their updates.
 (b) Linearity check: the cost of the first 10% of WM h96 × 10 vs. its full cost. A ratio outside [0.87, 1.15] means no freeze, diagnosis first.
 (c) Episodes: for every arm × structure, 2 full-size train-entry episodes (96 decisions, 64 candidates, λ=1.0), giving the measured cost per episode.
 Probe cap: 450 CPU-s, hard abort. Output: PROBE_v8.3.json with per-unit costs, per-update CPU-s and wall time, and the sidecar values (sha in the RELEASE).

C3. Projection and caps from the probe, rule fixed beforehand [L] (B2, B3).
 Projection per phase from measured per-unit costs: fits = the WM h96 full cost + 10 × the 10% costs of the other 11; tuning, test, in_distribution and adaptation = number of episodes per the v8.2 CONFIG × measured episode cost (test, in_distribution and adaptation at the per-arm maximum over structures, as in v8.2). Fixed phases (overhead 60, data 30, pi_oracle 30, bootstrap_stats 60, audit 80 = 260) as in v8.2.
 Phase cap = ceil(1.25 × projection) for the five measured phases. There is ONE budget: the global cap = the sum of all phase caps, which must be ≤ 3600. That makes a global abort with every phase under its cap impossible. Gate: sum ≤ 3600, i.e. measured projections ≤ (3600 − 260) / 1.25 = 2672 CPU-s. If it does not fit, the v8.2 ladder applies (adaptation, in-distribution, tuning, in that order; λ and structure grids are never shrunk). If it still does not fit: NOT_LAUNCHED, and the numbers go to dahill and mixxy. The values go into CONFIG_v8.3 only after the probe. The plan stays unchanged.

C4. Exclusivity and start rule (as in exp001). Start only if the 1-minute load is < 1.0, MemAvailable is ≥ 4 GB, and no exp001 arm or other training process is running on the box. All of this is checked and logged right before launch. Alternative: the laptop CPU, as long as nothing else large is running there. The hardware is then declared in the RELEASE, and the toy preflight is re-measured there.

C5. Load sampling that keeps running. An independent sidecar with a heartbeat runs as its own process (nohup, setsid) and writes load, MemAvailable, steal time (/proc/stat), CPU MHz (/proc/cpuinfo), the process CPU-s of the run (/proc/<pid>/stat) and the wall clock every 30 s to `LOAD_SAMPLES_production_v8.3.txt`. Wall gaps > 90 s are listed in RESULTS (box paused). A gap invalidates nothing, because caps are CPU-s.

C6. Ledger events per phase [L]. Every phase writes, and inside fits additionally per fit (arm, structure, updates, CPU-s) and every 100 updates, `phase_start` and `phase_end` with process CPU-s, wall time and the thread readout to LEDGER.jsonl. RESULTS reports each phase's measured cost against its projection. A phase cap is checked against these events.

C6b. Cap check [L]. Inside fits every 10 updates, otherwise after every episode or every bootstrap block of 1000. INVALID as soon as a check measures > phase cap. A tolerated overshoot is not defined; the probe's measured per-check jump is reported.

C7. Write-once outputs. `normalization.json` and every sealed file are opened with exclusive create. Precaution only: according to TMbot the 15:25 mtime stems from a file system restore, not from a rewrite. The time of the run's end comes from the ledger event, not from mtimes.

## 3. Unchanged rules
An exceeded cap means INVALID, with no rerun under v8.3. Exactly one run. Any code change means a new manifest and a new review. Nothing gets published.

## 4. Diff budget
Implementation as a minimal diff on manifest a6544b6f: thread readback, probe script (same code, update limit, train episodes), caps from the probe in CONFIG, cap checks per C6b, ledger events, write-once, and the sidecar as a separate script outside the frozen code. The diff is hashed and reviewed independently, as in v8.2.
