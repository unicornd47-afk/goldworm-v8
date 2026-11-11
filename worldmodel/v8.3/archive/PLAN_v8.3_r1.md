# v8.3: budget and measurement correction of v8.2 (PREREG r1, for dahill's review; supersedes r0 3107d350…4f2f)

Author: uni · KI-Forschung. Implementation: TMbot (non-author), as in v8.2. Status: DRAFT, nothing run. No publishing without Dark's OK.
Base: v8.2 frozen unchanged. PLAN_v8.2.md a1e5b1cd…26c3, CONFIG_v8.2.json 7f31dc4e…ef63, ERRATUM_v8.2_E1.md 67b4ecdb…8c47, PASS PLAN_PASS_v8.2.json, code manifest a6544b6f…b643.
Reason: v8.2 ended INVALID. Fits took 803.67 CPU-s against an 800 s cap, while the preflight had projected 336.8 s (factor 2.39). RESULT_AUDIT_v8.2.json 65563…644a confirmed it.

## 1. What stays identical
The science is unchanged: model, arms, structures, the λ grid {0.5, 0.625, 0.75, 0.875, 1.0}, D_max, selection rule, primary S, P1–P8, bootstrap, data and model-init seeds, and all statistics.
Tuning, test and adaptation seeds also stay the same (7300–7319, 9500–9539, 4100–4109). v8.2 never generated a tuning, test or adaptation episode (audit: "Tuning, Test und Selektion sind nie gelaufen"), so no outcome on these seeds has been seen. Condition: before launch, TMbot shows from `run_output/` and the ledger that no tuning, test or adaptation file exists. Otherwise the fresh seeds 7400–7419, 9600–9639 and 4200–4209 apply, and nothing else changes.

## 2. Changes (only these)
C1. Threads: a check, not a change. v8.2 was already single-threaded (TMbot, RESULTS_v8.2 83c45978…b8e5). v8.3 keeps that and additionally reads the values back into the ledger at start. A mismatch means NOT_LAUNCHED. This rules out BLAS threads as a cause. The factor 2.39 remains unexplained (likely a box pause or host overcommit, not provable).

C2. Fits timing probe before the freeze [L], replacing the toy calibration from r0. Conditions are those of C4 (start rule, 1 thread). The same fit code as in production runs every arm × structure (12 fits) for the first 10% of its updates, on train/val data seeds only. No tuning, test or adaptation seeds are used. Probe cap: 240 CPU-s, hard abort. Per-fit CPU-s and wall time are logged at least every 10 updates. Output: FITS_PROBE_v8.3.json (sha in the RELEASE). The probe is run 2× in a row. The spread between the two runs (max/min of total CPU-s) is reported and must be ≤ 1.15, else no freeze and diagnosis first.

C3. Budget from the probe [L]. P_fits = 10 × the larger of the two probe sums. Fits cap = ceil(1.3 × P_fits), with a minimum of 800. The total budget stays 3600 CPU-s with a total gate of 3240. Check: P_fits + the v8.2 projections of the other phases ≤ 3240 − (fits cap − P_fits). If it does not fit, the v8.2 ladder applies first (adaptation, in-distribution, tuning, in that order). If it still does not fit: no v8.3 launch, and back to dahill/mixxy with the numbers. The fits cap is written into CONFIG_v8.3 only after the probe; the plan text is frozen before the probe, the number after it (rule fixed, value derived).

C4. Exclusivity and start rule (as in exp001). Start only if the 1-minute load is < 1.0, MemAvailable is ≥ 4 GB, and no exp001 arm or other training process is running on the box. All of this is checked and logged right before launch. Alternative: the laptop CPU, as long as nothing else large is running there. The hardware is then declared in the RELEASE, and the toy preflight is re-measured there.

C5. Load sampling that keeps running. An independent sidecar with a heartbeat runs as its own process (nohup, setsid) and writes load, MemAvailable and the wall clock every 30 s to `LOAD_SAMPLES_production_v8.3.txt`. Wall gaps > 90 s are listed in RESULTS (box paused). A gap invalidates nothing, because caps are CPU-s.

C6. Ledger events per phase [L]. Every phase writes, and inside fits additionally per fit (arm, structure, updates, CPU-s) and every 100 updates, `phase_start` and `phase_end` with process CPU-s, wall time and the thread readout to LEDGER.jsonl. RESULTS reports each phase's measured cost against its projection. A phase cap is checked against these events.

C7. Write-once outputs. `normalization.json` and every sealed file are opened with exclusive create. Precaution only: according to TMbot the 15:25 mtime stems from a file system restore, not from a rewrite. In addition, the cap check in fits runs at least every 10 updates, so overshoot is ≤ 1 check interval (v8.2: ≥ 3.6 s jump). The time of the run's end comes from the ledger event, not from mtimes.

## 3. Unchanged rules
An exceeded cap means INVALID, with no rerun under v8.3. Exactly one run. Any code change means a new manifest and a new review. Nothing gets published.

## 4. Diff budget
Implementation as a minimal diff on manifest a6544b6f: thread readback, probe script (same fit code, update limit), cap from the probe in CONFIG, ledger events, write-once, and the sidecar as a separate script outside the frozen code. The diff is hashed and reviewed independently, as in v8.2.
