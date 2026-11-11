# v8.3: budget and measurement correction of v8.2 (PREREG r0, for dahill's review)

Author: uni · KI-Forschung. Implementation: TMbot (non-author), as in v8.2. Status: DRAFT, nothing run. No publishing without Dark's OK.
Base: v8.2 frozen unchanged. PLAN_v8.2.md a1e5b1cd…26c3, CONFIG_v8.2.json 7f31dc4e…ef63, ERRATUM_v8.2_E1.md 67b4ecdb…8c47, PASS PLAN_PASS_v8.2.json, code manifest a6544b6f…b643.
Reason: v8.2 ended INVALID. Fits took 803.67 CPU-s against an 800 s cap, while the preflight had projected 336.8 s (factor 2.39). RESULT_AUDIT_v8.2.json 65563…644a confirmed it.

## 1. What stays identical
The science is unchanged: model, arms, structures, the λ grid {0.5, 0.625, 0.75, 0.875, 1.0}, D_max, selection rule, primary S, P1–P8, bootstrap, data and model-init seeds, and all statistics.
Tuning, test and adaptation seeds also stay the same (7300–7319, 9500–9539, 4100–4109). v8.2 never generated a tuning, test or adaptation episode (audit: "Tuning, Test und Selektion sind nie gelaufen"), so no outcome on these seeds has been seen. Condition: before launch, TMbot shows from `run_output/` and the ledger that no tuning, test or adaptation file exists. Otherwise the fresh seeds 7400–7419, 9600–9639 and 4200–4209 apply, and nothing else changes.

## 2. Changes (only these)
C1. Threads pinned [L]. OMP_NUM_THREADS = MKL_NUM_THREADS = OPENBLAS_NUM_THREADS = NUMEXPR_NUM_THREADS = 1 and torch.set_num_threads(1), set before the imports. The values are read back and written to the ledger at start, and a mismatch means NOT_LAUNCHED. Reason: under contention, multi-threaded BLAS inflates process CPU-s, which is dahill's hypothesis, unproven. One thread makes CPU-s a stable cost measure.

C2. Fits cap raised. fits: 800 → 1200 CPU-s, which is 1.5 × the measured 803.67. The global cap stays 3600, the total gate stays 3240, and all other phase caps stay unchanged.

C3. Calibrated fits projection [L]. projected_fits = toy-preflight fits projection × 2.39, the measured v8.2 ratio 803.67 / 336.8. Gate: projected_fits ≤ 0.9 × 1200 = 1080 s, else NOT_LAUNCHED. The total projection uses the calibrated fits value. Other phases keep their v8.2 projection rule: their toy projections were never tested, because those phases never ran. The calibration factor is reported as [H] for the other phases.

C4. Exclusivity and start rule (as in exp001). Start only if the 1-minute load is < 1.0, MemAvailable is ≥ 4 GB, and no exp001 arm or other training process is running on the box. All of this is checked and logged right before launch. Alternative: the laptop CPU, as long as nothing else large is running there. The hardware is then declared in the RELEASE, and the toy preflight is re-measured there.

C5. Load sampling that keeps running. An independent sidecar runs as its own process (nohup, setsid) and writes load, MemAvailable and the wall clock every 30 s to `LOAD_SAMPLES_production_v8.3.txt`. Wall gaps > 90 s are listed in RESULTS (box paused). A gap invalidates nothing, because caps are CPU-s.

C6. Ledger events per phase [L]. Every phase writes `phase_start` and `phase_end` with process CPU-s, wall time and the thread readout to LEDGER.jsonl. RESULTS reports each phase's measured cost against its projection. A phase cap is checked against these events.

C7. Write-once outputs. `normalization.json` and every sealed file are opened with exclusive create, so a second write raises. This answers audit finding 4.

## 3. Unchanged rules
An exceeded cap means INVALID, with no rerun under v8.3. Exactly one run. Any code change means a new manifest and a new review. Nothing gets published.

## 4. Diff budget
Implementation as a minimal diff on manifest a6544b6f: thread pinning, cap and calibration in CONFIG and preflight, ledger events, write-once, and the sidecar as a separate script outside the frozen code. The diff is hashed and reviewed independently, as in v8.2.
