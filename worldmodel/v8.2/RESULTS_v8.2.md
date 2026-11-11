# RESULTS v8.2 — thermal world-model experiment

Author: TMbot · Research (non-author implementer). Written 2026-11-11 Europe/Berlin, read-only analysis of the
finished run. Nothing was rerun, no code changed, run_output / v8.1 / v7 / audit dirs untouched, nothing published.

## 1. Outcome: INVALID (pre-registered rule, no rerun)

`run_output/FINAL_STATUS.json`: `outcome = INVALID`, `budget.cap_exceeded = "phase fits: 803.61 > cap 800.0"`,
`budget.used.fits = 803.67`, `budget.total = 805.04` CPU-s. Under PLAN v8.2 / budget.py a phase-cap breach raises
CapExceeded → INVALID, with no resume and no rerun (dahill N2: "acceptable and must be reported as such"). Precedence
NOT_LAUNCHED > INVALID > ADVANCE/NO_ADVANCE; the run was launched, so the outcome is INVALID.

**This run makes no claim about the hypothesis, in either direction.** No tuning, selection, test or statistic exists.

- Run: one production run, pid 895218, started 2026-11-10 13:19:39 Europe/Berlin, ended 14:04:47 Europe/Berlin
  (LEDGER `final` event time_unix 1794315887.32). Entrypoint `code/run_production.py`, CPU only.
- `production_run.log` (26 bytes) contains only `{"outcome": "INVALID"}`; no traceback.
- dahill's RESULT_AUDIT_v8.2.json (09:40 Europe/Berlin, sha 655633a9…644a): verdict RESULT_CONFIRMED_INVALID.

## 2. Budget (CPU-s, process CPU via time.process_time)

| phase | cap | gate 0.9×cap | preflight att. 1 (FAILED) | preflight att. 2 (binding) | v8.1 actual | v8.2 actual |
|---|---|---|---|---|---|---|
| data | 30 | 27 | 0.35 | 0.31 | 0.18 | 0.22 |
| fits | 800 | 720 | 815.4 | 336.8 | 305.4 | **803.67 (cap breached)** |
| tuning | 1100 | 990 | 946.9 | 848.7 | 465.9 | not run |
| test | 800 | 720 | 545.6 | 512.2 | 436.7 | not run |
| in_distribution | 120 | 108 | 68.2 | 64.0 | 57.9 | not run |
| adaptation | 520 | 468 | 537.4 | 268.3 | 277.7 | not run |
| pi_oracle | 30 | 27 | 10.4 | 12.3 | 10.4 | not run |
| bootstrap_stats | 60 | 54 | 4.4 | 4.5 | 3.5 | not run |
| audit | 80 | 72 | 0.03 | 0.03 | 0.03 | not run |
| overhead | 60 | 54 | 16.0 | 16.0 | 1.0 | 0 (not finalized) |
| **total** | 3240 global | — | 2944.7 | 2063.2 | 1558.6 | 805.04 |

The fits figure is 2.39× the binding projection and 2.63× the v8.1 actual for the same fits workload (same arms,
structures, n_updates, init seeds, training windows). Message 803.61 vs field 803.67: the message is computed in
`Budget.check()` at the failing per-update check; the field adds the CPU spent unwinding the exception through
`Budget.phase()` (0.06 s). Explained by code, not a data error.

## 3. What did not run

fits did not finish (`fit_info.json` was never written; it is saved only after all fits). Not run: tuning,
sealed selection / COMPARATOR_SELECTION, test, in-distribution, adaptation, PI/oracle, bootstrap/MSE CI, audit.
`selection`, `selection_fallback_flags`, `primary` are null; no S, ΔD, MSE CI, or fallback flag exists. Written:
`normalization.json` (data phase), `FINAL_STATUS.json`, `LEDGER.jsonl` (only the `final` event, because the
first ledger events in the pipeline come after fits), and the log above.

## 4. Integrity (verified 2026-11-11 09:38 Europe/Berlin)

| item | sha256 | check |
|---|---|---|
| RELEASE_v8.2.json (FINAL, read-only) | 00b4766dcd7af84ca10bddd9724d3d21daf7c83e49fc223df7f9f783347c0b9e | all 8 binding_sha256 match disk |
| RELEASE_v8.2_DRAFT.json (mixxy-signed) | 262021511fd5ba49093344de2ddc2bd705ec0cef2333b6bfe9a32d300231b42d | = derived_from_draft_sha256 |
| code/CODE_HASHES_v8.2.sha256 | a6544b6fdafb11bf853bd1b57ce8d3f8d508f06dbafebcafb2a7dbede2a7b643 | 32/32 OK |
| HASHES_v8.2_E1.sha256 (bundle) | 304bc28de338c92ee72817e7ed4ad498c367165e7f3ca47eed38e1732c636ff3 | 6/6 OK |
| DIFF_v8.1_to_v8.2.patch | 223a442f5eb2dcd579d482e75cafa422a31d619047599356dcdfb70052139473 | match |
| IMPL_ERRATUM_v8.2_W1.md | 08f62233dadb4d009dfe5d7ce3bd3529337886211c87b48c1db5743c1668bdf8 | match |
| PREFLIGHT_PROJECTION.json (attempt 2) | 7e1643629afe9f4d0dc5a1401d62e2e5b5b6bfda48069810c3a2b4b61a670a29 | match |
| PREFLIGHT_PROJECTION_attempt1_FAILED.json | f2a0e6b2a608d71bd0f1d554a2aca481ac97621f2be08985a565bdf73c4b17f7 | match, read-only |
| PLAN_PASS_v8.2.json (dahill) | 8ad13f896e2677498048f864ee3abed6e6534c4dcb75c889b59643de34bd49fe | match |
| CODE_PASS_v8.2.json (dahill) | 2b10b0f56c671c97ed17bdf6f0ca0b56a810031c16bfde241a338732de77f7b4 | match |
| run_output/FINAL_STATUS.json | 42c77e9ee29ed0894c82bb113b800adb3fc06fb409540a0767c6655933ec156e | — |
| run_output/LEDGER.jsonl | e67434423fdc7e17a542fc9957d92012134b5702cd571be401c03a762c0ad542 | — |
| run_output/normalization.json | a8f5b344bf93b9856e701379f71d5a669cf4acc73c3f80e36cbe4b621c332ebf | — |
| production_run.log | 9c35fae907617314e575fc17388d9f1b1ea4d6f4aecde7de15071fea51a940fd | — |
| LOAD_SAMPLES_production_v8.2.txt | 2fd95140b901d2767884f5043611bb6f64c4a33fe8026739799c1a68f67ce0f0 | — |

v8.1 code manifest still verifies (`sha256sum -c --quiet` OK). All files in v8.2 and in /workspace/audit_dahill_v8.2
(except RESULT_AUDIT, written 2026-11-11) have mtime 2026-11-10 15:25:46.4–.9 Europe/Berlin, including read-only bundle files nobody
wrote after 13:19. That is a box filesystem restore/sync, so **mtimes after 15:25 do not show write times**. The
hashes above are the integrity evidence. (Correction to RESULT_AUDIT F2/F4: the run ended 14:04:47 per the LEDGER, not 15:25:46,
and normalization.json's mtime comes from the restore, not from a rewrite.)

## 5. Preflight attempts vs actual

- Attempt 1 (13:04:58 Europe/Berlin, manifest da529e82): FAILED. Fits 815.4 > 720 and adaptation 537.4 > 468. Cause: WM lazy init charged
  to WM_32 (0.0805 s/update vs WM_64 0.0133). Load 1.42 / 4.04 / 5.58 (uptime 13:05:17). Kept read-only.
- Attempt 2 (13:12:20 Europe/Berlin, manifest a6544b6f, under W1, the single binding run): PASS. Fits 336.8 / 720, plausibility WM_32
  0.008361 ≤ WM_64 0.008923 s/update. Load before 0.84 / 1.63 / 3.84, after 0.93 / 1.63 / 3.83. Preflight CPU 6.16 / 120 s.
- Actual production fits: ≥ 803.67 CPU-s, fits not finished. Attempt 2's per-update unit costs agree with v8.1's
  actual fits (305 s), so the preflight is consistent with v8.1. The production fits phase differs from both.

## 6. Machine load during production

`LOAD_SAMPLES_production_v8.2.txt` (/proc/loadavg format; written by the run supervisor, not by the frozen code):

```
13:20:49 0.98 1.22 2.68 3/1271
13:21:49 1.45 1.29 2.61 2/1269
13:22:49 1.21 1.25 2.51 2/1270
13:23:49 1.08 1.20 2.42 3/1270
13:29:27 1.00 1.08 1.99 32/1267
14:04:40 9.24 3.15 2.55 32/1168
```

The sampling interval was 60 s. Gaps: 13:23:49 → 13:29:27 (5.6 min) and 13:29:27 → 14:04:40 (35.2 min). The machine has 8 vCPUs, a KVM guest.

## 7. Diagnosis of the fits overrun (read-only; evidence first, then clearly marked speculation)

### Evidence
E1. **The clock counts all threads.** `budget.Budget` uses `time.process_time` (CLOCK_PROCESS_CPUTIME_ID): user+sys of
    every thread of the process. CPU-s from several threads would be summed.
E2. **Thread settings were single-threaded and identical to the preflight.** `run_production.py` sets
    OMP/MKL/OPENBLAS/… `_NUM_THREADS=1` before `import torch` (lines 5–12) and calls `torch.set_num_threads(1)` (line 14). The
    same venv with the same settings reports torch 2.15.0+cpu, intra-op 1, `omp_get_max_threads()` 1, MKL (oneAPI 2024.2). The inter-op
    pool size is 8, but eager-mode training does not use it. `run_preflight.py` uses the same settings. So "spinning BLAS/OpenMP threads"
    (RESULT_AUDIT F2) is not supported by the configuration. No setting is a known source of multi-thread CPU.
E3. **Wall time vs CPU time.** 13:19:39 → 14:04:47 is 2708 s of wall time for 805 CPU-s, so on average ≤ 30 % of one core.
    The 1-min load ≈ 1.0–1.45 from 13:20 to 13:29 says the run was the only busy task then.
E4. **Sampler gaps.** The 60-s sampler wrote nothing for 5.6 min and then for 35.2 min. Plain contention by 32 runnable tasks on 8 vCPUs
    does not normally starve a sleep-60 sampler for 35 min. At 13:29:27 the 1/5/15-min loads (1.00/1.08/1.99) are what
    ~5.6 min of steady load ≈1 after 13:23:49 would give (15-min: 1 + 1.42·e^(−5.6/15) ≈ 1.98), but 32 tasks are
    runnable at that instant. At 14:04:40 the 1-min load jumped to 9.24 (5-min 3.15).
E5. **There is one ≥3.61 CPU-s chunk without a budget check.** In fits, `Budget.check()` runs after every Adam update
    (≈0.007–0.011 CPU-s per update in attempt 2), so the cap should fire at ≈800.01. It fired at 803.61. So the last
    passing check (≤800.0) and the failing one were ≥3.61 CPU-s apart. That interval is either one update or the work between
    two models (val eval, build, a LIN fit, or `window_arrays` at a replicate boundary), each expected ≪ 1 s (not separately measured in the preflight).
E6. **Precedent from v8.1.** v8.1's LEDGER puts 2 h 25 min of wall time between `sealed PRIMARY_RESULT` (09:56:53) and
    `boot_matrix_saved` (12:22:17) Europe/Berlin. RESULTS_v8.1.md attributes this to a box pause (~09:57–~11:56, inferred from `ps` elapsed time, not
    measured). In that window in_distribution + adaptation used only 57.9 + 277.7 CPU-s (projection-consistent), so that pause did
    **not** inflate process CPU-s.
E7. **History is gone.** No journal exists ("No journal files were found"). The box rebooted at 2026-11-11 00:15 Europe/Berlin, so
    /proc, cgroup cpu.stat, PSI and ps histories for 13:19–14:05 are lost. The `/agent` cgroup has no cpu.max (no CFS throttling now).
    The filesystem was restored/synced at about 15:22–15:27 (root dir mtime 15:22:50, box-doctor 15:27:10, all v8.2 mtimes 15:25:46).
    No file in /tmp or /workspace has a genuine write time in 13:24–14:06 (the only hits are zip-internal 13:54:16 stamps from an
    extraction at 16:10). So I can't show from files that another workload ran in the gap.
E8. **Kernel accounting.** The kernel is 6.12 (KVM, kvm-clock, `tsc=unstable`), with CONFIG_TICK_CPU_ACCOUNTING=y and CONFIG_PARAVIRT_TIME_ACCOUNTING=y.
    The steal counter is non-zero in the current boot (521 s in 9.3 h), so the host does overcommit at times.
    dmesg in the current boot shows 133 `virtio_balloon … Out of puff` messages (host memory reclaim). These are from after the run and
    not evidence about 13:19–14:05. Swap is 0.

### Inference (follows from the evidence above, but not proven)
I1. If the sampler gaps (E4) were box pauses, the run had only ~10–11 min of non-paused wall time (13:19:39 → ~13:30, plus seconds after
    14:04:40). A single thread (E2) cannot accrue 805 CPU-s in ~650 s. In that case the CPU-s figure includes time
    when the process was not computing: CPU time charged around a pause/resume or a clock jump. The single ≥3.61 s chunk (E5)
    fits a lump charge better than a uniform slowdown. Against this: v8.1's pause did not inflate CPU-s (E6).
I2. If there was no pause and the box was only heavily contended, per-op CPU cost would have to rise ~2.4× (shared physical cores / SMT,
    cache and memory-bandwidth contention, or steal not fully excluded). Contention inflates CPU-time per op much less than wall time,
    and attempt 1 at load ~5.6 was only ~15–20 % above attempt 2 for non-WM units, so 2.4× is large for contention alone.
    This also doesn't explain the 35-min sampler silence.

### Speculation (no direct evidence; labelled as such)
S1. The most likely explanation in my view: a host-side pause/suspend or heavy host overcommit episode between ~13:24 and ~14:04, during which
    guest CPU accounting charged non-compute time to the running training process. The 32-runnable burst and the load-1 spike to 9.24 at
    14:04:40 look like a resume signature. Unverifiable now (E7).
S2. Torch intra-op threads under load inflating CPU-s is unlikely given E2 (intra-op = 1, OMP = 1).

### What can't be determined
- How far fits progressed (which replicate/arm/structure hit the cap), because fit_info.json was never written and the LEDGER has no
  per-phase events (RESULT_AUDIT F3).
- Any process list, steal or PSI for 13:19–14:05 (reboot, no journal).

## 8. Plan items reported as required

- **Grid override E1.1 (dahill N4).** CONFIG_v8.2.json (sha 7f31dc4e…) still lists λ ∈ {0.5, 1, 2, 4, 8}. The effective grid is
  {0.5, 0.625, 0.75, 0.875, 1.0} from ERRATUM_v8.2_E1.md §E1.1 (sha 67b4ecdb…8c47), recorded in FINAL_STATUS.grid_override,
  PREFLIGHT_PROJECTION and RELEASE. The grid was never exercised because tuning did not run.
- **E1.5 fit reproduction vs v8.1 fit_info (dahill N3).** Not available. v8.2 `fit_info.json` was not written (fits aborted
  before the save), so no comparison with v8.1/run_output/fit_info.json is possible.
- **Machine load logging (N3).** Section 6. The log is incomplete (two gaps, 6 samples).
- **N2.** A cap was hit (fits, not the tuning phase N2 flagged at 86 %). The outcome is INVALID, no rerun, as N2 anticipated.
- **N1** (not asked for, for completeness): the preflight manifest (a6544b6f) equals the release manifest. dahill checked this manually.

## 9. Process notes and errata

- **CODE_PASS time_berlin erratum.** CODE_PASS_v8.2.json states `"time_berlin": "2026-11-10T13:25+0100"`. As relayed by the coordinator,
  the file was written at ~13:16 Europe/Berlin, before mixxy's signature (13:19) and the start (13:19:39). 13:25 is a wrong
  stamp, not a late pass. File mtime can't confirm this now (restore, §4). CODE_PASS also names `release_draft_sha256` 81fa75dd…,
  an earlier draft, not the signed draft 26202151. The binding is via manifest/diff/preflight/W1/bundle/PLAN_PASS hashes, which all match.
- **Overwrite / re-sign incident.** mixxy's first signature on RELEASE_v8.2_DRAFT.json (13:18, draft 0c982105) was overwritten by a
  concurrent write from me (the implementer) while I updated the draft. I reported it. mixxy re-signed on draft 4d36315b, giving signed
  draft 26202151…b42d (backup checked identical). RELEASE_v8.2.json = that draft with status FINAL plus derived_from_draft_sha256, nothing
  else changed. The signature text records the incident. check_start reported zero reasons, and the explicit signature check (signer mixxy,
  code_pass 2b10b0f5) passed before launch.
- **Lessons for any v8.3 (not a plan change, for the plan author/reviewer).** Per-phase/periodic ledger events with CPU and wall time;
  a load sampler that records wall vs monotonic time (to detect pauses); an exclusive or pause-free box or a cap robust to accounting jumps;
  log thread count and /proc/self/stat utime/stime at cap breach; re-measure the fits projection under realistic load.
