# IMPL_NOTES v8.3 — code for PLAN_v8.3.md r3, revision 2 (DRAFT, awaiting independent code review)

Revision 2 (11:2x Berlin): implements uni's interpretations of 11:12 and dahill's CODE_PASS_CONDITIONAL
`CODE_PASS_v8.3.json` = 19bea022…68aa (on revision 1, manifest a9841b45; that PASS does not cover this manifest).

Implementer: TMbot · Research (non-author). Nothing run except the unit tests (no probe, no production run).
Plan: `PLAN_v8.3.md` r3 = `e2debca54bb2d303b910eb5d32d691b14652bd61422b53e3eadf757d4c5784ec` (listed in `HASHES_v8.3.sha256`
= `8b3b53e0…9e70`; diff r2→r3 = `DIFF_r2_to_r3.patch` b4dd2a23…23e5). Review: dahill `PLAN_REVIEW_v8.3_r3.json`
= `43a6e207…8b3f` (PLAN_PASS_CONDITIONAL; A1, A2 implemented here; A3 = closed in RELEASE/CONFIG, plan untouched).
Base: frozen v8.2 code, manifest `CODE_HASHES_v8.2.sha256` = `a6544b6f…b643` (re-verified 32/32; nothing under v8.2/ modified).
Science bundle = v8.2 (HASHES_v8.2_E1 304bc28d), unchanged: `spec.PLAN_DIR` still points at v8.2/; v8.3 items live in `spec.V83`.
Seed reuse condition (plan §1): `../SEED_EVIDENCE_v8.3.md` = `e33710d7e552b05519905f8d2a4c74027c57e45ff69a26549771b9cce2f85dc9`.

## Changes (only what r3 + A1/A2 specify)
| Plan item | Where | What |
|---|---|---|
| C1 thread readback | `v81/hostenv.py`, `run_probe.py`, `run_production.py`, `pipeline.run_all` | OMP/MKL/OPENBLAS env + `torch.get_num_threads()` read back and logged (ledger `run_start`, every `phase_start/phase_end`, probe `attempt_start`); mismatch → NOT_LAUNCHED / PROBE_NOT_STARTED. Interop threads logged only. |
| C2 probe | `v81/probe.py` `measure_probe`, `run_probe.py` | Replicate r1 train/val data (32+8 data episodes), WM h96 full fit (2000 updates), other 8 neural fits first 200 updates (same batch indices + cosine schedule), LIN 3 full closed-form fits, 2 train-entry episodes per arm×structure (λ=1, 96 decisions, 64 cand.), 2 PI episodes. Hard cap 450 CPU-s (process CPU since start), checked every 10 updates / after every episode. SeedGuard mode `probe`: train/val seeds only. |
| C2b linearity / A1 | `probe.linearity_ok`, `probe.decide` | ratio = 10 × CPU(first 200 updates of WM h96 training loop) / CPU(full loop); inclusive [0.87, 1.15]. Outside, or cap hit, or incomplete → `PROBE_DIAGNOSIS`, projection = None, no CONFIG. Measurement stops right after a failed linearity check. |
| A1 attempts | `run_probe.py` | Every invocation = attempt N in `probe/PROBE_LEDGER.jsonl` (attempt_start/attempt_end); `PROBE_v8.3_attempt<N>.json`, `LOAD_SAMPLES_probe_attempt<N>.txt` exclusive-create; refused once `CONFIG_v8.3.json` exists. |
| C3 projection/caps | `probe.fits_projection`, `episode_phase_projection`, `caps_from`, `decide` | fits = 3 × (windows + WM h96 full + 10 × each other neural 10 % cost incl. build+eval + LIN full). tuning/test/in_distribution/adaptation = the v8.2 preflight formulas (counts from CONFIG_v8.2) with probe unit costs (test/ID/adaptation at per-arm max). Caps: fixed 260 + ceil(1.25 × projection) for the 5 measured phases; ONE budget: global cap = sum of caps, gate sum ≤ 3600; ladder a → b → c cumulative; else NOT_LAUNCHED. CONFIG_v8.3.json written only on GATE_PASS (exclusive). |
| K1 | `budget.Budget` | `global_cap` (= sum of caps) AND a separate `k1_cap` = 3600: cumulative CPU-s > 3600 → CapExceeded("K1 …") → INVALID, also when every phase is under its cap; also in the static end-of-phase check and `finalize` (`k1_respected`). |
| K2 | `probe.apply_k2`, `toy_episode_projection` | R = P_fits / 336.8 reported. Toy projections per episode phase recomputed from the bound v8.2 toy preflight (7e164362) with the same formulas and the same ladder step (reproduces the stored v8.2 values exactly; tested). If R > 1.5 and measured < 0.8 × toy × R → toy × R. Per-phase record in `decision.tries[*].k2`. |
| C4 start rule | `hostenv.start_rule` | load1 < 1.0, MemAvailable ≥ 4 GiB, no process whose cmdline contains a pattern (default `exp001`, `run_arm.py`; env `WM_EXCLUDE_PATTERNS`). Unreadable value → fail closed. Logged in probe ledger / production ledger `start_rule` / NOT_LAUNCHED jsonl. |
| A2 host | `hostenv.host_fingerprint/same_host`, `release.check_probe_config`, `run_production` | hostname, CPU model, logical CPU count recorded in probe output, CONFIG and RELEASE; production refuses if the running host differs from the probe's. |
| C5 sidecar | `../sidecar/load_sidecar.py` (outside code/), `hostenv.start_sidecar` | own session (setsid), stdin/stdout detached, SIGHUP ignored; every 30 s JSON line: wall clock, loadavg, runnable/tasks, MemAvailable, steal ticks + steal fraction, mean CPU MHz, watched process CPU-s; stops when the run exits. `gaps()` lists wall gaps > 90 s. |
| C6 ledger | `budget.Budget(on_event)`, `pipeline._fit_progress`, `pipeline.phase_fits` | `phase_start/phase_end` (process CPU-s, wall, thread readout, cap, n_checks, max jump), `fit_end` per fit (arm, structure, rep, updates, fit CPU-s), `fit_progress` every 100 updates, `run_start`, `final` with process CPU-s. |
| C6b cap checks | `models_torch.train(check_every)`, `pipeline._run_set`, data/adaptation loops, `stats._blocks` | fits every 10 updates (+ after the last update; LIN: once after each fit); otherwise after every episode (closed-loop, data, PI); bootstrap: after every block of 1000 resamples (bit-identical results; tested). The per-check jump is recorded (Budget.max_jump, probe max_check_jump_cpu_s). No tolerance. |
| C7 write-once | `ledger.write_sealed` ('xb'), `ledger.write_once`, `pipeline.save_json(once=True)` for normalization.json | second write raises FileExistsError. Run end comes from the ledger `final` event. |
| Release | `v81/release.py`, `run_production.py` | v8.3 start rule: v8.2 bundle + PLAN_PASS_v8.2 (content checked), v8.3 bundle + review r3 + A1–A3 closed, grid override, diff v8.2→v8.3, non-author, CODE_PASS on the v8.3 manifest, seed evidence, sidecar sha, probe GATE_PASS + CONFIG equal to the recomputed decision, A2 host. Relative paths resolve against WM_ROOT / WM_AUDIT_ROOT. NOT_LAUNCHED records are append-only (`NOT_LAUNCHED_run_output.jsonl`). |
| Portability | `spec.py` | `WM_ROOT` (dir holding v8.1/ v8.2/ v8.3/; default derived from this file), `WM_AUDIT_ROOT` (default /workspace), `WM_OPTION1_DIR` (tests only; test skips if absent). |
| restored | `run_preflight.py` | v8.2 toy preflight, `main()` = v8.2 logic except the output path (arg / `WM_TOY_PREFLIGHT_OUT`); `attempt_main()` = numbered append-only attempt on this host (`probe/TOY_LEDGER.jsonl`, `TOY_PREFLIGHT_attempt<N>.json`), same gates as the probe. `v81/preflight.py` byte-identical to v8.2 (tested). |
| new | `v81/gates.py` | shared gates: C1, C4, frozen manifest, dahill CODE_PASS on the CURRENT manifest, sidecar fields readable; attempt ledgers; K2 reference selection per host. |
| tests | `tests/test_v83_mechanisms.py` (+28), updated release tests | 106 passed (78 carried over; `test_check_start_w1_reasons` removed with the v8.2 release rule). |

## Interpretations by uni 11:12 (not a plan change) — as implemented
1. **×3 replicates — confirmed.** P_fits = 3 × (formula + window build); R uses this all-replicate P_fits (dahill: v8.2 fits 336.8 = 3 × win_rep … covers 3 replicates, like for like).
2. **LIN one full fit per order — confirmed.**
3. **Linearity on training-loop CPU — confirmed;** dahill: the total CPU of the h96 fit (build + train + eval) is reported too: `measurements.linearity.full_fit_total_cpu_s`, `decision.full_fit_total_cpu_s`.
4. **Episode phases — condition: every per-unit cost from the probe.** Audited (table below); no toy-sourced term existed. Per dahill ruling 4 each episode phase = max(v8.2 formula with probe unit costs, episodes × measured episode cost); both values and the winner are in `decision.tries[*].measured_projection_detail`. Tests: `test_projection_uses_no_toy_input` (NaN toy ⇒ identical projection), `test_projection_inputs_all_produced_by_probe`, `test_episode_phase_max_rule_reports_both`.
5. **K2 never across hosts.** "Box" = `spec.BOX_FINGERPRINT` (hostname `grok-bot-vm-408406529`, CPU model `Intel(R) Xeon(R) Processor`, 8 logical CPUs), read on the box on 2026-11-11 because the v8.2 file 7e164362 records no fingerprint (assumption: 10.11. preflight ran on the same VM; any difference in any field ⇒ non-box). Box: K2 reference = v8.2 file, toy fits 336.8. Non-box: `run_preflight.py` must have produced a TOY_OK attempt on the same host (120 s cap, `preflight_within_cap` true, file sha as logged); the latest such attempt is used, R = probe P_fits / that file's `projection_cpu_s.fits`. Without it `run_probe.py` refuses to start (PROBE_NOT_STARTED). The reference (kind, file, sha, toy_fits, attempt) is written into the probe output and CONFIG and must be bound in RELEASE (`k2_reference`); `release.check_probe_config` re-checks kind vs. probe host and recomputes.
6. **K2 under the ladder — confirmed (stricter reading below, S1).**
7. **Gate rounding — confirmed** (cap-sum rule).
8. **Adaptation checks after episodes only — confirmed;** dahill ruling 7: report the max jump: production ledger `phase_end` (adaptation) `max_check_jump_cpu_s` and FINAL_STATUS `budget.max_check_jump_cpu_s`; probe: `decision.adaptation_projected_max_jump_cpu_s` = 3 × 100 × max per-update cost (or 3 × max LIN refit) + max episode cost.
9. **Probe requires dahill's CODE_PASS on the current manifest.** `gates.find_code_pass`: a file `<WM_AUDIT_ROOT>/audit_dahill_v8.3/CODE_PASS_v8.3*.json` with `reviewer` = "dahill", `verdict` ∈ {CODE_PASS, CODE_PASS_CONDITIONAL}, `plan_sha256` = r3, and `manifest_sha256` (or v8.2-style `code_manifest_sha256`) = sha256 of the current `CODE_HASHES_v8.3.sha256`, which must itself verify. The existing PASS for a9841b45 does not satisfy it (tested). Same gate for the host toy attempt (stricter, S6). Fail closed.

### Per-unit cost terms of the projection and their source (audit for interpretation 4)
| phase | term | source |
|---|---|---|
| fits | windows (train+val windows, replicate r1) | probe `windows_per_replicate_cpu_s` |
| fits | WM h96 full fit (build + 2000 updates + val eval) | probe `unit.WM_96.fit_cpu_s` |
| fits | other 8 neural fits ×10 (build + 200 updates + eval) | probe `unit.<arm>_<s>.fit_cpu_s` |
| fits | LIN 3 orders (full closed form) | probe `unit.LIN_<o>.fit_cpu_s` |
| tuning, test, ID, adaptation | per-episode MPC cost (fallback-scaled to 96 decisions) | probe `unit.*.episode_cpu_s` (2 train-entry episodes each) |
| tuning, test, ID, adaptation | scenario build | probe `scenario_build_cpu_s` |
| adaptation | PI episode | probe `pi_episode_cpu_s` |
| adaptation | data episode (288 transitions) | probe `data_episode_cpu_s` (40 r1 train/val episodes) |
| adaptation | fine-tune per-update cost (per-arm max) | probe `unit.*.per_update_cpu_s` (WM h96: full loop/2000; others: 10 % loop/200) |
| adaptation | LIN refit (42 episodes) | probe LIN fit × 42/32 |
| all | episode/update counts (40, 320, 40, 3 reps, 5 λ, 100 updates, 36 fine-tunes) | CONFIG_v8.2 (counts, not costs) |
| K2 only | toy episode projections, toy fits | v8.2 file (box) or same-host toy attempt — never in the measured projection |

## Remaining items — stricter reading applied (marked)
- **S1 (stricter)** K2 toy reference = the toy projection WITHOUT ladder reduction (larger), for every phase still run; a phase dropped by ladder (b) stays 0.
- **S2 (stricter)** A refused start (start rule, gates) is a numbered attempt (PROBE_NOT_STARTED / TOY_NOT_STARTED), for the probe and the host toy.
- **S3 (no alternative)** Ladder a, b, c: r3 names it explicitly; the v8.2 W1 'a only' restriction is not carried over (plan text, not an interpretation).
- **S4 (stricter)** Fresh-seed branch not implemented: without the bound SEED_EVIDENCE the probe and production refuse (fail closed).
- **S5 (stricter)** C5 fields must be readable on the host (loadavg, runnable, MemAvailable, steal, CPU MHz, process CPU-s; checked with the bound sidecar script itself) — else the probe / toy attempt / production do not start. On a host without /proc (macOS, Windows) this refuses.
- **S6 (stricter)** The host toy attempt runs under the same gates as the probe (C1, C4, frozen manifest, CODE_PASS, sidecar) and with its own sidecar.
- **S7 (no alternative)** Probe episode scenarios: train seeds r1[0], r1[1], TRAIN entries 1–2, onset 120.
- **S8 (stricter)** "4 GB" = 4 GiB; unreadable load / memory / process list = refuse.

# Code revision r3 (from manifest 2b4528cc; requested by dahill + uni; ~12:00–12:20 Berlin)
Plan binding: **plan r4 = PLAN_v8.3.md r3 (e2debca5…84ec) + ERRATUM_v8.3_E1.md (ea6421146159f2ab549b6a9290906737a6f1778220586c9643744022a3651cb0) + ERRATUM_v8.3_E2.md (3d6f8c0a54101a681bbaf8a05c2725f1c8cb6efea70249c98d2782fa0c1b1bcf); E2 wins over E1.** HASHES_v8.3.sha256 (uni, rewritten 11:59) = c5b6de26…1025 lists both errata; `verify_v83_bundle` now requires both. Note: that HASHES file lists PLAN_v8.3.md three times (same sha), harmless. No separate PLAN_v8.3_r4.md existed in v8.3/ at freeze; E2's first line defines r4.

## Items of the r3 brief
1. **R_fresh + drift (descriptive, box only)**: `gates.k2_fresh` — on the box, with a same-host TOY_OK attempt: R_fresh = P_fits / fresh fits, drift_flag = |fresh/336.8 − 1| > 0.15 (both values reported, `k2_fresh` in the probe output). Gate R on the box still P_fits/336.8. Box K2 reference carries `host_assumption` ("assumed same host", dahill P2). Non-box: unchanged (None).
2. **Production accepts only `CODE_PASS`** (`spec.CODE_PASS_VERDICTS_PRODUCTION`, release.check_start: both the RELEASE record and the file). Toy + probe still accept CODE_PASS_CONDITIONAL.
3. **Hermetic tests** (dahill req 6): the 3 tests use an isolated tmp CODE_PASS dir via monkeypatched `spec.CODE_PASS_DIR` (copies only, never deletes). They pass with or without real CODE_PASS files.
4. **SHA256SUMS self-listing**: was my ad-hoc shell command, not code. Fixed generator `gates.write_sha256sums` (excludes itself, exclusive create, tested). The read-only probe_unicorn files are untouched; `probe_unicorn/SHA256SUMS_probe_unicorn_v2.txt` written with it (it lists the old v1 list as a normal file).

## ERRATUM E1 / E2 (plan r4) — as implemented
- **E1.1 warm-up** (`v81/warmup.py`): 200 updates, throwaway model, **stricter reading: same architecture (arm, structure) as the segment that follows**, train windows only, numpy `default_rng([970001, 7])`, torch global RNG forked/restored (MT.build seeds it), weights discarded. Test: fit results bit-identical with/without warm-up, torch RNG state unchanged; seed 970001 not in any seed set. Ledger event `warmup` (+ CPU-s per 50-update block, E2.2). Probe: before EACH timed neural fit segment (the WM h96 pair = one segment); counts toward 450 s. Not before LIN (closed form, no updates) nor before episode segments (no updates) — interpretation.
- **E2.1 (replaces "not part of any projection")**: production runs ONE warm-up per process start (before the first neural fit, inside the fits phase → counts toward the fits cap, global cap, K1; `Budget.warmup` tallies it). `production_process_starts` is a RELEASE field (draft: 1); the probe refuses if absent.
- **E2.2 + dahill req (2)**: two full WM h96 fits back-to-back after one warm-up, each with its own first-10 % window, both reported. Per fit k: post-window rate r_k = (train_k − first_k)/1800, first-window excess e_k = first_k − 200·r_k; W_warmup = measured warm-up CPU-s (max over the probe's warm-ups). Recurrence (implementer rule, flag for uni/dahill): e1,e2 ≤ 0 → none; e2 < 0.5·e1 → per process; else per fit. Fixed cost added to fits projection (conservative superset of E2.1's "W × n_coldstarts"): none → W_warmup × process starts; per process → (W_warmup + e1) × process starts; per fit → max(W_warmup, e1, e2) × 27 production neural fits. Linear part of the full fit = build + eval + max_k r_k × 2000. Other neural fits keep the v8.2 rule 10 × (200-update fit) (already contains their own first-window excess; conservative, possible double count in per-fit mode). **Linearity gate = fit 2 only** (post-warm-up window of fit 2 vs fit 2 full, bounds unchanged); fit 1 reported. E2.2's phrase "only after adding W to the projection per fit" is satisfied by the per-fit W term; an "adjusted" ratio (excess removed) would be identically 1 under this W definition, so the raw fit-2 ratio gates (stricter).
- **dahill req (1)**: CPU-s per update in blocks of 50 for every measured fit and every warm-up (`block_per_update_cpu_s`, ledger `probe_fit_end` / `warmup` + probe output).
- **Reconciliation W (dahill) vs E1.1/E2.1**: E1.1 "warm-up excluded from projections" is superseded by E2.1 (warm-up CPU is part of W). dahill's W (residual cold-start per real fit) = e_k above; both enter as described.
- **E2.3 / dahill req (3)**: probe and host toy attempt sidecar interval `PROBE_SIDECAR_INTERVAL_S` = 2.0 s (env `WM_PROBE_SIDECAR_INTERVAL_S`, asserted ≤ 2); production stays 30 s (C5). Probe output `sidecar_summary`: max load1, min CPU MHz (sidecar's mean over CPUs — not the pinned core; pinned core MHz recorded at pin and at end separately), max steal fraction, steal ticks over the timed segment window.
- **E2.4 (replaces E1.3) + dahill req (5)**: `run_baseline.py` = own numbered attempt (probe/BASELINE_LEDGER.jsonl, BASELINE_attempt<N>.json), 600 s wall, 2 s samples, no excluded process at any sample (exp001, run_arm.py, stricter: v8.3 run_probe/run_production/run_preflight) else BASELINE_INVALID. Threshold = 1.5 × p95 (numpy default percentile). `set_start_threshold.py` writes field `start_rule_threshold` into RELEASE_v8.3_DRAFT.json once (refuses if present: "no later adjustment"). Toy attempt, probe and production recompute it from the bound baseline file and refuse if absent (non-box), changed, other host, tampered. Box without the field: < 1.0 (E2.4). Production: RELEASE field must equal the probe's.
- **E1.2 / E2.5 / dahill req (4) pinning**: `hostenv.cpu_topology` from /sys (Intel PMU cpu_core/cpu_atom lists; else differing cpu_capacity; else differing cpuinfo_max_freq; cpuinfo flag 'hybrid' without a P-core list → fail closed). Core = highest-numbered logical CPU in the P-core list (unicorn: P-cores 0–7 → core 7, cpu7 max 4700 MHz). `os.sched_setaffinity` + read-back; sidecar child un-pinned. Recorded (core list, core, max MHz, MHz at pin/end) in probe output and CONFIG; RELEASE field `pinning` must equal the probe's; production re-detects, must match exactly and pins identically, else NOT_LAUNCHED. The host toy attempt is pinned the same way (E2.5) and the non-box K2 reference requires a TOY_OK attempt pinned identically → **unicorn toy attempt 1 (11688cdb…, unpinned) no longer qualifies; a fresh pinned toy attempt 2 is needed** (E2.5). Box: not hybrid → no pinning (unchanged).
- **E1.5 / E2.7**: a linearity PROBE_DIAGNOSIS in any attempt under the errata (ledger `erratum` = "E1+E2") blocks every later probe attempt (`PROBE_BLOCKED_E1_5`, v8.3 NOT_LAUNCHED). Attempt 2 (pre-erratum) does not block; cap diagnoses do not block.
- **Flag (parent, E1.4 vs E2.5)**: E1.4 named toy attempt 1 as K2 reference; E2.5 replaces it with a fresh pinned toy attempt 2 on the same core. Implemented per E2. Had attempt 1 been used, R would mix unpinned toy with pinned probe conditions.

## dahill requirements (11:59) — marked
- dahill req 1: blocks of 50 — done. dahill req 2: two fits / W / mode — done (rule above). dahill req 3: sidecar ≤ 2 s probe — done. dahill req 4: core list bound, production identical — done. dahill req 5: threshold fixed in RELEASE from a measured baseline, refuse if absent/changed — done (E2.4 procedure). dahill req 6: hermetic tests — done. dahill req 7: SHA256SUMS v2 + generator — done.

## Next steps on unicorn (nothing run by me)
1 copy new layout; 2 `run_baseline.py` (10 min, nothing else running); 3 `set_start_threshold.py <BASELINE_attempt1.json>` (changes the RELEASE draft → new draft sha); 4 `run_preflight.py` (pinned toy attempt 2); 5 `run_probe.py` (attempt 3). Requires dahill CODE_PASS on the r3 manifest.

### r3 freeze fix (post-freeze test run)
- First post-freeze run: 1 failed / 144 passed in all three audit configurations: `test_process_starts_field` read the REAL
  `RELEASE_v8.3_DRAFT.json` (value=None falls back to the draft), which by then carried `production_process_starts: 1`.
  Not a code defect, a non-hermetic test. Fixed: the test passes an absent tmp draft; `tests/conftest.py` gains an autouse
  fixture pointing `S.RELEASE_DRAFT_FILE` to an absent tmp file, so no test reads the real draft (same spirit as dahill req 6).
  Manifest, both patches and the draft re-generated after this fix. No production code changed by the fix.
