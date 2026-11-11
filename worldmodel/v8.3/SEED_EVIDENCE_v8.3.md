# SEED_EVIDENCE_v8.3 — v8.2 produced no tuning, test or adaptation episode

Author: TMbot · Research (non-author implementer). Written 2026-11-11 10:56 (Europe/Berlin). Read-only after writing.
Purpose: condition in PLAN_v8.3.md r3 §1 (sha256 e2debca54bb2d303b910eb5d32d691b14652bd61422b53e3eadf757d4c5784ec): "before launch, TMbot shows from `run_output/` and the ledger that no tuning, test or adaptation file exists". If this evidence holds, v8.3 keeps tuning 7300–7319, test 9500–9539, adaptation 4100–4109.
Method: read-only inspection of `/workspace/nested_swarms/v8/worldmodel/v8.2/` (nothing under v8.2 was modified). Code manifest of the run re-verified: `code/CODE_HASHES_v8.2.sha256` = a6544b6fdafb11bf853bd1b57ce8d3f8d508f06dbafebcafb2a7dbede2a7b643, 32/32 files OK.

## Cited files (sha256)
| file (relative to v8.2/) | sha256 |
|---|---|
| `run_output/FINAL_STATUS.json` | `42c77e9ee29ed0894c82bb113b800adb3fc06fb409540a0767c6655933ec156e` |
| `run_output/LEDGER.jsonl` | `e67434423fdc7e17a542fc9957d92012134b5702cd571be401c03a762c0ad542` |
| `run_output/normalization.json` | `a8f5b344bf93b9856e701379f71d5a669cf4acc73c3f80e36cbe4b621c332ebf` |
| `production_run.log` | `9c35fae907617314e575fc17388d9f1b1ea4d6f4aecde7de15071fea51a940fd` |
| `RELEASE_v8.2.json` | `00b4766dcd7af84ca10bddd9724d3d21daf7c83e49fc223df7f9f783347c0b9e` |
| `code/CODE_HASHES_v8.2.sha256` | `a6544b6fdafb11bf853bd1b57ce8d3f8d508f06dbafebcafb2a7dbede2a7b643` |
| `code/v81/pipeline.py` | `02f0359d559602cc61be03c12a62fc945141475cb5ce823eb659509c380a533f` |
| `code/v81/seeds.py` | `3631b365071b449bae16fddc9816e3d38ad59b4835c3b7f0f95344de89cc82a0` |
| `code/v81/preflight.py` | `b56810b00e826c9d4f4041d6ca5f987aa22c6125ca8c4dba94f7b7c28aea3d5d` |
| `PREFLIGHT_PROJECTION.json` | `7e1643629afe9f4d0dc5a1401d62e2e5b5b6bfda48069810c3a2b4b61a670a29` |
| `RESULTS_v8.2.md` | `83c45978d704fc6eb24e41dc7e386ac6fda0b7f31f886ecfe8d7bc09b0c4b8e5` |

## E1. `run_output/` contains exactly three files
`ls run_output/` → ['FINAL_STATUS.json', 'LEDGER.jsonl', 'normalization.json'].
Absent: `fit_info.json`, `tuning_rows.npz`, `COMPARATOR_SELECTION.json`, `test_results.npz`, `PRIMARY_RESULT.json`, `SECONDARY_RESULTS.json`, any `BOOT_INDEX_*.npy`, `heldout_mse_rows.npz`. A recursive search of v8.2/ for `fit_info*`, `*tuning*`, `COMPARATOR*` and `test_results*` returns nothing.
`normalization.json` holds only normalisation statistics for the 3 replicates (keys ['0', '1', '2']); it is written at the end of phase `data` (pipeline.py line 137) from train/val data episodes only.

## E2. `FINAL_STATUS.json` (quoted fields)
- `"outcome": "INVALID"`
- `"budget.cap_exceeded": "phase fits: 803.61 > cap 800.0"`
- `"budget.used"`: data = 0.215034948, fits = 803.670797039, tuning = 0.0, test = 0.0, in_distribution = 0.0, adaptation = 0.0, pi_oracle = 0.0, bootstrap_stats = 0.0, audit = 0.0
- `"primary": null`, `"selection": null`, `"selection_fallback_flags": null`
- `"validity": {"stage0_model_imports": true}` (only the stage-0 check of phase `data` ran)

## E3. `LEDGER.jsonl` (complete content, 1 line)
```
{"event": "final", "outcome": "INVALID", "time_unix": 1794315887.3218334}
```
time_unix → 2026-11-10 14:04:47 Europe/Berlin. There is no `test_begin` event, no `sealed` event (COMPARATOR_SELECTION / PRIMARY_RESULT) and no `boot_matrix_saved` event. In the v8.2 code `test_begin` is appended before any test episode (pipeline.py line 242) and the selection is sealed at the end of tuning.

## E4. `production_run.log` (complete content)
```
{
 "outcome": "INVALID"
}
```

## E5. Code path (frozen v8.2 code, manifest above)
- `Run.run_all` calls the phases in the order data → fits → tuning → test → … (pipeline.py lines 485–493); `CapExceeded` raised inside `phase_fits` jumps straight to the `except CapExceeded` branch (line 497), so `phase_tuning`, `phase_test`, `phase_in_dist`, `phase_adaptation` were never entered. `fits` raised before `save_json("fit_info.json", …)` (line 167), consistent with E1.
- Tuning scenarios (seeds 7300–7319) are only constructed inside `phase_tuning` (line 173); test scenarios only inside `phase_test`/`phase_in_dist`; adaptation data (4100–4109) only inside `phase_adaptation`. In addition `SeedGuard` (seeds.py) refuses test seeds before the sealed selection and adaptation seeds before the sealed primary; neither seal exists (E3).
- Phase `data` draws only `train` and `val` purposes (train_data / val_data seeds of CONFIG_v8.2).

## E6. Preflights
Both v8.2 toy preflights used seeds 1–4 only (`PREFLIGHT_PROJECTION.json` `measurements.seeds`). Disclosure: the preflight's HELDOUT-scan timing builds 500 dummy dictionaries whose `seed` field is numbered from 7300 (preflight.py, `recs = [...]`); these are metadata records for timing the scan, no plant episode is simulated and no outcome is computed. No tuning, test or adaptation episode exists from the preflights either.

## Conclusion
v8.2 generated no tuning, test or adaptation episode and no outcome on these seeds. The condition of PLAN_v8.3 r3 §1 for reusing 7300–7319, 9500–9539 and 4100–4109 is met. This file is bound by sha256 in RELEASE_v8.3 (draft).
