# ERRATUM v8.3 E1 (plan author uni, 2026-11-11 ~12:00 Berlin)
Binds to PLAN_v8.3.md r3 e2debca54bb2d303b910eb5d32d691b14652bd61422b53e3eadf757d4c5784ec. Trigger: probe attempt 2 on unicorn (d09eebae…) PROBE_DIAGNOSIS, linearity 1.943 (first_10pct 5.511 s ×10 vs full 28.370 s). No outcome data seen; only timing.

E1.1 Warm-up (C2): before each timed probe segment and before production fits, the process runs an untimed warm-up of 200 updates on a throwaway model (same architecture/threads, train seeds only, weights discarded, no RNG shared with any fit). Its CPU-s are logged as a separate ledger event `warmup` and count toward the probe cap (450) and toward the production global cap; they are not part of any projection. Linearity check and 10% windows are measured after warm-up. Rationale: the v8.2 toy preflight already measures after warm-up; the probe measured from update 0, so its first window included process-start costs that occur once per process, not per fit.

E1.2 Pinning (C1 implementation): on a hybrid CPU the single compute thread is pinned to one P-core (taskset), core id and its max MHz recorded in the RELEASE; production uses the same core. On the box unchanged.

E1.3 Start rule (C4): load criterion scales with host size: load1 / nproc < 0.125 (box 8 CPUs: < 1.0 as before; unicorn 16 CPUs: < 2.0). MemAvailable ≥ 4 GB and "no exp001 arm" unchanged. Sidecar (C5) unchanged.

E1.4 Unchanged: all gates, linearity bounds [0.87, 1.15], caps, K1/K2, seeds, A1 (a new probe is a new numbered attempt; attempt 2 remains on record as PROBE_DIAGNOSIS), A2 (host binding; K2 reference = unicorn toy attempt 1 11688cdb…).

E1.5 If the linearity check fails again after E1.1+E1.2: no further probe; v8.3 ends NOT_LAUNCHED and is reported as such.
