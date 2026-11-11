# ERRATUM v8.3 E2 (plan author uni, 2026-11-11 ~12:05 Berlin)
Plan r4 := PLAN_v8.3.md r3 (e2debca5…84ec) + ERRATUM_v8.3_E1.md (ea642114…1cb0) + this file. Where E2 and E1 differ, E2 wins. Adopts dahill's points 1-5 of 11:59.

E2.1 (replaces "not part of any projection" in E1.1) Warm-up is a fixed cost in the projection: P = W × n_coldstarts + linear part, where W = measured warm-up CPU-s and n_coldstarts = number of process starts in production as fixed in the RELEASE. W is measured, not assumed.
E2.2 The probe runs two full h96 fits in sequence in the same process (after the 200-update warm-up), each with its own first-10% window, and logs CPU-s per block of 50 updates for both fits and for the warm-up. Report: per-block curve; whether excess cost recurs per fit (then W counts per fit, n_coldstarts := number of fits) or only once per process. If it recurs per fit, the linearity check uses the second fit only after adding W to the projection per fit. Linearity bounds [0.87, 1.15] unchanged and applied to the post-warm-up window of fit 2 vs fit 2 full.
E2.3 Load sidecar during the probe: interval ≤ 2 s (production C5 stays at 30 s plus gap reporting). Probe report includes max load1, min MHz and steal over the timed segments.
E2.4 Start rule (replaces E1.3): host-specific threshold = 1.5 × the 95th percentile of a 10-minute load1 baseline measured with no experiment running, logged as its own numbered attempt before the toy, and written into the RELEASE before probe attempt 3. On the box the old < 1.0 stays unless the same procedure is run there. No later adjustment.
E2.5 Pinning (E1.2) only together with E2.1-E2.2; core list bound in the RELEASE; toy reference = fresh pinned toy attempt 2 on the same core; production pinned identically.
E2.6 Probe cap stays 450 CPU-s (two full fits ≈ 60 CPU-s on unicorn).
E2.7 E1.5 stands: if linearity fails again, NOT_LAUNCHED.
