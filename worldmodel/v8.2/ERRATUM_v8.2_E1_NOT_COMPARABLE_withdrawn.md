# ERRATUM_v8.2_E1 (minimal erratum to bundle HASHES_v8.2.sha256 = 733cc25c235d6717f42223ce8868753bbccbfa80a234a4a88a8064deddb9b1d1)

Author: uni · KI-Forschung, 2026-11-10 (Europe/Berlin). Written in response to dahill PLAN_REVIEW_v8.2.json (f549a67a…d0f2, PLAN_BLOCKED B1/B2) and the coordinator's (mixxy) decision. Everything in PLAN_v8.2.md, CONFIG_v8.2.json and FEASIBILITY_v8.2.md not named below remains binding unchanged. On conflict, this erratum wins.

## E1.1 (B1): lambda grid
`lambda_grid` is replaced by **[0.5, 0.625, 0.75, 0.875, 1.0]** (previously [0.5, 1, 2, 4, 8]), identical for all arms (WM, GRU, LIN, RC_only).
Still 3 structures x 5 lambdas = 15 configs per arm, 1800 tuning episodes and 172800 planner decisions per arm. Because lambda does not change the per-episode cost (same 64 candidates), FEASIBILITY_v8.2.md, the phase caps, gates and shrink ladder hold unchanged.
Reason: applied to the v8.1 tuning rows, the old grid forced WM to its fallback lambda=0.5 while RC_only s2 at lambda=1 sits on D_max, so the comparison would have been decided by the grid. lambda >= 2 was infeasible for every arm (D >= 0.49).

## E1.2 (B2): no comfort hypothesis
v8.2 has only H1 (energy). No H2/HC. Reason: comfort is a constraint of the selection (D <= D_max), so all arms get pushed toward the same comfort limit, and a comfort-superiority hypothesis cannot be tested fairly in the same design. dD stays a purely descriptive secondary measure with no gate and no claim. A comfort strand would be a separate experiment with its own selection rule, e.g. lambda at matched energy.

## E1.3: post-hoc disclosure
The v8.2 hypothesis (energy saving at constrained comfort) and the selection rule were formulated only AFTER seeing the v8.1 results (RESULTS_v8.1.md 64f16f71…, dahill RESULT_AUDIT_v8.1.json f7fdcb32…). That is why entirely new tuning, test, adaptation and bootstrap seeds (as in PLAN_v8.2.md) are mandatory, and no v8.1 test, in-distribution or adaptation data may be used for any v8.2 decision.

## E1.4: note, RC_only at the threshold
D_max equals exactly the v8.1 tuning D of RC_only s2 at lambda=1. On the new tuning seeds 7300-7319, whether that config is feasible is therefore essentially chance. This is accepted. The predeclared selection and fallback rules apply unchanged, and RESULTS must state this point explicitly.

## E1.5: fit reproduction (report only)
Training, validation and init seeds equal v8.1, so RESULTS reports whether each fit reproduces v8.1's fit_info (per arm x structure, max abs difference of the logged metrics). This is reporting only: no gate, no consequence for validity.

## E1.6: NOT_COMPARABLE (coordinator decision mixxy, precise wording per dahill)
Trigger, evaluated on the SEALED selection table before the test: (a) the feasible set F is empty for WM, or (b) none of GRU, LIN and RC_only has a non-empty F (no feasible comparator). Then the primary verdict is NOT_COMPARABLE: no ADVANCE possible, reported honestly. Order of precedence in §11: INVALID before NOT_COMPARABLE before ADVANCE/NO_ADVANCE. On NOT_COMPARABLE the test phase still runs unchanged (run logic stays the same), and all test values, including P1-P4, are reported descriptively only, with no verdict. The comfort limit is inclusive (D <= D_max), as already stated in §8/CONFIG.
