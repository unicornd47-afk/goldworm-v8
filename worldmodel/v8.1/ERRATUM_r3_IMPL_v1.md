# Erratum r3-IMPL v1 (plan author: uni), binds HASHES_r3.sha256 b2fac5bf5e57e9e7013445aba7b57b7e192aa10e4c72e49247d00d0fc6d70ed7
Written before any test-purpose data exists. Changes nothing in the hypothesis, endpoints, gates, seeds, interventions or caps.

E1 (Q1): CONFIG_v8.1_r3.json labels LIN as "ARX ridge". This is an author error. The arm meant is the same LIN as Option1/v3 ("same arms, no new arms", PLAN section 2), i.e. thermal5/model_lin.py subspace ID + finite-history Kalman predictor, ridge rel 1e-6. The structure grid is order {3,5,8} (3 structures, same count as the other arms, so the equal-search rule is unchanged; order 5 = Option1 bit-for-bit).
E2 (Q8): FEASIBILITY_v8.1_r3.md says "3 matrices of 10000 x 40" for adaptation. Adaptation has 10 clusters (test seeds 9100-9109), so its matrices are 10000 x 10; primary and secondary stay 10000 x 40. Order-statistic rule [249]/[9750] is unchanged.

Author acceptance of the implementer's interpretations (no erratum needed, they follow the plan or fill a gap neutrally across arms):
Q2 RC equations with parameter counts 7/11/13; Q3 no arm sees future T_o, RC holds last measured T_o; Q4 warm-up under the frozen PI as the plan states (identical for all arms); Q5 common random numbers per seed across arms and entries; Q6 validation logged only, final weights used; Q7 adaptation episodes from seeds 8000+ (disjoint from tuning/test), 100 Adam updates lr 1e-4 for WM/GRU/RC_only, LIN one refit on train + K episodes.
