"""End-to-end smoke run of the production pipeline code on preflight seeds 1-4 with tiny counts
(no production seed is touched; numbers are meaningless)."""
import json

from v81 import pipeline as PP


def test_smoke_pipeline(tmp_path):
    out = tmp_path / "smoke"
    run = PP.Run(PP.smoke_design(n_updates=5, adapt_updates=2), str(out))
    final = run.run_all()
    assert final["outcome"] in ("ADVANCE", "NO_ADVANCE")
    v = final["validity"]
    for k in ("stage0_model_imports", "equal_search_effort", "no_heldout_pre_test", "selection_sealed_before_test",
              "identical_inputs", "all_caps_respected"):
        assert v[k] is True, k
    ev = [json.loads(l) for l in (out / "LEDGER.jsonl").read_text().splitlines()]
    order = [e.get("file") or e["event"] for e in ev]
    assert order.index("COMPARATOR_SELECTION.json") < order.index("test_begin") < order.index("PRIMARY_RESULT.json")
    assert order.index("BOOT_INDEX_primary.npy") < order.index("PRIMARY_RESULT.json")
    sel = json.loads((out / "COMPARATOR_SELECTION.json").read_text())
    assert sel["comparator"] in ("GRU", "LIN", "RC_only")
    assert all(e["episodes"] == 5 * 3 * 4 for e in sel["effort"].values())       # v8.2: 1 structure x 5 lambda x 3 reps x 4 eps
    assert v["selection_recomputed_from_sealed_table"] is True
    assert len(sel["tuning_table"]) == 4 * 5
    for t in sel["tuning_table"]:
        assert {"mean_E", "mean_D", "mean_J", "feasible", "rep_mean_E", "rep_mean_D", "n_rows"} <= set(t)
    assert sel["fallback_flags"] == final["selection_fallback_flags"]
    assert set(sel["fallback_flags"]) == {"WM_fallback", "comparator_fallback", "arms_fallback"}
    assert sel["lambda_grid"] == [0.5, 0.625, 0.75, 0.875, 1.0]
    assert (out / "PRIMARY_RESULT.json").exists()                  # fallback selections never stop the run (E1.6)
    assert sel["comfort_bound"]["form"] == "absolute" and sel["comfort_bound"]["value"] == 0.3269812989590207
    sec = json.loads((out / "SECONDARY_RESULTS.json").read_text())
    assert set(sec["heldout_mse_12step_ci"]["contrasts"]) == {"WM-GRU", "WM-RC_only", "WM-LIN", "GRU-RC_only",
                                                              "GRU-LIN", "RC_only-LIN"}
    assert "BOOT_INDEX_mse.npy" in order and order.index("BOOT_INDEX_mse.npy") > order.index("PRIMARY_RESULT.json")
