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
    assert all(e["episodes"] == 3 * 3 * 4 for e in sel["effort"].values())
