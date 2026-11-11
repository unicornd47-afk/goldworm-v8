import pytest

from v81 import plant_iv as P
from v81 import seeds as SD
from v81 import spec as S
from v81 import stage0


def test_compositions_never_contain_heldout():
    ids = [i for i, _ in SD.train_composition()] + [i for i, _ in SD.val_composition()]
    ids += [SD.tuning_entry(i) for i in range(20)] + ["N0"]
    assert not set(ids) & set(S.HELDOUT_IDS)
    assert set(S.TRAIN_IDS).isdisjoint(S.HELDOUT_IDS)
    tr_types = {S.IV_BY_ID[i]["type"] for i in S.TRAIN_IDS}
    ho_types = {S.IV_BY_ID[i]["type"] for i in S.HELDOUT_IDS}
    assert tr_types.isdisjoint(ho_types)


def test_heldout_scan_detects():
    recs = [dict(iv_id="T1a"), dict(iv_id="N0")]
    assert stage0.heldout_scan(recs)["valid"]
    recs.append(dict(iv_id="H3a"))
    assert not stage0.heldout_scan(recs)["valid"]


def test_guard_blocks_test_before_seal_and_adaptation_before_primary():
    G = SD.SeedGuard("production")
    with pytest.raises(SD.SeedGuardError):
        G.check(9500, "test_heldout")
    with pytest.raises(SD.SeedGuardError):
        G.check(9500, "tuning")
    with pytest.raises(SD.SeedGuardError):
        G.check(7300, "train")                      # tuning seed is not a training seed
    G.check(6100, "train")
    G.check(7300, "tuning")
    G.selection_sealed = True
    G.check(9500, "test_heldout")
    with pytest.raises(SD.SeedGuardError):
        G.check(4100, "adaptation_data")
    G.primary_sealed = True
    G.check(4100, "adaptation_data")
    G.check(9509, "adapt_eval")
    with pytest.raises(SD.SeedGuardError):
        G.check(9510, "adapt_eval")                 # eval only on 9500-9509
    with pytest.raises(SD.SeedGuardError):
        G.check(5000, "train")                      # burned


def test_preflight_guard_only_seeds_1_to_4_and_no_test():
    G = SD.SeedGuard("preflight")
    G.check(1, "preflight_toy")
    for s in (9500, 7300, 6100, 4100, 9100, 7000, 8000):
        with pytest.raises(SD.SeedGuardError):
            G.check(s, "preflight_toy")
    G.selection_sealed = G.primary_sealed = True
    with pytest.raises(SD.SeedGuardError):
        G.check(1, "test_heldout")


def test_training_data_runtime_has_no_heldout():
    G = SD.SeedGuard("smoke")
    comp = SD.train_composition()
    metas = [P.data_episode(1, i, comp[i][0], comp[i][1], G, "train", N=60)[1] for i in (0, 8, 31)]
    assert stage0.heldout_scan(metas)["valid"]


def test_model_modules_do_not_import_plant():
    r = stage0.model_import_check()
    assert r["valid"], r
