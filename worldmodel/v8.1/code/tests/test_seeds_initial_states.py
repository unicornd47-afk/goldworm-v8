import numpy as np

from v81 import plant_iv as P
from v81 import seeds as SD
from v81 import spec as S


def test_examples_from_plan():
    pos = {s: i for i, s in enumerate(S.TUNING_SEEDS)}
    assert SD.initial_state_index(pos[7000]) == 0
    assert SD.initial_state_index(pos[7001]) == 1
    assert SD.initial_state_index(pos[7003]) == 3
    assert SD.initial_state_index(pos[7004]) == 0
    tp = {s: i for i, s in enumerate(S.TEST_SEEDS)}
    assert SD.initial_state_index(tp[9100]) == 0 and SD.initial_state_index(tp[9139]) == 3
    counts = np.bincount([SD.initial_state_index(i) for i in range(40)])
    assert list(counts) == [10, 10, 10, 10]
    assert S.INITIAL_STATES[0] == (15.0, 14.5, 0.0) and S.INITIAL_STATES[3] == (22.5, 21.0, 0.0)


def test_all_episodes_of_a_seed_share_state_and_draw():
    G = SD.SeedGuard("smoke")
    a = P.Scenario(3, 2, "N0", G, "tuning")
    b = P.Scenario(3, 2, "T1a", G, "tuning")
    G.selection_sealed = True
    c = P.Scenario(3, 2, "H2b", G, "test_heldout")
    for sc in (a, b, c):
        assert (sc.d.T_r0, sc.d.T_w0, sc.d.q_h0) == S.INITIAL_STATES[2]
    assert np.array_equal(a.d.eps_d, c.d.eps_d) and a.d.mu_o == b.d.mu_o == c.d.mu_o
    # warm-up identical (interventions only start at k = 120)
    assert np.array_equal(a.y[:97], c.y[:97])


def test_tuning_entry_and_compositions():
    assert [SD.tuning_entry(i) for i in range(7)] == list(S.TRAIN_IDS) + [S.TRAIN_IDS[0]]
    comp = SD.train_composition()
    assert len(comp) == 32 and sum(1 for c in comp if c[0] == "N0") == 8
    for iid in S.TRAIN_IDS:
        ons = [o for (i, o) in comp if i == iid]
        assert ons == [72, 120, 168, 216]
    v = SD.val_composition()
    assert len(v) == 8 and sum(1 for c in v if c[0] == "N0") == 2 and set(i for i, _ in v[2:]) == set(S.TRAIN_IDS)


def test_episode_ids_unique():
    ids = {SD.episode_id(s, i) for s in list(S.TUNING_SEEDS) + list(S.TEST_SEEDS) for i in SD.SLOT}
    assert len(ids) == 60 * len(SD.SLOT)
