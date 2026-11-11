import os
import sys

sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture(autouse=True)
def _hermetic_release_draft(tmp_path, monkeypatch):
    """r3: tests never read the real v8.3/RELEASE_v8.3_DRAFT.json (its fields change between rounds).
    Default draft = an absent file in tmp; tests needing a draft pass draft_path explicitly."""
    from v81 import spec as S
    monkeypatch.setattr(S, "RELEASE_DRAFT_FILE", str(tmp_path / "RELEASE_v8.3_DRAFT_absent.json"))
