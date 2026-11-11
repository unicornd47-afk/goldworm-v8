"""Stage-0 static checks: model modules import no plant/intervention/scorer code; HELDOUT-id scan."""
import ast
import os

from . import spec as S

MODEL_MODULES = ("features.py", "models_torch.py", "model_lin.py", "predictors.py")
FORBIDDEN = ("plant_iv", "closedloop", "stats", "seeds", "spec", "pipeline", "thermal5", "plant", "oracle",
             "scoring", "interventions")


def imports_of(path):
    tree = ast.parse(open(path).read())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names += [mod] + ["%s.%s" % (mod, a.name) for a in node.names]
    return names


def model_import_check(pkg_dir=None):
    pkg_dir = pkg_dir or os.path.dirname(os.path.abspath(__file__))
    bad = {}
    for m in MODEL_MODULES:
        for name in imports_of(os.path.join(pkg_dir, m)):
            parts = name.replace(".", " ").split()
            if any(p in FORBIDDEN for p in parts):
                bad.setdefault(m, []).append(name)
    return dict(valid=not bad, violations=bad)


def heldout_scan(records):
    """records: iterable of dicts (training/tuning/selection logs). Any HELDOUT id anywhere -> invalid."""
    hits = []
    for i, r in enumerate(records):
        s = repr(r)
        for hid in S.HELDOUT_IDS:
            if ("'%s'" % hid) in s:
                hits.append((i, hid))
    return dict(valid=not hits, hits=hits[:20], n_records=len(records) if hasattr(records, "__len__") else None)
