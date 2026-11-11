"""Append-only ledger (JSONL) with sha256 of sealed artefacts (COMPARATOR_SELECTION before test data)."""
import hashlib
import json
import os
import time


def canon(obj):
    return json.dumps(obj, sort_keys=True, indent=1, default=_default)


def _default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


class Ledger:
    def __init__(self, out_dir):
        self.path = os.path.join(out_dir, "LEDGER.jsonl")
        self.out = out_dir

    def append(self, event, **kw):
        rec = dict(event=event, time_unix=time.time(), **kw)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, sort_keys=True, default=_default) + "\n")
        return rec

    def write_sealed(self, name, obj):
        path = os.path.join(self.out, name)
        if os.path.exists(path):
            raise FileExistsError(path)
        data = canon(obj).encode()
        with open(path, "wb") as f:
            f.write(data)
        h = hashlib.sha256(data).hexdigest()
        self.append("sealed", file=name, sha256=h)
        return h

    def events(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(l) for l in f if l.strip()]
