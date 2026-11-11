"""ERRATUM_v8.3_E2.4: 10-minute load1 baseline with no experiment running, its own numbered attempt
(probe/BASELINE_LEDGER.jsonl, BASELINE_attempt<N>.json, exclusive create, never overwritten), BEFORE the toy attempt.
Threshold = 1.5 x p95 (numpy default percentile) of the load1 samples. Any excluded process seen during the window
(exp001 / run_arm.py, and - stricter - any v8.3 run_* entry point other than this one) -> BASELINE_INVALID.
The threshold is then fixed in the RELEASE draft with set_start_threshold.py (refuses if already set)."""
import json
import os
import time

from . import gates as GT
from . import hostenv as HE
from . import spec as S
from .ledger import canon, write_once

EXTRA_EXCLUDE = ("run_probe.py", "run_production.py", "run_preflight.py")


def run_baseline(probe_dir=None, duration_s=S.BASELINE_DURATION_S, interval_s=S.BASELINE_INTERVAL_S,
                 load_fn=HE.load1, procs_fn=HE.process_cmdlines, sleep=time.sleep, wall=time.time):
    probe_dir = probe_dir or os.path.join(S.V83_DIR, S.V83["probe_dir"])
    os.makedirs(probe_dir, exist_ok=True)
    led = os.path.join(probe_dir, GT.BASELINE_LEDGER)
    n = GT.count_attempts(led) + 1
    out = os.path.join(probe_dir, "BASELINE_attempt%d.json" % n)
    if os.path.exists(out):
        raise SystemExit("%s exists; refusing to overwrite" % out)
    host = HE.host_fingerprint()
    pats = tuple(S.START_EXCLUDE_PATTERNS) + EXTRA_EXCLUDE
    me = {os.getpid(), os.getppid()}
    GT.append(led, "attempt_start", attempt=n, host=host, duration_s=duration_s, interval_s=interval_s,
              exclude_patterns=list(pats), erratum=S.ERRATUM)
    samples, hits_all = [], []
    t0 = wall()
    k = 0
    while True:
        t = wall()
        if t - t0 > duration_s + 1e-9:
            break
        procs = procs_fn()
        hits = None if procs is None else [[p, c[:200]] for p, c in procs if p not in me and any(x in c for x in pats)]
        samples.append(dict(seq=k, wall_unix=t, load1=load_fn(), excluded_hits=hits))
        if hits:
            hits_all += hits
        k += 1
        sleep(interval_s)
    summ = GT.baseline_summary(samples, duration_s, interval_s)
    expected = int(duration_s / interval_s)
    reasons = []
    if hits_all:
        reasons.append("experiment / excluded process running during the baseline: %s" % hits_all[:3])
    if any(s["excluded_hits"] is None or s["load1"] is None for s in samples):
        reasons.append("load or process list unreadable during the baseline")
    if summ["n_samples"] < 0.95 * expected or (samples and samples[-1]["wall_unix"] - t0) < 0.95 * duration_s:
        reasons.append("baseline too short: %d samples / %.0f s" % (summ["n_samples"], samples[-1]["wall_unix"] - t0
                                                                    if samples else 0.0))
    outcome = "BASELINE_INVALID" if reasons else "BASELINE_OK"
    res = dict(kind="v8.3 E2.4 load1 baseline", attempt=n, host=host, outcome=outcome, reasons=reasons,
               duration_s=duration_s, interval_s=interval_s, exclude_patterns=list(pats), samples=samples,
               summary=summ, erratum=S.ERRATUM, erratum2_sha256=S.V83["erratum2_sha256"],
               time_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    write_once(out, canon(res))
    GT.append(led, "attempt_end", attempt=n, outcome=outcome, file=os.path.basename(out), sha256=S.sha256_file(out),
              host=host, summary=summ, reasons=reasons)
    print(json.dumps(dict(outcome=outcome, reasons=reasons, summary=summ, file=out), indent=1))
    return res


def set_start_threshold(baseline_path, draft_path=None):
    """Writes the dahill-req-5 field into the RELEASE draft ONCE (E2.4 'no later adjustment'): refuses if present."""
    draft_path = draft_path or GT.release_draft_path()
    with open(draft_path) as f:
        d = json.load(f)
    if d.get(S.START_RULE_FIELD):
        raise SystemExit("start-rule threshold already fixed in %s; no later adjustment (E2.4)" % draft_path)
    field = GT.threshold_field_from_baseline(baseline_path)
    d[S.START_RULE_FIELD] = field
    mode = os.stat(draft_path).st_mode & 0o777
    os.chmod(draft_path, mode | 0o200)
    tmp = draft_path + ".tmp"
    with open(tmp, "x") as f:
        f.write(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, draft_path)
    os.chmod(draft_path, mode)
    return field
