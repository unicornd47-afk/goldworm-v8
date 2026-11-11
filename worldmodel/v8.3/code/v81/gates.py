"""v8.3 shared start gates and append-only attempt ledgers for the probe and the host toy preflight
(PLAN r3 C1/C4/C5; dahill A1/A2; uni 11:12 interpretations 5 and 9)."""
import hashlib
import json
import os
import time

from . import hostenv as HE
from . import spec as S

PROBE_LEDGER = "PROBE_LEDGER.jsonl"
TOY_LEDGER = "TOY_LEDGER.jsonl"


def append(path, event, **kw):
    rec = dict(event=event, time_unix=time.time(), **kw)
    with open(path, "a") as f:
        f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
    return rec


def events(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def count_attempts(path):
    return sum(1 for e in events(path) if e["event"] == "attempt_start")


def manifest_sha():
    man = os.path.join(S.CODE_DIR, S.V83["code_manifest"])
    return S.sha256_file(man) if os.path.exists(man) else None


def manifest_ok():
    from .release import verify_code_manifest
    h = manifest_sha()
    return bool(h) and verify_code_manifest(S.CODE_DIR, h)[0], h


def find_code_pass(msha, cp_dir=None):
    """uni 11:12 (9) / dahill: the dahill CODE_PASS bound to manifest `msha`. Returns (path, sha256) or (None, reasons)."""
    import glob
    cp_dir = cp_dir or S.CODE_PASS_DIR
    seen = []
    for path in sorted(glob.glob(os.path.join(cp_dir, S.CODE_PASS_GLOB))):
        try:
            with open(path) as fh:
                j = json.load(fh)
        except (OSError, ValueError):
            seen.append("%s unreadable" % os.path.basename(path))
            continue
        mk = [j.get(k) for k in S.CODE_PASS_MANIFEST_KEYS if j.get(k)]
        if (j.get("reviewer") == "dahill" and j.get("verdict") in S.CODE_PASS_VERDICTS and msha in mk
                and j.get("plan_sha256", S.V83["plan_sha256"]) == S.V83["plan_sha256"]):
            return path, S.sha256_file(path)
        seen.append("%s: reviewer=%r verdict=%r manifest=%r" % (os.path.basename(path), j.get("reviewer"),
                                                                j.get("verdict"), mk))
    return None, seen


def code_pass_reasons(msha, cp_dir=None):
    """Fail closed: no current manifest, or no dahill CODE_PASS file for exactly this manifest sha."""
    if not msha:
        return ["CODE_PASS: no current manifest"]
    path, info = find_code_pass(msha, cp_dir)
    if path:
        return []
    return ["no dahill CODE_PASS for the current manifest %s in %s (seen: %s)" % (
        msha, cp_dir or S.CODE_PASS_DIR, "; ".join(info) or "none")]


def common_gate_reasons(threads, start, code_pass_dir=None):
    """C1 readback, C4 start rule, frozen manifest, dahill CODE_PASS (9), sidecar fields readable (stricter S5)."""
    reasons = []
    ok_t, bad_t = HE.thread_readback_ok(threads)
    if not ok_t:
        reasons.append("C1 thread readback mismatch: %s" % bad_t)
    if not start["ok"]:
        reasons += ["C4 start rule: %s" % r for r in start["reasons"]]
    ok_m, msha = manifest_ok()
    if not ok_m:                       # a stale manifest must never be matched against an old CODE_PASS
        reasons.append("code manifest %s missing or not verifying" % S.V83["code_manifest"])
        reasons.append("CODE_PASS not checkable: manifest not verifying")
    else:
        reasons += code_pass_reasons(msha, code_pass_dir)
    ok_s, bad_s = HE.sidecar_capable()
    if not ok_s:
        reasons.append("C5 sidecar fields unreadable on this host: %s" % bad_s)
    return reasons


def k2_reference(host, probe_dir, pinning=None):
    """uni 11:12 (5): K2 toy reference for THIS host.
    box -> the bound v8.2 file (7e164362), toy fits = 336.8 (plan literal).
    non-box -> the LATEST attempt in TOY_LEDGER.jsonl with outcome TOY_OK whose recorded host equals this host and whose
    file still has the logged sha; toy fits = that file's projection_cpu_s.fits (computed exactly like 336.8).
    Returns (ref dict, reasons)."""
    if HE.is_box(host):
        p = os.path.join(S.PLAN_DIR, S.V83["toy_preflight_file"])
        if not os.path.exists(p) or S.sha256_file(p) != S.V83["toy_preflight_sha256"]:
            return None, ["K2: v8.2 toy preflight missing or hash mismatch"]
        return dict(kind="box_v8.2", file=os.path.relpath(p, S.WM_ROOT), sha256=S.V83["toy_preflight_sha256"],
                    toy_fits=S.K2_TOY_FITS, attempt=None, host_assumption=S.BOX_K2_HOST_ASSUMPTION), []
    ref = latest_host_toy(host, probe_dir, pinning)
    if ref:
        return ref, []
    return None, ["K2: non-box host %r has no valid same-host toy preflight attempt (run run_preflight.py first)"
                  % host.get("hostname")]


def load_k2_reference(ref):
    """Reads the reference file (sha re-checked). Returns the toy dict or None."""
    if not ref:
        return None
    p = os.path.join(S.WM_ROOT, ref["file"])
    if not os.path.exists(p) or S.sha256_file(p) != ref["sha256"]:
        return None
    with open(p) as f:
        return json.load(f)


def latest_host_toy(host, probe_dir, pinning=None):
    """The LATEST TOY_OK attempt in TOY_LEDGER.jsonl recorded on this same host whose file still has the logged sha,
    within its 120 s cap, with a fits projection. ERRATUM E2.5: if `pinning` is pinned, only an attempt pinned
    identically (same P-core list and core) counts. Returns a reference dict or None."""
    pinned = bool(pinning and pinning.get("pinned"))
    ok = [e for e in events(os.path.join(probe_dir, TOY_LEDGER))
          if e["event"] == "attempt_end" and e.get("outcome") == "TOY_OK" and HE.same_host(e.get("host"), host)[0]
          and (not pinned or HE.pinning_matches(e.get("pinning"), pinning)[0])]
    for e in reversed(ok):
        p = os.path.join(probe_dir, e["file"])
        if os.path.exists(p) and S.sha256_file(p) == e["sha256"]:
            with open(p) as f:
                j = json.load(f)
            if j.get("preflight_within_cap") and j.get("projection_cpu_s", {}).get("fits"):
                return dict(kind="host_toy", file=os.path.relpath(p, S.WM_ROOT), sha256=e["sha256"],
                            toy_fits=float(j["projection_cpu_s"]["fits"]), attempt=e["attempt"],
                            pinning=e.get("pinning"))
    return None


def k2_fresh(host, probe_dir, fits_total):
    """r3 item 1 (descriptive only, no gate change): on the BOX, if a same-host fresh toy attempt (TOY_OK) exists,
    R_fresh = P_fits / fresh toy fits and drift_flag = |fresh_fits / 336.8 - 1| > 0.15. Gate R stays P_fits / 336.8.
    Non-box hosts: None (behaviour unchanged; their gate R already uses the same-host toy)."""
    if not HE.is_box(host):
        return None
    fresh = latest_host_toy(host, probe_dir)
    if not fresh:
        return dict(fresh_toy=None, R_fresh=None, drift_flag=None, reference_toy_fits=S.K2_TOY_FITS,
                    note="no same-host fresh toy attempt on the box")
    ff = fresh["toy_fits"]
    drift = abs(ff / S.K2_TOY_FITS - 1.0)
    return dict(fresh_toy=fresh, fresh_toy_fits=ff, reference_toy_fits=S.K2_TOY_FITS, drift_rel=drift,
                drift_threshold=S.K2_DRIFT_FRAC, drift_flag=bool(drift > S.K2_DRIFT_FRAC),
                R_fresh=(fits_total / ff) if fits_total is not None else None,
                R_gate=(fits_total / S.K2_TOY_FITS) if fits_total is not None else None,
                note="descriptive only; the box gate R uses 336.8")


def e1_5_blocked(led_path):
    """ERRATUM E1.5: a linearity PROBE_DIAGNOSIS in an attempt run under the erratum (E1.1 warm-up + E1.2 pinning)
    is the 'second' linearity failure (attempt 2 on unicorn was the first): no further probe, v8.3 NOT_LAUNCHED."""
    for e in events(led_path):
        if e["event"] == "attempt_end" and e.get("outcome") == "PROBE_DIAGNOSIS" and \
                e.get("diagnosis_kind") == "linearity" and e.get("erratum") == S.ERRATUM:
            return ["ERRATUM E1.5: attempt %s failed linearity under E1 -> no further probe; v8.3 ends NOT_LAUNCHED"
                    % e.get("attempt")]
    return []


def release_draft_path():
    return os.path.join(S.V83_DIR, S.RELEASE_DRAFT_FILE)


BASELINE_LEDGER = "BASELINE_LEDGER.jsonl"


def baseline_summary(samples, duration_s, interval_s):
    """E2.4: p95 (numpy default linear method) of the load1 samples; threshold = 1.5 x p95."""
    import numpy as np
    l1 = [x["load1"] for x in samples if x.get("load1") is not None]
    p95 = float(np.percentile(l1, S.BASELINE_PERCENTILE)) if l1 else None
    return dict(n_samples=len(l1), load1_p95=p95, load1_max=max(l1) if l1 else None, load1_min=min(l1) if l1 else None,
                factor=S.BASELINE_FACTOR, threshold_load1_max_exclusive=(S.BASELINE_FACTOR * p95) if p95 is not None else None,
                duration_s=duration_s, interval_s=interval_s)


def threshold_field_from_baseline(path):
    """The RELEASE field (dahill req 5 / E2.4) derived from a BASELINE_OK attempt file. Raises ValueError otherwise."""
    with open(path) as f:
        b = json.load(f)
    if b.get("outcome") != "BASELINE_OK":
        raise ValueError("baseline outcome %r" % b.get("outcome"))
    return dict(rule="ERRATUM E2.4: load1 < 1.5 x p95 of a 10-minute no-experiment load1 baseline (fixed before the probe)",
                host={k: b["host"].get(k) for k in HE.FINGERPRINT_KEYS},
                baseline=dict(file=os.path.relpath(path, S.WM_ROOT), sha256=S.sha256_file(path), attempt=b["attempt"]),
                load1_p95=b["summary"]["load1_p95"], factor=S.BASELINE_FACTOR,
                load1_max_exclusive=b["summary"]["threshold_load1_max_exclusive"])


def _draft_field(draft_path, key):
    p = draft_path or release_draft_path()
    if not os.path.exists(p):
        return None, ["%s missing" % os.path.basename(p)]
    with open(p) as f:
        return json.load(f).get(key), []


def start_threshold(host, draft_path=None, field=None):
    """dahill req (5) + ERRATUM E2.4: returns (load1_max, field, field_sha, reasons).
    Field present -> must be for this host, bound to an intact BASELINE_OK file whose recomputed 1.5 x p95 equals it.
    Field absent -> box: the old < 1.0 (E2.4); any other host: refused (threshold must be fixed in RELEASE first)."""
    from .ledger import canon
    r = []
    if field is None:
        field, r = _draft_field(draft_path, S.START_RULE_FIELD)
    if not field:
        if HE.is_box(host):
            return S.START_LOAD1_MAX, None, None, []
        return None, None, None, r + ["start-rule threshold: RELEASE field %r absent for non-box host %r (run "
                                      "run_baseline.py and fix it in the RELEASE draft first)"
                                      % (S.START_RULE_FIELD, host.get("hostname"))]
    sha = hashlib.sha256(canon(field).encode()).hexdigest()
    if not HE.same_host(field.get("host"), host)[0]:
        r.append("start-rule threshold fixed for host %r, not this host %r" % ((field.get("host") or {}).get("hostname"),
                                                                            host.get("hostname")))
    bl = field.get("baseline") or {}
    bp = os.path.join(S.WM_ROOT, bl.get("file") or "\0")
    if not bl.get("file") or not os.path.exists(bp) or S.sha256_file(bp) != bl.get("sha256"):
        r.append("start-rule threshold: baseline file missing or hash mismatch")
    else:
        with open(bp) as f:
            b = json.load(f)
        summ = baseline_summary(b.get("samples", []), b.get("duration_s"), b.get("interval_s"))
        if b.get("outcome") != "BASELINE_OK" or not HE.same_host(b.get("host"), host)[0]:
            r.append("start-rule threshold: baseline not BASELINE_OK on this host")
        elif (summ["load1_p95"] != field.get("load1_p95") or field.get("factor") != S.BASELINE_FACTOR
              or summ["threshold_load1_max_exclusive"] != field.get("load1_max_exclusive")):
            r.append("start-rule threshold differs from 1.5 x p95 recomputed from the baseline")
    if r:
        return None, field, sha, r
    return float(field["load1_max_exclusive"]), field, sha, []


def process_starts(draft_path=None, value=None):
    """E2.1: number of production process starts, fixed in RELEASE (draft). Returns (n, reasons)."""
    r = []
    if value is None:
        value, r = _draft_field(draft_path, S.PROCESS_STARTS_FIELD)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        return None, r + ["E2.1: RELEASE field %r (production process starts) absent or invalid" % S.PROCESS_STARTS_FIELD]
    return value, []


def sidecar_summary(path, t0_wall, t1_wall):
    """E2.3: max load1, min CPU MHz (mean over CPUs, as sampled) and steal over [t0_wall, t1_wall] (timed segments)."""
    rows = []
    if path and os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    j = json.loads(line)
                except ValueError:
                    continue
                if j.get("event") == "sample" and t0_wall <= j.get("wall_unix", -1) <= t1_wall:
                    rows.append(j)
    l1 = [x["load_1_5_15"][0] for x in rows if x.get("load_1_5_15")]
    mhz = [x["cpu_mhz_mean"] for x in rows if x.get("cpu_mhz_mean") is not None]
    st = [x["steal_frac_since_prev"] for x in rows if x.get("steal_frac_since_prev") is not None]
    return dict(n_samples=len(rows), window_wall_unix=[t0_wall, t1_wall], load1_max=max(l1) if l1 else None,
                cpu_mhz_mean_min=min(mhz) if mhz else None, steal_frac_max=max(st) if st else None,
                steal_ticks_delta=(rows[-1]["steal_ticks"] - rows[0]["steal_ticks"]) if len(rows) > 1 else None)


def write_sha256sums(dirpath, name):
    """Checksum list of every file under dirpath EXCEPT the list itself (r3 item 4 / dahill req 7). Exclusive create."""
    out = os.path.join(dirpath, name)
    rows = []
    for root, _, files in os.walk(dirpath):
        for fn in files:
            p = os.path.join(root, fn)
            if os.path.abspath(p) == os.path.abspath(out):
                continue
            rows.append((os.path.relpath(p, dirpath), S.sha256_file(p)))
    with open(out, "x") as f:
        for rel, h in sorted(rows):
            f.write("%s  ./%s\n" % (h, rel))
    return out
