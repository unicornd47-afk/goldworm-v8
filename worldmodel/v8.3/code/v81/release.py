"""v8.3 start rule (PLAN_v8.3 r3 §3/§4, dahill PLAN_REVIEW_v8.3_r3 A1-A3). Launch requires ALL of:
 (1) v8.2 science bundle unchanged (HASHES_v8.2_E1 304bc28d + dahill PLAN_PASS_v8.2) and the v8.3 bundle
     (HASHES_v8.3, plan r3 e2debca5, dahill PLAN_REVIEW_v8.3_r3 with A1/A2/A3 closed in the RELEASE);
 (2) non-author implementation, hashed diff v8.2 -> v8.3; (3) independent CODE_PASS on the v8.3 code manifest;
 (4) SEED_EVIDENCE_v8.3 bound (seed reuse condition, plan §1); (5) sidecar script bound;
 (6) a GATE_PASS probe file and CONFIG_v8.3.json bound, and CONFIG caps/ladder equal to the decision recomputed
     from the probe measurements; (7) the host fingerprint equal to the probe's (A2; checked against the running
     host in run_production). Anything missing -> NOT_LAUNCHED."""
import json
import os

from . import spec as S

CODE_MANIFEST = S.V83["code_manifest"]             # v8.3: CODE_HASHES_v8.3.sha256
DIFF_FILE = os.path.join(S.V83_DIR, S.V83["diff_file"])
CONFIG_FILE = os.path.join(S.V83_DIR, S.V83["config_file"])
PLAN_AUTHORS = ("uni", "uni · KI-Forschung")


def verify_code_manifest(code_dir, manifest_sha):
    path = os.path.join(code_dir, CODE_MANIFEST)
    if not os.path.exists(path) or S.sha256_file(path) != manifest_sha:
        return False, "manifest hash mismatch"
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            h, name = line.split(None, 1)
            name = name.strip()
            if S.sha256_file(os.path.join(code_dir, name)) != h:
                return False, "code file changed: %s" % name
    return True, "ok"


def _sha_ok(path, h):
    return bool(h) and os.path.exists(path) and S.sha256_file(path) == h


def check_probe_config(rel, current_host=None):
    """(6)+(7): probe file + CONFIG_v8.3 bound, GATE_PASS, CONFIG == decision recomputed from the probe, A2 host."""
    from . import probe as PB
    reasons = []
    pr = rel.get("probe") or {}
    ppath = os.path.join(S.V83_DIR, pr.get("file", "")) if pr.get("file") else None
    if not ppath or not _sha_ok(ppath, pr.get("sha256")):
        return ["probe file missing or hash mismatch"], None
    with open(ppath) as f:
        pj = json.load(f)
    if pj.get("outcome") != "GATE_PASS":
        reasons.append("probe outcome is %r, not GATE_PASS" % pj.get("outcome"))
    cf = rel.get("config_v8_3") or {}
    if not _sha_ok(CONFIG_FILE, cf.get("sha256")):
        return reasons + ["CONFIG_v8.3.json missing or hash mismatch"], None
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    if cfg.get("probe", {}).get("sha256") != pr.get("sha256"):
        reasons.append("CONFIG_v8.3 not derived from the bound probe")
    # K2 reference (uni 11:12 (5)): box -> v8.2 toy file; other host -> that host's bound toy attempt
    from .gates import load_k2_reference
    from .hostenv import is_box
    ref = pj.get("k2_reference") or {}
    toy = load_k2_reference(ref) if ref else None
    exp_kind = "box_v8.2" if is_box(pj.get("host_fingerprint") or {}) else "host_toy"
    if toy is None or ref.get("kind") != exp_kind or (rel.get("k2_reference") or {}) != ref:
        reasons.append("K2 reference missing, wrong kind for the probe host (%s), hash mismatch or not bound in RELEASE"
                       % exp_kind)
    elif exp_kind == "box_v8.2" and (ref.get("sha256") != S.V83["toy_preflight_sha256"] or ref.get("toy_fits") != S.K2_TOY_FITS):
        reasons.append("K2 box reference is not v8.2 PREFLIGHT_PROJECTION.json / 336.8")
    else:
        dec = PB.decide(pj.get("measurements", {}), toy, toy_fits=ref["toy_fits"])
        caps = cfg.get("caps_cpu_s", {})
        if dec["outcome"] != "GATE_PASS" or dec["caps"] != caps.get("per_phase") or \
                dec["ladder_applied"] != cfg.get("ladder_applied") or dec["global_cap"] != caps.get("global_cap"):
            reasons.append("CONFIG_v8.3 caps/ladder differ from the decision recomputed from the probe")
        if caps.get("k1_global_cap") != S.K1_GLOBAL_CAP or not (caps.get("global_cap") or 1e18) <= S.K1_GLOBAL_CAP:
            reasons.append("CONFIG_v8.3 global cap > 3600 or K1 cap missing")
    # ERRATUM E1 / dahill reqs (4), (5): pinning and start-rule threshold bound in RELEASE exactly as in the probe
    from .hostenv import pinning_matches
    if pj.get("erratum_sha256") != S.V83["erratum_sha256"] or pj.get("erratum2_sha256") != S.V83["erratum2_sha256"]:
        reasons.append("probe not run under plan r4 (ERRATUM E1 %s + E2 %s)" % (S.V83["erratum_sha256"][:8],
                                                                             S.V83["erratum2_sha256"][:8]))
    if rel.get(S.PROCESS_STARTS_FIELD) != pj.get("production_process_starts") or not pj.get("production_process_starts"):
        reasons.append("E2.1: production process starts in RELEASE differ from the probe's projection input")
    if not pinning_matches(rel.get("pinning"), pj.get("pinning"))[0] or \
            not pinning_matches(cfg.get("pinning"), pj.get("pinning"))[0]:
        reasons.append("E1.2: pinning (core list / core) not bound in RELEASE/CONFIG as recorded by the probe")
    from .hostenv import is_box as _is_box
    box_default = _is_box(pj.get("host_fingerprint") or {}) and pj.get("start_rule_threshold") is None
    if rel.get(S.START_RULE_FIELD) != pj.get("start_rule_threshold") or not (pj.get("start_rule_threshold") or box_default):
        reasons.append("dahill req 5: start-rule threshold in RELEASE absent or changed since the probe")
    elif current_host is not None and not box_default:
        from .gates import start_threshold
        reasons += start_threshold(current_host, field=rel.get(S.START_RULE_FIELD))[3]
    from .hostenv import same_host
    hp = pj.get("host_fingerprint")
    if not same_host(hp, rel.get("host_fingerprint"))[0] or not same_host(hp, cfg.get("host_fingerprint"))[0]:
        reasons.append("A2: RELEASE/CONFIG host fingerprint differs from the probe's")
    if current_host is not None and not same_host(hp, current_host)[0]:
        reasons.append("A2: this host %r differs from the probe host %r" % (
            {k: current_host.get(k) for k in ("hostname", "cpu_model", "cpu_count")},
            {k: (hp or {}).get(k) for k in ("hostname", "cpu_model", "cpu_count")}))
    return reasons, cfg


def check_start(release_path, code_dir=S.CODE_DIR, current_host=None):
    """Returns (ok: bool, reasons: list, release dict or None, CONFIG_v8.3 dict or None)."""
    reasons = []
    if not os.path.exists(release_path):
        return False, ["%s missing" % S.V83["release_file"]], None, None
    with open(release_path) as f:
        rel = json.load(f)
    if rel.get("status") != "FINAL":
        reasons.append("release status is %r, not FINAL" % rel.get("status"))
    # (1) bundles
    try:
        S.verify_plan_hashes()
    except S.SpecError as ex:
        reasons.append("v8.2 plan hashes: %s" % ex)
    if rel.get("plan_bundle_v8_2", {}).get(S.HASHES_FILE) != S.HASHES_FILE_SHA256:
        reasons.append("release does not bind %s" % S.HASHES_FILE)
    if not _sha_ok(S.PLAN_PASS_FILE, S.PLAN_PASS_SHA256) or rel.get("plan_pass_v8_2", {}).get("sha256") != S.PLAN_PASS_SHA256:
        reasons.append("v8.2 plan PASS file missing or hash mismatch")
    else:
        with open(S.PLAN_PASS_FILE) as f:
            ppj = json.load(f)
        if ppj.get("verdict") != "PLAN_PASS" or ppj.get(S.BUNDLE["plan_pass_hashes_key"]) != S.HASHES_FILE_SHA256:
            reasons.append("v8.2 plan PASS not bound to %s" % S.HASHES_FILE)
        if ppj.get("reviewer") != S.BUNDLE["plan_pass_reviewer"]:
            reasons.append("v8.2 plan PASS reviewer is not %s" % S.BUNDLE["plan_pass_reviewer"])
    try:
        S.verify_v83_bundle()
    except S.SpecError as ex:
        reasons.append("v8.3 bundle: %s" % ex)
    pb = rel.get("plan_bundle_v8_3", {})
    if pb.get(S.V83["plan_file"]) != S.V83["plan_sha256"] or pb.get(S.V83_HASHES_FILE) != S.V83["hashes_file_sha256"]:
        reasons.append("release does not bind plan r3 / HASHES_v8.3")
    pr = rel.get("plan_review_v8_3") or {}
    if pr.get("sha256") != S.V83["plan_review_sha256"]:
        reasons.append("release does not bind PLAN_REVIEW_v8.3_r3")
    closed = pr.get("conditions_closed") or {}
    missing = [c for c in S.V83["plan_review_conditions"] if not closed.get(c)]
    if missing:
        reasons.append("plan review conditions not closed in RELEASE: %s" % missing)
    if rel.get("grid_override") != S.GRID_OVERRIDE:
        reasons.append("release grid_override missing or differs from spec.GRID_OVERRIDE")
    # (2) diff + implementer
    if not _sha_ok(DIFF_FILE, (rel.get("diff") or {}).get("sha256")):
        reasons.append("code diff v8.2 -> v8.3 missing or hash mismatch")
    impl = rel.get("implementer", "")
    if not impl or impl.strip().lower().startswith("uni"):
        reasons.append("implementer missing or a plan author")
    # (3) code review + manifest
    cr = dict(rel.get("code_review_pass") or {})
    if cr.get("file") and not os.path.isabs(cr["file"]):           # relative -> WM_AUDIT_ROOT (portable)
        cr["file"] = os.path.join(S.AUDIT_ROOT, cr["file"])
    if not isinstance(cr, dict) or not cr.get("file") or not cr.get("sha256"):
        reasons.append("code-review PASS missing")
    elif cr.get("verdict") not in S.CODE_PASS_VERDICTS_PRODUCTION:
        reasons.append("code-review verdict %r: production requires an unconditional CODE_PASS" % cr.get("verdict"))
    elif not _sha_ok(cr["file"], cr["sha256"]):
        reasons.append("code-review PASS file hash mismatch")
    else:
        with open(cr["file"]) as f:
            crj = json.load(f)
        if rel.get("code_hashes", {}).get("manifest_sha256") not in [crj.get(k) for k in S.CODE_PASS_MANIFEST_KEYS]:
            reasons.append("code-review PASS not bound to the frozen code manifest")
        if crj.get("verdict") not in S.CODE_PASS_VERDICTS_PRODUCTION:
            reasons.append("code-review file verdict %r: production requires an unconditional CODE_PASS"
                           % crj.get("verdict"))
        if crj.get("reviewer", "").lower().startswith(("uni", "tmbot")):
            reasons.append("code reviewer is not independent")
    ok, why = verify_code_manifest(code_dir, rel.get("code_hashes", {}).get("manifest_sha256", ""))
    if not ok:
        reasons.append("code hashes: %s" % why)
    # (4) seed evidence, (5) sidecar
    se = rel.get("seed_evidence") or {}
    if se.get("sha256") != S.V83["seed_evidence_sha256"] or \
            not _sha_ok(os.path.join(S.V83_DIR, S.V83["seed_evidence_file"]), S.V83["seed_evidence_sha256"]):
        reasons.append("SEED_EVIDENCE_v8.3 not bound or hash mismatch")
    sc = rel.get("sidecar") or {}
    if not _sha_ok(os.path.join(S.V83_DIR, S.V83["sidecar_file"]), sc.get("sha256")):
        reasons.append("sidecar script not bound or hash mismatch")
    # (6) + (7)
    r67, cfg = check_probe_config(rel, current_host)
    reasons += r67
    for name, h in rel.get("bound_files", {}).items():          # relative names resolve against WM_ROOT (portable)
        if not _sha_ok(name if os.path.isabs(name) else os.path.join(S.WM_ROOT, name), h):
            reasons.append("bound file mismatch: %s" % name)
    return (not reasons), reasons, rel, cfg
