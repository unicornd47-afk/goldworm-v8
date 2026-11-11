"""Start rule (PLAN_v8.2 Status). Launch requires ALL of:
 (1) dahill plan PASS on the HASHES_v8.2 hashes, (2) non-author implementation reusing the v8.1 frozen code with a
 hashed minimal diff, (3) independent code-review PASS on the frozen v8.2 code hashes, (4) RELEASE_v8.2.json
 binding HASHES_v8.2, code hashes, the diff hash, both PASS files and the measured preflight projection with every
 gate passing. Anything missing -> NOT_LAUNCHED."""
import json
import os

from . import spec as S

CODE_MANIFEST = S.BUNDLE["code_manifest"]          # v8.2: CODE_HASHES_v8.2.sha256
DIFF_FILE = os.path.join(S.PLAN_DIR, "DIFF_v8.1_to_v8.2.patch")
PREFLIGHT_FILE = os.path.join(S.PLAN_DIR, "PREFLIGHT_PROJECTION.json")
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


def check_start(release_path, code_dir=S.CODE_DIR):
    """Returns (ok: bool, reasons: list, release dict or None)."""
    reasons = []
    if not os.path.exists(release_path):
        return False, ["%s missing" % S.BUNDLE["release_file"]], None
    with open(release_path) as f:
        rel = json.load(f)
    if rel.get("status") != "FINAL":
        reasons.append("release status is %r, not FINAL" % rel.get("status"))
    try:
        S.verify_plan_hashes()
    except S.SpecError as ex:
        reasons.append("plan hashes: %s" % ex)
    if rel.get("plan_bundle", {}).get(S.HASHES_FILE) != S.HASHES_FILE_SHA256:
        reasons.append("release does not bind %s" % S.HASHES_FILE)
    pp = rel.get("plan_pass", {})
    if S.PLAN_PASS_SHA256 == S.TODO_AFTER_PASS:
        reasons.append("plan PASS sha256 is a TODO_AFTER_PASS placeholder in spec.BUNDLE")
    elif not (os.path.exists(S.PLAN_PASS_FILE) and S.sha256_file(S.PLAN_PASS_FILE) == S.PLAN_PASS_SHA256
              and pp.get("sha256") == S.PLAN_PASS_SHA256):
        reasons.append("plan PASS file missing or hash mismatch")
    else:
        with open(S.PLAN_PASS_FILE) as f:
            ppj = json.load(f)
        if ppj.get("verdict") != "PLAN_PASS" or ppj.get(S.BUNDLE["plan_pass_hashes_key"]) != S.HASHES_FILE_SHA256:
            reasons.append("plan PASS not bound to %s" % S.HASHES_FILE)
        if ppj.get("reviewer") != S.BUNDLE["plan_pass_reviewer"]:
            reasons.append("plan PASS reviewer is not %s" % S.BUNDLE["plan_pass_reviewer"])
    # v8.2: the erratum override of CONFIG numbers must be shown verbatim in the RELEASE
    if rel.get("grid_override") != S.GRID_OVERRIDE:
        reasons.append("release grid_override missing or differs from spec.GRID_OVERRIDE")
    # v8.2 start rule (2): the v8.1 -> v8.2 diff itself is hashed and bound
    dh = (rel.get("diff") or {}).get("sha256")
    if not dh or not os.path.exists(DIFF_FILE) or S.sha256_file(DIFF_FILE) != dh:
        reasons.append("code diff vs v8.1 missing or hash mismatch")
    impl = rel.get("implementer", "")
    if not impl or impl.strip().lower().startswith("uni"):
        reasons.append("implementer missing or a plan author")
    cr = rel.get("code_review_pass") or {}
    if not isinstance(cr, dict) or cr.get("verdict") != "CODE_PASS" or not cr.get("file") or not cr.get("sha256"):
        reasons.append("code-review PASS missing")
    else:
        if not os.path.exists(cr["file"]) or S.sha256_file(cr["file"]) != cr["sha256"]:
            reasons.append("code-review PASS file hash mismatch")
        else:
            with open(cr["file"]) as f:
                crj = json.load(f)
            if crj.get("code_manifest_sha256") != rel.get("code_hashes", {}).get("manifest_sha256"):
                reasons.append("code-review PASS not bound to the frozen code manifest")
            if crj.get("reviewer", "").lower().startswith(("uni", "tmbot")):
                reasons.append("code reviewer is not independent")
    ch = rel.get("code_hashes", {})
    ok, why = verify_code_manifest(code_dir, ch.get("manifest_sha256", ""))
    if not ok:
        reasons.append("code hashes: %s" % why)
    pf = rel.get("preflight", {})
    if not pf.get("all_gates_pass"):
        reasons.append("preflight gates not passing")
    # v8.2 (dahill v8.1 CODE_PASS condition 1): the measured preflight file must be bound and must itself pass
    bf = {os.path.normpath(k): v for k, v in rel.get("bound_files", {}).items()}
    if bf.get(os.path.normpath(PREFLIGHT_FILE)) is None:
        reasons.append("PREFLIGHT_PROJECTION.json not in bound_files")
    elif os.path.exists(PREFLIGHT_FILE):
        with open(PREFLIGHT_FILE) as f:
            pfj = json.load(f)
        if not pfj.get("all_gates_pass"):
            reasons.append("PREFLIGHT_PROJECTION.json all_gates_pass is not true")
        # W1 §W1.3(4) plausibility, recomputed from the unit costs (not trusted from the flag)
        u = pfj.get("measurements", {}).get("unit", {})
        a = u.get("WM_32", {}).get("per_update_cpu_s")
        b = u.get("WM_64", {}).get("per_update_cpu_s")
        if a is None or b is None or not a <= b:
            reasons.append("preflight plausibility: WM_32 per-update cost > WM_64 (W1)")
        # W1 §W1.3(3): ladder only (a)
        if any(st not in S.W1_ALLOWED_LADDER for st in pfj.get("ladder_applied", [])):
            reasons.append("preflight ladder step outside W1 (only 'a' allowed)")
        if pfj.get("impl_errata") != S.BUNDLE["impl_errata"]:
            reasons.append("preflight not run under the bound implementation errata (W1)")
    # W1: implementation errata and the preserved failed first attempt are bound by sha
    for name, h in S.BUNDLE["impl_errata"].items():
        fp = os.path.join(S.PLAN_DIR, name)
        if not os.path.exists(fp) or S.sha256_file(fp) != h or (rel.get("impl_errata") or {}).get(name) != h:
            reasons.append("implementation erratum missing or hash mismatch: %s" % name)
    a1 = S.BUNDLE["preflight_attempt1"]
    a1p = os.path.join(S.PLAN_DIR, a1["file"])
    if not os.path.exists(a1p) or S.sha256_file(a1p) != a1["sha256"] or \
            (rel.get("preflight_attempts") or [{}])[0].get("sha256") != a1["sha256"]:
        reasons.append("failed preflight attempt 1 not preserved/listed")
    for name, h in rel.get("bound_files", {}).items():
        if not os.path.exists(name) or S.sha256_file(name) != h:
            reasons.append("bound file mismatch: %s" % name)
    return (not reasons), reasons, rel
