"""Start rule (PLAN Status, fix 1). Launch requires ALL of:
 (1) dahill plan PASS on the HASHES_r3 hashes, (2) non-author implementation, (3) independent code-review PASS
 on the frozen code hashes, (4) RELEASE_v8.1.json binding HASHES_r3, code hashes, both PASS files and the
 measured preflight projection with every gate passing. Anything missing -> NOT_LAUNCHED."""
import json
import os

from . import spec as S

CODE_MANIFEST = "CODE_HASHES_v8.1.sha256"
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
        return False, ["RELEASE_v8.1.json missing"], None
    with open(release_path) as f:
        rel = json.load(f)
    if rel.get("status") != "FINAL":
        reasons.append("release status is %r, not FINAL" % rel.get("status"))
    try:
        S.verify_plan_hashes()
    except S.SpecError as ex:
        reasons.append("plan hashes: %s" % ex)
    if rel.get("plan_bundle", {}).get("HASHES_r3.sha256") != S.HASHES_FILE_SHA256:
        reasons.append("release does not bind HASHES_r3")
    pp = rel.get("plan_pass", {})
    if not (os.path.exists(S.PLAN_PASS_FILE) and S.sha256_file(S.PLAN_PASS_FILE) == S.PLAN_PASS_SHA256
            and pp.get("sha256") == S.PLAN_PASS_SHA256):
        reasons.append("plan PASS file missing or hash mismatch")
    else:
        with open(S.PLAN_PASS_FILE) as f:
            ppj = json.load(f)
        if ppj.get("verdict") != "PLAN_PASS" or ppj.get("hashes_r3_sha256") != S.HASHES_FILE_SHA256:
            reasons.append("plan PASS not bound to HASHES_r3")
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
    for name, h in rel.get("bound_files", {}).items():
        if not os.path.exists(name) or S.sha256_file(name) != h:
            reasons.append("bound file mismatch: %s" % name)
    return (not reasons), reasons, rel
