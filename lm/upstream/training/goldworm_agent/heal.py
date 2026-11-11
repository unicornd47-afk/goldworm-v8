"""Deterministic rule-based repairs (CATALYST-01 'heal' pattern, no ML). Each rule: detect -> patch.
Applied to a candidate body BEFORE resampling. Returns (new_body, [rule names applied])."""
import ast, re
def _ok(prompt, body):
    try: ast.parse(prompt + body); return True
    except SyntaxError: return False
def fix_indent(prompt, body):
    lines = body.splitlines(); out = []
    for l in lines:
        if l.strip() and not l.startswith((" ", "\t")): l = "    " + l
        out.append(l.replace("\t", "    "))
    return "\n".join(out) + "\n"
def close_brackets(prompt, body):
    ls = body.rstrip("\n").splitlines(); res = []
    for l in ls:
        st = []
        for ch in re.sub(r"(\"[^\"]*\"|'[^']*')", "", l):
            if ch in "([{": st.append(ch)
            elif ch in ")]}" and st and "([{"[")]}".index(ch)] == st[-1]: st.pop()
        res.append(l + "".join({"(": ")", "[": "]", "{": "}"}[c] for c in reversed(st)))
    return "\n".join(res) + "\n"
def truncate_to_valid(prompt, body):
    ls = body.splitlines()
    for n in range(len(ls), 0, -1):
        b = "\n".join(ls[:n]) + "\n"
        if _ok(prompt, b): return b
    return body
def missing_return(prompt, body):
    ls = body.rstrip("\n").splitlines()
    if not ls or "return" in body or "yield" in body: return body
    last = ls[-1].strip()
    try:
        node = ast.parse(last)
        if len(node.body) == 1 and isinstance(node.body[0], ast.Expr): ls[-1] = ls[-1][: len(ls[-1]) - len(ls[-1].lstrip())] + "return " + last
        elif len(node.body) == 1 and isinstance(node.body[0], ast.Assign) and isinstance(node.body[0].targets[0], ast.Name):
            ls.append(ls[-1][: len(ls[-1]) - len(ls[-1].lstrip())] + "return " + node.body[0].targets[0].id)
    except SyntaxError: pass
    return "\n".join(ls) + "\n"
def signature_mismatch(prompt, body):
    """Rename unknown free names to the sole parameter if function has exactly one param (e.g. 'n' vs 'x')."""
    m = re.search(r"def\s+\w+\(([^)]*)\)", prompt)
    if not m: return body
    params = [p.split("=")[0].split(":")[0].strip() for p in m.group(1).split(",") if p.strip()]
    if len(params) != 1: return body
    try: tree = ast.parse(prompt + body)
    except SyntaxError: return body
    assigned = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)} | set(params)
    import builtins
    free = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} - assigned - set(dir(builtins))
    if len(free) == 1:
        f = free.pop(); return re.sub(rf"\b{re.escape(f)}\b", params[0], body)
    return body
RULES = [("indent", fix_indent), ("brackets", close_brackets), ("truncate", truncate_to_valid), ("missing_return", missing_return), ("signature", signature_mismatch)]
def heal(prompt, body):
    applied = []
    for name, f in RULES:
        if name in ("indent", "brackets", "truncate") and _ok(prompt, body): continue
        nb = f(prompt, body)
        if nb != body: applied.append(name); body = nb
    return body, applied
