"""Sandboxed test execution: isolated python, temp dir, rlimits, timeout."""
import os, sys, subprocess, tempfile, resource, ast, re

def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))

def valid(code):
    try: ast.parse(code); return True
    except Exception: return False

def run_tests(code, tests, timeout=5):
    """Returns dict(passed, err, line) where line = first failing test line or error line."""
    src = code + "\n\n" + "\n".join(tests) + "\n"
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "t.py"); open(f, "w").write(src)
        try:
            r = subprocess.run([sys.executable, "-I", f], cwd=d, capture_output=True, timeout=timeout,
                               preexec_fn=_limits, stdin=subprocess.DEVNULL, env={"PATH": "/usr/bin:/bin"})
        except subprocess.TimeoutExpired:
            return dict(passed=False, err="timeout", line=None)
    err = r.stderr.decode("utf-8", "replace")
    if r.returncode == 0: return dict(passed=True, err="", line=None)
    ln = re.findall(r'File ".*t\.py", line (\d+)', err)
    lines = src.splitlines(); line = None
    if ln:
        i = int(ln[-1]) - 1
        if 0 <= i < len(lines): line = lines[i].strip()
    last = err.strip().splitlines()[-1] if err.strip() else f"exit {r.returncode}"
    return dict(passed=False, err=last[-300:], line=line)
