"""v8.3 host environment checks (PLAN r3 C1, C4; dahill PLAN_REVIEW r3 A2). Portable: Linux /proc first, then
psutil if installed, then `ps`/`sysctl`; anything that cannot be read FAILS CLOSED (start refused)."""
import os
import platform
import socket
import subprocess
import sys

from . import spec as S

THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


# ---------------- C1 thread readback ----------------
def thread_readout():
    import torch
    out = {v: os.environ.get(v) for v in THREAD_ENV}
    out["torch_num_threads"] = int(torch.get_num_threads())
    out["torch_num_interop_threads"] = int(torch.get_num_interop_threads())     # logged only (not pinned in v8.2)
    out["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return out


def thread_readback_ok(r):
    """C1: pinned values read back; mismatch -> NOT_LAUNCHED. Pinned = the v8.2 pins (3 BLAS env vars, torch=1)."""
    bad = [v for v in THREAD_ENV if r.get(v) != "1"]
    if r.get("torch_num_threads") != 1:
        bad.append("torch_num_threads")
    return (not bad), bad


# ---------------- A2 host fingerprint ----------------
def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass
    return platform.processor() or None


def host_fingerprint():
    """A2: hostname, CPU model, thread count (logical CPUs). Also platform, for the record only."""
    return dict(hostname=socket.gethostname(), cpu_model=_cpu_model(), cpu_count=os.cpu_count(),
                platform=platform.platform(), python=platform.python_version())


FINGERPRINT_KEYS = ("hostname", "cpu_model", "cpu_count")


def same_host(a, b):
    """A2: production refuses unless hostname, CPU model and thread count equal the probe's."""
    if not a or not b:
        return False, ["fingerprint missing"]
    diff = [k for k in FINGERPRINT_KEYS if a.get(k) != b.get(k) or a.get(k) is None]
    return (not diff), diff


# ---------------- C4 start rule ----------------
def load1():
    try:
        return float(os.getloadavg()[0])
    except (OSError, AttributeError):
        return None


def mem_available_kb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])
    except OSError:
        pass
    try:
        import psutil
        return int(psutil.virtual_memory().available // 1024)
    except Exception:
        return None


def process_cmdlines():
    """[(pid, cmdline)] of all processes, or None if unreadable."""
    out = []
    if os.path.isdir("/proc"):
        for d in os.listdir("/proc"):
            if d.isdigit():
                try:
                    with open("/proc/%s/cmdline" % d, "rb") as f:
                        c = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
                except OSError:
                    continue
                if c:
                    out.append((int(d), c))
        return out
    try:
        txt = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True)
    except Exception:
        return None
    for line in txt.splitlines():
        p = line.strip().split(None, 1)
        if len(p) == 2 and p[0].isdigit():
            out.append((int(p[0]), p[1]))
    return out


def start_rule(patterns=None, self_pids=None, load_fn=load1, mem_fn=mem_available_kb, procs_fn=process_cmdlines,
               load_max=None):
    """C4: 1-min load < load_max (ERRATUM E2.4: the host threshold fixed in RELEASE from the measured baseline; box
    default 1.0), MemAvailable >= 4 GiB, no exp001 arm / training process (pattern match on cmdline).
    Returns dict with ok, reasons and every measured value (logged right before launch)."""
    l1max = S.START_LOAD1_MAX if load_max is None else float(load_max)
    patterns = tuple(S.START_EXCLUDE_PATTERNS if patterns is None else patterns)
    self_pids = set(self_pids or ()) | {os.getpid(), os.getppid()}
    reasons = []
    l1 = load_fn()
    if l1 is None:
        reasons.append("load unreadable")
    elif not l1 < l1max:
        reasons.append("load1 %.2f >= %.3f" % (l1, l1max))
    mem = mem_fn()
    if mem is None:
        reasons.append("MemAvailable unreadable")
    elif mem < S.START_MEMAVAIL_MIN_KB:
        reasons.append("MemAvailable %d kB < %d kB" % (mem, S.START_MEMAVAIL_MIN_KB))
    procs = procs_fn()
    hits = []
    if procs is None:
        reasons.append("process list unreadable")
    else:
        hits = [(pid, c) for pid, c in procs if pid not in self_pids and any(p in c for p in patterns)]
        if hits:
            reasons.append("excluded process running: %s" % "; ".join("%d %s" % (p, c[:120]) for p, c in hits[:5]))
    return dict(ok=not reasons, reasons=reasons, load1=l1, load1_max_exclusive=l1max,
                mem_available_kb=mem, mem_available_min_kb=S.START_MEMAVAIL_MIN_KB, exclude_patterns=list(patterns),
                excluded_hits=[[p, c[:200]] for p, c in hits])


# ---------------- E1.2 P-core pinning (hybrid CPUs only) ----------------
def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def parse_cpulist(txt):
    out = []
    for part in (txt or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(out)


def cpu_topology(sysfs="/sys", cpuinfo="/proc/cpuinfo", ncpu=None):
    """E1.2: hybrid detection and P-core list from /sys. Order of evidence:
    (1) Intel hybrid PMUs /sys/devices/cpu_core/cpus + cpu_atom/cpus -> P-cores = cpu_core list;
    (2) /sys/devices/system/cpu/cpu*/cpu_capacity differing -> P-cores = max capacity;
    (3) cpufreq cpuinfo_max_freq differing -> P-cores = max frequency (stricter: treated as hybrid).
    Hybrid also if /proc/cpuinfo flags contain 'hybrid'. Returns dict(hybrid, p_cores, method, reason);
    p_cores None on a hybrid CPU whose P-cores cannot be determined (caller fails closed)."""
    ncpu = os.cpu_count() if ncpu is None else ncpu
    flags_hybrid = False
    txt = _read(cpuinfo)
    if txt:
        for line in txt.splitlines():
            if line.lower().startswith("flags"):
                flags_hybrid = "hybrid" in line.split(":", 1)[1].split()
                break
    core, atom = _read(os.path.join(sysfs, "devices/cpu_core/cpus")), _read(os.path.join(sysfs, "devices/cpu_atom/cpus"))
    if core and atom:
        return dict(hybrid=True, p_cores=parse_cpulist(core), e_cores=parse_cpulist(atom), method="pmu cpu_core/cpu_atom")
    base = os.path.join(sysfs, "devices/system/cpu")
    cap = {c: _read(os.path.join(base, "cpu%d" % c, "cpu_capacity")) for c in range(ncpu)}
    if all(v is not None for v in cap.values()) and len(set(cap.values())) > 1:
        m = max(int(v) for v in cap.values())
        return dict(hybrid=True, p_cores=sorted(c for c, v in cap.items() if int(v) == m), method="cpu_capacity")
    fq = {c: _read(os.path.join(base, "cpu%d" % c, "cpufreq/cpuinfo_max_freq")) for c in range(ncpu)}
    if all(v is not None for v in fq.values()) and len(set(fq.values())) > 1:
        m = max(int(v) for v in fq.values())
        return dict(hybrid=True, p_cores=sorted(c for c, v in fq.items() if int(v) == m), method="cpuinfo_max_freq")
    if flags_hybrid:
        return dict(hybrid=True, p_cores=None, method=None, reason="cpuinfo flag 'hybrid' but P-cores not determinable")
    return dict(hybrid=False, p_cores=None, method="no hybrid evidence in /sys or /proc/cpuinfo")


def core_mhz(core, sysfs="/sys"):
    b = os.path.join(sysfs, "devices/system/cpu/cpu%d/cpufreq" % core)
    mx, cur = _read(os.path.join(b, "cpuinfo_max_freq")), _read(os.path.join(b, "scaling_cur_freq"))
    return (int(mx) / 1000.0 if mx else None), (int(cur) / 1000.0 if cur else None)


def plan_pinning(topo=None, sysfs="/sys"):
    """E1.2: on a hybrid CPU one P-core (the highest-numbered logical CPU of the P-core list; deterministic, avoids
    cpu0 interrupt load). Not hybrid: no pinning (box unchanged). Returns (pinning dict, reasons)."""
    topo = topo or cpu_topology(sysfs)
    if not topo["hybrid"]:
        return dict(pinned=False, hybrid=False, core=None, p_cores=None, method=topo.get("method")), []
    if not topo.get("p_cores"):
        return None, ["E1.2: hybrid CPU but P-cores not determinable (%s); start refused" % topo.get("reason")]
    core = max(topo["p_cores"])
    mx, _ = core_mhz(core, sysfs)
    return dict(pinned=True, hybrid=True, core=core, p_cores=list(topo["p_cores"]), method=topo["method"],
                core_max_mhz=mx), []


def apply_pinning(pin, setter=None, getter=None, sysfs="/sys"):
    """Pins this process (and threads created later) to pin['core']; verifies by read-back. Returns reasons."""
    if not pin or not pin.get("pinned"):
        return []
    setter = setter or getattr(os, "sched_setaffinity", None)
    getter = getter or getattr(os, "sched_getaffinity", None)
    if setter is None or getter is None:
        return ["E1.2: sched_setaffinity unavailable on this OS; start refused"]
    try:
        setter(0, {int(pin["core"])})
    except OSError as ex:
        return ["E1.2: pinning to core %s failed: %s" % (pin["core"], ex)]
    got = set(getter(0))
    if got != {int(pin["core"])}:
        return ["E1.2: affinity read-back %s != {%s}" % (sorted(got), pin["core"])]
    pin["core_cur_mhz_at_pin"] = core_mhz(int(pin["core"]), sysfs)[1]
    return []


def pinning_matches(bound, current):
    """dahill req (4): production pinned identically: same hybrid status, same P-core list, same core."""
    if bound is None or current is None:
        return False, ["pinning record missing"]
    diff = [k for k in ("pinned", "hybrid", "core", "p_cores") if bound.get(k) != current.get(k)]
    return (not diff), diff


def _unpin_child():
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, range(os.cpu_count()))
        except OSError:
            pass


# ---------------- C5 sidecar launcher ----------------
def sidecar_path():
    return os.path.join(S.V83_DIR, S.V83["sidecar_file"])


def start_sidecar(pid, out_path, interval=S.SIDECAR_INTERVAL_S, path=None, python=None):
    """Starts load_sidecar.py as its own process: new session (setsid), stdin/stdout detached, SIGHUP ignored inside
    the sidecar (nohup). It survives the parent's terminal and stops by itself when `pid` exits."""
    import subprocess as sp
    path = path or sidecar_path()
    err = open(out_path + ".err", "ab")
    p = sp.Popen([python or sys.executable, path, "--pid", str(int(pid)), "--out", out_path, "--interval", str(interval)],
                 stdin=sp.DEVNULL, stdout=sp.DEVNULL, stderr=err, start_new_session=True, close_fds=True,
                 preexec_fn=_unpin_child)          # E1.2: the sidecar never shares the pinned compute core
    err.close()
    return p


def is_box(host):
    """uni 11:12 (5): True only if hostname, CPU model and CPU count all equal spec.BOX_FINGERPRINT."""
    return same_host(host, S.BOX_FINGERPRINT)[0]


def sidecar_capable(path=None):
    """Stricter reading (IMPL_NOTES S5): every C5 field must be readable on this host, else the start is refused.
    Uses the bound sidecar script itself (one sample of this process)."""
    import importlib.util
    path = path or sidecar_path()
    if not os.path.exists(path):
        return False, ["sidecar script missing"]
    spec = importlib.util.spec_from_file_location("load_sidecar_check", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    r1 = m.sample(os.getpid(), None)
    r2 = m.sample(os.getpid(), r1)
    need = ("load_1_5_15", "runnable", "mem_available_kb", "steal_ticks", "cpu_mhz_mean", "run_proc_cpu_s")
    bad = [k for k in need if r2.get(k) is None]
    return (not bad), bad
