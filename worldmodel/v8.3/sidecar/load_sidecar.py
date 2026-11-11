"""v8.3 C5 load sidecar (PLAN_v8.3 r3): independent process (started with setsid, stdin/stdout detached, SIGHUP
ignored = nohup) that writes one JSON line every 30 s with wall clock, loadavg 1/5/15, runnable/total tasks,
MemAvailable, steal time (/proc/stat), mean CPU MHz (/proc/cpuinfo) and the CPU-s of the watched run process
(/proc/<pid>/stat utime+stime). It stops when the watched pid is gone. Wall gaps > 90 s are listed by gaps()
(box paused). A gap invalidates nothing; caps are CPU-s. Standalone: imports only the standard library
(psutil optional off Linux). Usage: python load_sidecar.py --pid PID --out FILE [--interval 30]"""
import argparse
import json
import os
import signal
import sys
import time

INTERVAL_S = 30.0
GAP_S = 90.0


def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def loadavg():
    t = _read("/proc/loadavg")
    if t:
        p = t.split()
        run, tot = p[3].split("/")
        return [float(p[0]), float(p[1]), float(p[2])], int(run), int(tot)
    try:
        return list(os.getloadavg()), None, None
    except (OSError, AttributeError):
        return None, None, None


def mem_available_kb():
    t = _read("/proc/meminfo")
    if t:
        for line in t.splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    try:
        import psutil
        return int(psutil.virtual_memory().available // 1024)
    except Exception:
        return None


def cpu_ticks():
    """Aggregate /proc/stat 'cpu' line -> (steal_ticks, total_ticks) or (None, None)."""
    t = _read("/proc/stat")
    if not t:
        return None, None
    f = [int(x) for x in t.splitlines()[0].split()[1:]]
    steal = f[7] if len(f) > 7 else 0
    return steal, sum(f[:8])


def cpu_mhz():
    t = _read("/proc/cpuinfo")
    if not t:
        return None
    v = [float(l.split(":", 1)[1]) for l in t.splitlines() if l.lower().startswith("cpu mhz")]
    return sum(v) / len(v) if v else None


def proc_cpu_s(pid):
    t = _read("/proc/%d/stat" % pid)
    if t:
        rest = t[t.rindex(")") + 2:].split()          # fields after comm; utime = field 14 -> rest[11]
        return (int(rest[11]) + int(rest[12])) / os.sysconf("SC_CLK_TCK")
    try:
        import psutil
        c = psutil.Process(pid).cpu_times()
        return c.user + c.system
    except Exception:
        return None


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    st = _read("/proc/%d/stat" % pid)
    if st and st[st.rindex(")") + 2:].startswith("Z"):
        return False
    return True


def sample(pid, prev):
    la, run, tot = loadavg()
    steal, total = cpu_ticks()
    rec = dict(wall_unix=time.time(), wall_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"), pid=pid,
               load_1_5_15=la, runnable=run, tasks=tot, mem_available_kb=mem_available_kb(),
               steal_ticks=steal, total_ticks=total, cpu_mhz_mean=cpu_mhz(), run_proc_cpu_s=proc_cpu_s(pid))
    if prev and steal is not None and prev.get("steal_ticks") is not None and total != prev["total_ticks"]:
        rec["steal_frac_since_prev"] = (steal - prev["steal_ticks"]) / float(total - prev["total_ticks"])
    else:
        rec["steal_frac_since_prev"] = None
    return rec


def run(pid, out, interval=INTERVAL_S, sleep=time.sleep, max_samples=None):
    prev, n = None, 0
    with open(out, "a", buffering=1) as f:
        f.write(json.dumps(dict(event="sidecar_start", wall_unix=time.time(), pid=pid, sidecar_pid=os.getpid(),
                                interval_s=interval)) + "\n")
        while alive(pid):
            rec = sample(pid, prev)
            rec["event"] = "sample"
            rec["seq"] = n
            f.write(json.dumps(rec) + "\n")
            prev, n = rec, n + 1
            if max_samples is not None and n >= max_samples:
                break
            sleep(interval)
        f.write(json.dumps(dict(event="sidecar_end", wall_unix=time.time(), pid=pid, samples=n,
                                watched_alive=alive(pid))) + "\n")
    return n


def gaps(path, gap_s=GAP_S):
    """Wall-clock gaps > gap_s between consecutive samples (C5: listed in RESULTS as 'box paused')."""
    ts = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("event") == "sample":
                    ts.append(r["wall_unix"])
    return [dict(after_unix=a, before_unix=b, gap_s=b - a) for a, b in zip(ts, ts[1:]) if b - a > gap_s]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=INTERVAL_S)
    a = ap.parse_args(argv)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    run(a.pid, a.out, a.interval)


if __name__ == "__main__":
    sys.exit(main())
