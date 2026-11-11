"""CPU budget (process CPU seconds via time.process_time; single-threaded BLAS/torch).
Exceeding a phase cap or the global cap raises CapExceeded -> INVALID (no resume, no rerun).

v8.3 (PLAN r3 C3/K1, C6, C6b):
 - global_cap = sum of the phase caps (C3, <= 3600) AND an explicit, separate K1 abort at 3600 CPU-s
   (k1_cap): cumulative CPU-s > 3600 -> INVALID even if every phase is under its own cap.
 - on_event(event, **fields): phase_start / phase_end (process CPU-s, wall time, thread readout) -> LEDGER (C6).
 - every check() records the CPU-s since the previous check of the same phase; the maximum per phase is the
   'measured per-check jump' that RESULTS reports (C6b). No overshoot tolerance."""
import time
from contextlib import contextmanager

from . import spec as S


class CapExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, caps=None, global_cap=S.GLOBAL_CAP, clock=time.process_time, t0=None, k1_cap=S.K1_GLOBAL_CAP,
                 on_event=None, thread_readout=None):
        self.caps = dict(S.CAPS if caps is None else caps)
        self.global_cap = float(global_cap)
        self.k1_cap = float(k1_cap)
        self.clock = clock
        self.t0 = clock() if t0 is None else float(t0)     # production: t0 = 0 (process start incl. imports)
        self.used = {k: 0.0 for k in self.caps}
        self.cur = None
        self.cur_t0 = None
        self.log = []
        self.on_event = on_event
        self.thread_readout = thread_readout
        self.n_checks = {k: 0 for k in self.caps}
        self.max_jump = {k: 0.0 for k in self.caps}
        self._last_check = None
        self.warmup_cpu_s = 0.0                            # ERRATUM E2.1: inside the phase (fits) accounting
        self.n_warmups = 0

    def total(self):
        return self.clock() - self.t0

    def _emit(self, event, **kw):
        if self.on_event is not None:
            self.on_event(event, **kw)

    @contextmanager
    def phase(self, name):
        if name not in self.caps:
            raise KeyError(name)
        if self.cur is not None:
            raise AssertionError("nested phase")
        self.cur, self.cur_t0 = name, self.clock()
        self._last_check = self.cur_t0
        self._emit("phase_start", phase=name, process_cpu_s=self.cur_t0 - self.t0, wall_unix=time.time(),
                   cap_cpu_s=self.caps[name], threads=self.thread_readout() if self.thread_readout else None)
        ok = False
        try:
            yield self
            ok = True
        finally:
            dt = self.clock() - self.cur_t0
            self.used[name] += dt
            self.log.append(dict(phase=name, cpu_s=dt, total_cpu_s=self.total()))
            self._emit("phase_end", phase=name, phase_cpu_s=dt, process_cpu_s=self.total(), wall_unix=time.time(),
                       cap_cpu_s=self.caps[name], completed=ok, n_checks=self.n_checks[name],
                       max_check_jump_cpu_s=self.max_jump[name],
                       threads=self.thread_readout() if self.thread_readout else None)
            self.cur = None
        self.check_static(name)

    @contextmanager
    def warmup(self):
        """E1.1/E2.1 warm-up CPU: tallied separately (reported) and, per ERRATUM E2.1, part of the projection: it
        stays inside the current phase (fits) and counts toward that phase cap, the global cap and K1."""
        t = self.clock()
        try:
            yield self
        finally:
            self.warmup_cpu_s += self.clock() - t
            self.n_warmups += 1

    def check_global(self):
        self._check_global(self.total())

    def check(self):
        if self.cur is None:
            return
        now = self.clock()
        jump = now - self._last_check
        self._last_check = now
        self.n_checks[self.cur] += 1
        if jump > self.max_jump[self.cur]:
            self.max_jump[self.cur] = jump
        used = self.used[self.cur] + (now - self.cur_t0)
        if used > self.caps[self.cur]:
            raise CapExceeded("phase %s: %.2f > cap %.1f" % (self.cur, used, self.caps[self.cur]))
        self._check_global(now - self.t0)

    def _check_global(self, total):
        if total > self.k1_cap:
            raise CapExceeded("K1 global: %.2f > %.1f" % (total, self.k1_cap))
        if total > self.global_cap:
            raise CapExceeded("global: %.2f > %.1f" % (total, self.global_cap))

    def check_static(self, name):
        if self.used[name] > self.caps[name]:
            raise CapExceeded("phase %s: %.2f > cap %.1f" % (name, self.used[name], self.caps[name]))
        self._check_global(self.total())

    def overhead(self):
        return self.total() - sum(v for k, v in self.used.items() if k != "overhead")

    def finalize(self):
        self.used["overhead"] = self.overhead()
        tot = self.total()
        ok = all(self.used[k] <= self.caps[k] for k in self.caps) and tot <= self.global_cap and tot <= self.k1_cap
        return dict(used=self.used, caps=self.caps, total=tot, all_caps_respected=bool(ok), log=self.log,
                    global_cap=self.global_cap, k1_cap=self.k1_cap, k1_respected=bool(tot <= self.k1_cap),
                    n_checks=self.n_checks, max_check_jump_cpu_s=self.max_jump,
                    warmup_cpu_s=self.warmup_cpu_s, n_warmups=self.n_warmups)
