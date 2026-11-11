"""CPU budget (process CPU seconds via time.process_time; single-threaded BLAS/torch).
Exceeding a phase cap or the global cap raises CapExceeded -> INVALID (no resume, no rerun)."""
import time
from contextlib import contextmanager

from . import spec as S


class CapExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, caps=None, global_cap=S.GLOBAL_CAP, clock=time.process_time, t0=None):
        self.caps = dict(S.CAPS if caps is None else caps)
        self.global_cap = float(global_cap)
        self.clock = clock
        self.t0 = clock() if t0 is None else float(t0)     # production: t0 = 0 (process start incl. imports)
        self.used = {k: 0.0 for k in self.caps}
        self.cur = None
        self.cur_t0 = None
        self.log = []

    def total(self):
        return self.clock() - self.t0

    @contextmanager
    def phase(self, name):
        if name not in self.caps:
            raise KeyError(name)
        if self.cur is not None:
            raise AssertionError("nested phase")
        self.cur, self.cur_t0 = name, self.clock()
        try:
            yield self
        finally:
            dt = self.clock() - self.cur_t0
            self.used[name] += dt
            self.log.append(dict(phase=name, cpu_s=dt, total_cpu_s=self.total()))
            self.cur = None
        self.check_static(name)

    def check(self):
        if self.cur is None:
            return
        used = self.used[self.cur] + (self.clock() - self.cur_t0)
        if used > self.caps[self.cur]:
            raise CapExceeded("phase %s: %.2f > cap %.1f" % (self.cur, used, self.caps[self.cur]))
        if self.total() > self.global_cap:
            raise CapExceeded("global: %.2f > %.1f" % (self.total(), self.global_cap))

    def check_static(self, name):
        if self.used[name] > self.caps[name]:
            raise CapExceeded("phase %s: %.2f > cap %.1f" % (name, self.used[name], self.caps[name]))
        if self.total() > self.global_cap:
            raise CapExceeded("global cap")

    def overhead(self):
        return self.total() - sum(v for k, v in self.used.items() if k != "overhead")

    def finalize(self):
        self.used["overhead"] = self.overhead()
        ok = all(self.used[k] <= self.caps[k] for k in self.caps) and self.total() <= self.global_cap
        return dict(used=self.used, caps=self.caps, total=self.total(), all_caps_respected=bool(ok), log=self.log)
