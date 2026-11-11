#!/bin/bash
# probe retry launcher (outside frozen code): waits up to 20 min for load1 < 0.80 on 3 consecutive 5-s samples, then one probe attempt.
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 WM_ROOT="$HOME/wm_v83/worldmodel" WM_AUDIT_ROOT="$HOME/wm_v83"
PY="$HOME/wm_v83/venv/bin/python"; C="$WM_ROOT/v8.3/code"; L="$HOME/wm_v83/launcher.log"; ok=0
sleep 5
for i in $(seq 1 240); do l=$(cut -d' ' -f1 /proc/loadavg); echo "$(date -Is) retry load1=$l" >>"$L"
  if awk "BEGIN{exit !($l<0.80)}"; then ok=$((ok+1)); else ok=0; fi; [ $ok -ge 3 ] && break; sleep 5; done
cd "$WM_ROOT"; echo "$(date -Is) START probe retry (consecutive_ok=$ok)" >>"$L"
"$PY" "$C/run_probe.py" > "$HOME/wm_v83/probe_stdout_retry.txt" 2>&1; echo "$(date -Is) END probe retry rc=$?" >>"$L"
