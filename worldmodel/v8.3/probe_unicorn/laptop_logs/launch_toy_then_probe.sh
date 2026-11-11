#!/bin/bash
# launcher outside the frozen code: waits (max 15 min each) for load1 < 0.95 before each attempt; the attempt itself applies C4.
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 WM_ROOT="$HOME/wm_v83/worldmodel" WM_AUDIT_ROOT="$HOME/wm_v83"
PY="$HOME/wm_v83/venv/bin/python"; C="$WM_ROOT/v8.3/code"; L="$HOME/wm_v83/launcher.log"
waitload(){ for i in $(seq 1 180); do l=$(cut -d' ' -f1 /proc/loadavg); echo "$(date -Is) load1=$l" >>"$L"; awk "BEGIN{exit !($l<0.95)}" && return 0; sleep 5; done; return 1; }
cd "$WM_ROOT"
waitload || echo "$(date -Is) load never < 0.95 before toy; starting anyway (attempt will log refusal)" >>"$L"
echo "$(date -Is) START toy" >>"$L"; "$PY" "$C/run_preflight.py" > "$HOME/wm_v83/toy_stdout.txt" 2>&1; echo "$(date -Is) END toy rc=$?" >>"$L"
last=$(tail -1 "$WM_ROOT/v8.3/probe/TOY_LEDGER.jsonl")
echo "$last" | grep -q '"outcome": *"TOY_OK"' || { echo "$(date -Is) toy not TOY_OK; no probe" >>"$L"; exit 0; }
waitload || echo "$(date -Is) load never < 0.95 before probe; starting anyway" >>"$L"
echo "$(date -Is) START probe" >>"$L"; "$PY" "$C/run_probe.py" > "$HOME/wm_v83/probe_stdout.txt" 2>&1; echo "$(date -Is) END probe rc=$?" >>"$L"
