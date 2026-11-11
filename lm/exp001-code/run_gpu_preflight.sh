#!/bin/bash
# PENDING: run only on unicorn (machineId 698f92ba-b13a-4c02-9bd5-d4f8603389dd)
# AFTER a lock note in the room confirms the RTX 4060 is free for this project.
# Do not start training arms from this script.
set -euo pipefail
# Expected environment on unicorn: a torch+cuda install.
# Example (adjust python once the lock is in place):
#   python /workspace/growth/code/preflight.py --device cuda --out /workspace/growth/results/preflight_torch_cuda.json
echo "GPU preflight is pending a room lock on unicorn. Not running."
exit 2
