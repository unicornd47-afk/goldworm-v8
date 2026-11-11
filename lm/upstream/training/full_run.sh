#!/bin/bash
set -euo pipefail
cd "${OWN_CODER_ROOT:?set OWN_CODER_ROOT to the original project root}"
# (original: sourced a local env file that only set a cache directory; removed for release)
trap 'printf "{\"status\":\"PIPELINE_FAILED\",\"exit\":%s,\"time\":\"%s\"}\n" "$?" "$(date -Iseconds)" > v8/full1/FAILED.json' ERR
rm -f v8/full1/FAILED.json
cd v8
python -u full_train.py
python -u full_eval.py
