#!/bin/bash
set -euo pipefail
cd "${OWN_CODER_ROOT:?set OWN_CODER_ROOT to the original project root}"
# (original: sourced a local env file that only set a cache directory; removed for release)
trap 'printf "{\"status\":\"PIPELINE_FAILED\",\"exit\":%s}\n" "$?" > v8/trial1/FAILED.json' ERR
python -u v8/trial_train.py
python -u v8/trial_eval.py
