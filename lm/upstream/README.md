---
license: mit
library_name: pytorch
pipeline_tag: text-generation
tags:
- code
- python
- byte-level
- small-model
- research
- goldworm
---

# GoldWorm v8 coder (85M, byte-level) — research checkpoint

**Small research model. Low absolute pass rates. Not production-ready.**

GoldWormV8 is our own byte-level causal transformer (no tokenizer, vocabulary = 256 raw bytes) for completing
short Python function bodies. This repo contains the `full1` checkpoint `sft_best.pt`
(SFT step 809, selected only by dev cross-entropy), a `safetensors` export of the same weights, the model code,
the v8 training/eval code, and the raw result files. All numbers below are copied from the result files in `results/`.

## Model

| | |
|---|---|
| Architecture | GoldWormV8 = GoldWormV6 blocks (pre-LN, bias-free QKVO + GELU MLP, affine LayerNorm, tied head) with RoPE and causal SDPA |
| Size | d_model 768, 12 layers, 12 heads, FFN ×4 — 85,169,664 parameters |
| Context | 2048 bytes |
| Vocabulary | 256 (UTF-8 bytes), no tokenizer |
| Weights | trained from scratch; no external pretrained weights |
| Init of full1 | `v8/trial1/pretrain_best.pt` (own v8 lineage, trial1 pretrain started from random init) |

## Training data

- **Pretraining (continued, 350M byte positions):** Python source files from the installed packages torch, transformers,
  sympy, scipy, numpy, datasets, gradio and huggingface_hub (BSD-3-Clause / Apache-2.0; their license/notice files are in
  `pretrain_corpus_licenses/`). 2049-byte windows, 108,013,035 train bytes / 4,231,185 dev bytes. Windows sharing any 13-byte
  n-gram with eval prompts/tests, eval function names or a concept blocklist were removed (see `results/data_stats.json`).
  The corpus itself is not uploaded; `training/trial_data.py` rebuilds it.
- **SFT:** `teacher_v4` — 324 train / 38 dev prompt→body pairs. Tasks and solutions were proposed by an external model,
  **Qwen/Qwen2.5-Coder-7B-Instruct** (NF4 quantized, used only as a *teacher* for distillation), then filtered by executing
  tests and by n-gram/name/topic guards against the eval sets. The teacher data is **not** uploaded here; only its
  description. No teacher weights are part of this model.

Training code: `training/full_train.py` (+ `trial_train.py`, `trial_data.py` for the trial1 lineage). It was run on a single
laptop GPU; it expects the original project layout (`OWN_CODER_ROOT`) and the non-uploaded data.

## Evaluation

Frozen protocol: k = 16 samples per task, temperature 0.8, top-k 40, max 160 new bytes, sampling seed 20261107 + task index;
paired bootstrap over tasks, B = 10000, seed 777, 95% percentile CI. Each eval set has 30 small Python function tasks
with hidden assert tests (task files are not uploaded; sha256 eval1 `3792a731…`, eval2 `60c20524…`).

### eval1 (30 tasks)

| metric | GoldWormV8 full1 |
|---|---|
| pass@1 (mean over 16 samples) | **7.08%** |
| pass@16 | **16.67%** (5/30 tasks) |
| syntactically valid | 92.92% |

Solved tasks (passing samples out of 16): T01 7/16, T03 1/16, T04 10/16, T05 15/16, T07 1/16. The other 25 tasks: 0/16.

Paired comparison vs. our previous v2 model on the same tasks and seeds:

| metric | v2 | v8 full1 | Δ | 95% CI |
|---|---|---|---|---|
| pass@1 | 3.54% | 7.08% | +3.54 pp | **−2.29 to +10.83 pp (includes 0)** |
| pass@16 | 6.67% | 16.67% | +10.00 pp | +0.00 to +20.00 pp |
| valid | 76.46% | 92.92% | +16.46 pp | +7.91 to +24.79 pp |

The pass@1 improvement over v2 is **not statistically established**.

### eval2 (30 harder tasks)

pass@1 **0.42%**, pass@16 **6.67%** (2/30 tasks: E2_01 1/16, E2_16 1/16), valid 91.67%.
Vs v2: pass@1 +0.42 pp (CI +0.00 to +1.04), pass@16 +6.67 pp (CI +0.00 to +16.67).
Caveat from the run: teacher_v4 used a relaxed concept filter, so eval2 is not a guaranteed test on unseen concepts.

Both eval sets were reused across several experiments, are tiny (n = 30), and should be treated as development sets,
not as a clean held-out benchmark.

## Follow-up experiments on this checkpoint (all rejected)

Inference-only selection/harness experiments with pre-registered gates. None improved the model; details in
`results/rejected_followups/`.

- **EnvOpt-v8 — REJECTED.** Pre-registered plan sha256 `bd010a0c47bd9689e80b13c4354353f5e250232eec5450dfb6cb7822412211ed`.
  60 tasks (eval1 + eval2). G0 (leakage audit) PASS; **G1 FAIL**: harness-p1 5.00% → 3.33% (Δ −1.67 pp, CI −6.67 to +3.33);
  G2 (validity) PASS. Raw pass@1 dropped 3.75% → 1.46%.
- **MBR-EXEC-style selection — REJECTED.** Pooled 60 tasks: 3/60 = 5.00% vs raw pass@1 3.75%, Δ +1.25 pp (CI −0.52 to +3.85);
  vs the frozen v3 harness −3.33 pp (CI −8.33 to +0.00).
- **Dynamic test-writing harness (Qwen2.5-0.5B-Instruct as test-writing teacher) — REJECTED.** Pooled selected 6.67% vs raw
  3.75%, Δ +2.92 pp (CI −0.21 to +7.29); vs v3 −1.67 pp (CI −5.00 to +0.00). Only 16/60 tasks received discriminating tests.

The oracle over all 16 samples is 7/60 pooled tasks, so selection alone cannot go far; candidate generation has to improve.

## Usage (CPU, tested)

```bash
pip install torch safetensors
CUDA_VISIBLE_DEVICES='' python inference_cpu.py
```

```python
from inference_cpu import load, complete
m = load()  # CPU
prompt = 'def add(a, b):\n    """Return the sum of a and b."""\n'
print(prompt + complete(m, prompt, seed=0))
```

Tested on CPU (torch 2.12.1): loads in about 1 s; `model.safetensors` and `sft_best.pt` give identical logits (max abs diff 0.0).
Expect frequently wrong code: with seeds 0–2 on the prompt above the model produced `return a * b == 0`,
`return a + b > 1 if b > 1 else a` and `return a * (b - 1) * b`, all wrong. On eval1 this task (T01) passes 7 of 16 samples.

`sft_best.pt` is the original training checkpoint (keys: `cfg`, `model_state_dict`, `metadata`, RNG states; no optimizer state)
and loads with `torch.load(..., weights_only=True)`.

## Limitations

- Very low absolute pass rates; only trivial functions are solved.
- Python only; byte-level 2048 context.
- Small, repeatedly used eval sets; wide confidence intervals.
- SFT data comes from an external teacher model; possible semantic overlap with eval tasks cannot be fully excluded despite
  n-gram/name/topic filters.
- Generated code must be executed only in a sandbox.

## Files

- `model.safetensors`, `sft_best.pt`, `config.json` — weights and config
- `gwlm_v6.py`, `goldworm_v8.py` — model code (import path adapted for this flat layout)
- `inference_cpu.py` — CPU sampling
- `training/` — v8 training/eval code (paths scrubbed; eval data, teacher data and baseline sample files not included)
- `results/` — RESULTS_FULL.md, DONE.json, TRAIN_DONE.json, eval summaries, data stats, lineage proof, rejected follow-ups
- `pretrain_corpus_licenses/` — licenses of the pretraining source packages

License: MIT (code and weights in this repo). Third-party license files in `pretrain_corpus_licenses/` keep their own terms.
