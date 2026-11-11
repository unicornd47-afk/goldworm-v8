# PREREG_exp001_v2.4 — GoldWorm v8 function-preserving growth

**Status:** DRAFT for PLAN_PASS / independent code review. **No training arm has been started.** No publish/push.
**Date (Europe/Berlin):** 2026-11-11.
**Owner:** Dark Unicorn (HF: Uuuuuuniiiiiii). Work tree: `/workspace/growth/` only.
**Supersedes:** `PREREG_exp001_v2.3.md` (sha256 `03b43e3d5eb9220481e887159b373d109309d98845b2978c192ec82deb63c5ff`, left intact). v2.2 `e6988097db437e58c0d7a84e115b3049f44b38b59408b54d4ab736ec895b9c88`, v2.1 `b5f0db1382165c2b55303c49213e1b9e4afb90cb873e87bf0060452a7b725a3e`, and v2 `aba40ea487ad6714d766802dd02df28ba276b6b4587b72de6100f62d31ab048b` are also left intact. Earlier amended draft: `archive/PREREG_exp001_amended.md` / `PREREG_exp001.md`, sha256 `5494682e2f99e4088975e9d687f817b2f9ff832f7fec2020c1210d90a5808a63`. Held-out v1 is at `data/heldout_exp001_v1_superseded.bin`. The v2.3 code manifest is archived at `archive/CODE_MANIFEST_v2.3_89af0138.json` (sha256 `89af0138f5e1a08ce19d7764ac1709db75d47e69bd1bddcb5f0fbb5f3ab82487`).

**Provenance [L].** The first lock-era draft (v1, sha256 `dbc735fb289b17a77b30409f490bc5cb0fe028d309d7e2dfd2c65830eeea6cf0`) was overwritten in place during amendment, before archival. Only that dahill-verified hash remains; the v1 bytes are not recoverable from the growth tree. **Future versions are written to new files** (this document is `PREREG_exp001_v2.4.md`; v2, v2.1, v2.2, and v2.3 were not overwritten) and must not be overwritten in place.

**Reviews folded in.** dahill `REVIEW_exp001_v2.json` sha256 `70137b618b6aae403c92f4c716f8a080a3b7bc626e13f44ebdd491067fa3833d` (on v2 / manifest `620087b6…`) and `REVIEW_exp001_v2.1.json` sha256 `91106a83f9d7957380e502e0dd12d179d241310cb7c207052c4a47b5ef1a2ba6` (on v2.1 / manifest `42f79b97…`). v2.2 closed P1 and K4–K7 (and kept the already-closed K1–K3 path, fail-closed FP gate, and hypothesis (b) [H]). v2.3 added the frozen analyzer. **This revision is fail-closed cleanup after uni’s v2.3 review of `analyze.py` and the outstanding v8 load / RAM-preflight items:** official analysis may not leak extra seeds into the CI; a missing FP gate fails for grown arms; unexpected checkpoint keys abort; projected peak RAM is part of `PREFLIGHT_ABORT`. No real arms were started.

**Order of work [L]:** corpus locked (done) → measured torch preflight on the box CPU (done) → **code freeze + independent review** (this re-freeze) → first arm is **G, ratio A**, then C/S/mirrors for that seed, then remaining seeds, then ratio B only after A. One large model at a time. All arms, including the proxy, run on the **box CPU in fp32**. GPU / bf16 are out of scope for exp001.

Claim tags: **[L]** load-bearing, **[H]** heuristic, **[D]** decorative.

## Changelog (v2.3 → v2.4)

- **Official seeds only [L].** The official decision (`decide_growth` / `decide_proxy` and official layout discovery) uses exactly seeds **11, 22, 33**. Any other seed in the inputs or the layout raises `AnalyzeError` (it is not averaged into the CI). A separate `--smoke` / `--non-official` path is required for format checks (e.g. seed 7) and always returns **NO_ADVANCE** with reason `non-official`.
- **Missing FP gate [L].** For grown arms (`G`, `G_zeroall`, `G_signflip`, and the proxy grown arm), an `fp_pass` that is missing or `None` makes `growth_gate_pass` false. “If present” remains allowed only for C and S.
- **Checkpoint load [L].** The v8 / arm checkpoint load is fail-closed on unexpected keys after the documented tied-embedding, `pos_emb`, and `rope_freq` handling (`model_io.load_state_dict_fail_closed`, used by `load_v8` and `run_arm.load_or_empty`). Loading `upstream_v8_c0dc0f13/model.safetensors` still works (forward pass finite; unexpected-key negative test raises).
- **RAM preflight [L].** `arm_preflight` measures peak RSS during the preflight steps, projects a conservative full-run peak (eval activations plus one extra parameter copy for growth), compares it to `MemAvailable` from `/proc/meminfo`, and raises `PREFLIGHT_ABORT` if the projection is above **0.9×** available. A negative test with injected `MemAvailable = 1` kB aborts.
- **`G_zeroall` barely learns** is tagged **[H]**. Once noise makes `wo`/`fc2` nonzero, `fc1`/`Wv` get gradients from step 1, so the branch can learn; if it does, that is reported as a result, not a failure.
- New synthetic tests: (e) stray seed 7 → `AnalyzeError`; (f) `fp_pass is None` → `NO_ADVANCE`; (g) noisy margin case (reported CI) plus a noisy case that must `ADVANCE`. Prior tests still pass.

---

## 0. Hardware and precision lock [L]

**exp001 device/precision lock:** every arm, including the full-pipeline proxy, runs on the shared **box CPU in fp32**. The FP tolerance is **1×10⁻⁵** max abs logit difference. Thread count is frozen at **4**. `EXP001_DEVICE="cpu"` and `EXP001_DTYPE="fp32"` are set in `code/constants.py`. The frozen CLI refuses any other device or dtype.

| device | role in exp001 |
|---|---|
| shared **box** | 8 vCPU, 15.6 GiB RAM, 0 swap, no GPU. **The only legal device.** All locks, FP, Lean, torch preflight, negative tests, and the proxy-size smoke test were measured here. |
| **unicorn** laptop | RTX 4060 Laptop, 8 GB VRAM (only **~3.5 GB free** because a foreign **cosyvoice** process is resident). **Out of scope for exp001.** Not used. A GPU preflight script exists at `code/run_gpu_preflight.sh` and must not be run for this experiment. |

**GPU and bf16 are explicitly out of scope for exp001.** Any later use of the GPU or of bf16 requires a **new prereg**, including a measured identity baseline (original 85M logits in fp32 vs the same model in that device/precision) and a re-measured FP tolerance. The 1×10⁻⁵ fp32 tolerance does not carry over.

Compute is matched in **FLOPs**. Wall time is recorded for this CPU and is not transferable to another device.

---

## 1. Recon (short) [L]

GoldWorm v8 coder, 85,169,664 unique params, rev **c0dc0f13**, MIT. d_model 768, 12 layers, 12 heads, FFN×4, vocab 256 (raw bytes), ctx 2048, pre-LN, RoPE, tied embeddings. `n(L) = 198144 + L × 7080960`. Weights sha256 `8a7b5716ab340a672bc307772fd3e8387beee4793c62e1f256c31b8c70aea0a7`. SFT step 809. Original pretrain/SFT data licenses and the missing exact 350M-byte corpus are as in the amended draft; exp001 trains on the locked subset in §5.

---

## 2. Hypothesis, endpoints, decision rules

**Hypothesis (a) — growth vs continue [L].** Depth-identity growth from the 85M SFT checkpoint, followed by registered noise and continued training with the split LR rule in §3.2, reaches a lower held-out bits-per-byte than **continuing to train the original 85M model (C)** at the same incremental FLOPs, on the box CPU in fp32, under the stats in §2.1. This is the only load-bearing growth hypothesis.

**Hypothesis (b) — growth vs from-scratch [H].** The same grown model is also expected to beat a from-scratch model (S) of the grown depth at the same incremental FLOPs. This is a **heuristic reference only**. G beating S proves nothing about growth: S starts from random init and sees only the incremental budget, while G starts from the trained 85M function. There is **no** growth ADVANCE that rests on hypothesis (b).

**ONE primary endpoint [L]:** next-byte bits/byte on **non-overlapping 256-byte chunks** of the locked 32×2048-byte held-out windows (8 chunks per window, 256 chunks). Teacher-forcing is inside each 256-byte chunk, identical for every arm.

**Descriptive secondaries** (not used to ADVANCE the growth claim): full-window **2048-context** bits/byte **[L]** (registered descriptive); **S** loss versus position, especially **above pos 256** (RoPE; S is trained at T=256) **[L]**; pass@1 on the antigoldworm 16-task suite (fresh seed `20261111+task_index`) **[D]**; held-out A NLL **[D]**; 3-bit export gaps **[L]** (descriptive); G vs S bits/byte, the metric for hypothesis (b) **[H]**.

### 2.1 Stats (all decision rules) [L]

- Cluster bootstrap over **source-file IDs** (32 clusters, one window per file in the current held-out), B = 10,000, seed **77701**, 95% percentile CI.
- For each file and each seed, mean bits/byte over that file’s 256-byte chunks, then the paired difference.
- **Seed aggregation:** per file, mean of the paired difference over seeds **11, 22, 33**; then bootstrap those 32 file-level means.
- **Sign rule for ADVANCE:** the G−C mean difference (aggregated over files) must be **negative** (grown better) in **all three seeds**, not merely the same sign. Per-seed results are always reported.
- Margin: CI **upper bound ≤ −0.01 bits/byte**.

Held-out claims cover **only this locked distribution**. Composition of the current 32 files: **filelock 11, click 6, anyio 5, pyparsing 4, h11 4, PyYAML 1, tabulate 1** (MIT / BSD-3-Clause). This is a narrow code-library slice, not “Python at large.” **[L]** / narrowness **[N1]**.

### 2.2 Load-bearing comparison is G vs C only [L]

**ADVANCE (growth, ratio A or B)** requires all of:

1. G vs C: seed-averaged cluster-bootstrap 95% CI upper bound **≤ −0.01 bpb**, and G−C **negative in all 3 seeds**;
2. The divergence + knockout gate in §3.3 passes at **N_div** (a pass at step 0 does not count);
3. The torch FP gate in §3.4 passed on the box CPU in fp32, before noise;
4. No fail-closed abort and no 0.9× preflight abort.

**G vs S is hypothesis (b) [H], a heuristic reference only.** Clause 1 may still *report* G vs S with the same CI machinery. There is **no** growth ADVANCE that rests on G vs S.

**Growth-beats-brute-force** claims are allowed **only** from the total-FLOP **proxy** in §9, which has its **own** decision rule and NO_ADVANCE threshold (same stats: cluster bootstrap, **negative** G−S_proxy difference in all 3 seeds, CI upper bound ≤ −0.01 bpb). The proxy does not start until PLAN_PASS (and any remaining uni sign-off) says so.

Incremental FLOPs are the load-bearing compute match for G vs C (and for reporting G vs S). Sunk 85M pretrain/SFT cost is always reported and is not replayed in C or S. G is run **first** per seed; C and S stop at **G’s logged FLOPs** from the same seed/ratio (`6 × n_params × presented_bytes` in G’s train log, tied embeddings counted once). Registered bytes are only an **upper bound**. If G stops early, C/S get that actual FLOP count and **never more** (a step is not started if it would exceed G’s log).

### 2.3 Frozen analyzer (`code/analyze.py`) [L]

The decision rule in §2.1–§2.2 (and the proxy rule in §9) is implemented by `code/analyze.py`. It is part of the frozen code in §10. Official analysis consumes the raw `eval_heldout.json` files written by the eval path (`primary_256.per_chunk_bpb`, window-major, 8 chunks per file, plus `file_tags`) and the per-grown-arm `productive.json` plus `fp_gate.json`.

Procedure, fail-closed:

1. Per file, mean bits/byte over that file’s 8 chunks; then the paired difference **G−C** (negative = grown better).
2. The mean over files of that difference must be **negative in all three official seeds** (11, 22, 33). A missing official seed is fail-closed. Official runs use **exactly** those seeds: any other seed in the inputs or the layout raises `AnalyzeError` (it is not averaged into the CI).
3. Per file, average the paired difference over those official seeds only; cluster-bootstrap the 32 file-level means, **B = 10,000**, seed **77701**, 95% percentile CI (`np.percentile` 2.5 / 97.5).
4. **ADVANCE** iff (2) holds, the CI upper bound is **≤ −0.01 bpb**, **and** the divergence / productive-block gate (`arm_pass`) **and** the FP gate pass for every official seed. For grown arms (`G`, `G_zeroall`, `G_signflip`, and the proxy grown arm) a missing or `None` `fp_pass` is a gate **FAIL**. “If present” remains allowed only for C and S. A missing or failed gate yields **NO_ADVANCE**, even if the statistical clauses would have passed.
5. Otherwise **NO_ADVANCE**.

**Non-official path.** `analyze.py --smoke` (alias `--non-official`) is a format-check path only. It may read non-official seeds (e.g. smoke seed 7) and **always** returns **NO_ADVANCE** with reason `non-official`. It is never an official verdict.

**G vs S** is computed with the same machinery and reported as **[H]** (`HEURISTIC_PASS` / `HEURISTIC_FAIL`). It never produces a growth ADVANCE.

**Proxy decision rule:** `analyze.py --proxy` applies the same stats and gates to **grown vs S_proxy** (total-FLOP claim) and returns `PROXY_ADVANCE` or `PROXY_NO_ADVANCE`. Official proxy runs are under the same exact-seed and grown-arm FP rules. A failed productive-block or FP gate still blocks the brute-force claim.

Non-finite `per_chunk_bpb` values, mismatched file sets or chunk counts, and a missing required eval JSON are fail-closed (`ANALYZE_FAIL`).

**Synthetic tests** (command: `python code/analyze.py --synthetic-tests`). Stored at `results/analyze_synthetic_tests.json`, sha256 `6157185ea21e57b012a2a52c15908b43e2288f00613cd7317f58362ada5cbd7c`. Suite **PASS**. Expected and actual verdicts:

| case | construction | expected | got | CI (point, [lo, hi]) |
|---|---|---|---|---|
| (a) clear ADVANCE | G−C = −0.05 in all 3 seeds, tight CI, gate passes | ADVANCE | ADVANCE | −0.050000 [−0.050000, −0.050000] |
| (b) sign flip in one seed | seeds 11 and 22 negative; seed 33 flipped positive | NO_ADVANCE | NO_ADVANCE | −0.016667 [−0.016667, −0.016667] |
| (c) just misses margin | all seeds negative, mean = −0.006, CI upper > −0.01 | NO_ADVANCE | NO_ADVANCE | −0.006000 [−0.006000, −0.006000] |
| (d) gate-fail | same data as (a), `arm_pass=false` | NO_ADVANCE | NO_ADVANCE | −0.050000 [−0.050000, −0.050000] |
| (e) stray seed 7 | otherwise clear ADVANCE plus seed 7 on the official path | AnalyzeError | AnalyzeError | n/a (raised before a verdict) |
| (f) missing FP gate | otherwise clear ADVANCE, G `fp_pass` is `None` | NO_ADVANCE | NO_ADVANCE | −0.050000 [−0.050000, −0.050000] |
| (g) noisy around margin | true mean −0.012 plus per-seed/per-file noise, RNG seed 424242; rule applied twice identically | rule-consistent (no fixed verdict) | ADVANCE | −0.012341 [−0.013044, −0.011659] |
| (g′) noisy clear ADVANCE | true mean −0.05 plus modest noise, RNG seed 434343 | ADVANCE | ADVANCE | −0.049398 [−0.049972, −0.048811] |
| proxy positive control | grown vs S_proxy, same stats, gate passes | PROXY_ADVANCE | PROXY_ADVANCE | −0.080000 [−0.080000, −0.080000] |

**Smoke-layout read** (not a scientific result; official seeds unused). `python code/analyze.py --layout results/smoke_proxy --ratio proxy --seeds 7 --smoke` parsed the real `eval_heldout.json` / `productive.json` written by `--smoke-proxy` (32 files × 8 chunks, identical G/C `file_tags`, `primary_256.per_chunk_bpb`). Output: `results/analyze_smoke.json`, sha256 `2e91276ba98fadc15dd418bbcfcb13972de6a951c69209550bfd80b0413e79b4`. Verdict **NO_ADVANCE**, reason `non-official` (required on this path). The same layout without `--smoke` raises `AnalyzeError` because seed 7 is present. Observed G−C file-mean 0.047437 bpb, 95% CI [0.045108, 0.049948] — a format/path check only.

---

## 3. Operator, ratios, LR, divergence, FP

### 3.1 Operator and ratios [L]

Depth identity: append residual blocks with `attn.wo` and `mlp.fc2` exactly 0 (in-projections copied from source `t mod old_L` with `load_state_dict(..., strict=True)` — a copy failure is fail-closed — or **exactly zero** in-projections in `G_zeroall`, not N(0, 1e-3)). LN gamma=1, beta=0 in the zero-all case. **`G_zeroall` barely learns** is a heuristic expectation **[H]**, not a load-bearing claim. In-projections start at exactly zero, so the new residual is zero before noise. Once symmetry-breaking noise makes `wo`/`fc2` nonzero, `fc1`/`Wv` (and the other in-projections) get gradients from step 1, so the branch can learn; if it does, that is reported as a result, not a failure. Ratio A: L=24, **170,141,184** params. Ratio B: L=36, **255,112,704**. Run A before B. Mirrors `G_zeroall` and `G_signflip` are reported regardless of outcome; `G_signflip` is not FP and does not abort on the FP gate; `G_zeroall` must pass the FP gate.

### 3.2 Fixed LR rule [L]

- **Old blocks** (indices `< old_L`) **and** shared tensors (`tok_emb` / tied `head`, `ln_final`): constant **lr = 1e-5**, weight decay 0.01. This is **exactly C’s learning rate**.
- **New blocks** (indices `≥ old_L`): the **pretrain schedule used by S** — peak **1.5e-4**, 200-step linear warmup, cosine decay to **0.1 × peak**, weight decay 0.1. The cosine length is the step count implied by the registered `target_bytes` budget (or the FLOP-implied step count for C/S). There is no 1e9 default; a missing budget is fail-closed.
- **C:** every parameter is in the old group (1e-5).
- **S:** every parameter is on the pretrain schedule (peak 1.5e-4, same warmup/decay). `old_L` is treated as 0.
- Optimizer: AdamW, β=(0.9, 0.95). Default train T=256, batch=1. No hyperparameter search.

### 3.3 Divergence and knockout gate [L]

Logged for every new block, on the frozen FP input set (concat sha256 `898d3770faeef1901f9d614a79c8491d3de9f644415d5d1e761ea39d48e833b7`):

- **Step 0** (after symmetry-breaking noise, before training): cosine-to-source and RMS of (a) residual-branch output `wo(·)+fc2(·)` and (b) **pre-output-projection** activations (inputs to `wo` and to `fc2`). These step-0 numbers are diagnostics of the init. **A gate that would pass at step 0 does not count.**
- **N_div** = `min(200, max(50, floor(0.2 × steps_in_arm)))`. Same metrics, plus comparison **to the same block at step 0**.

**RMS test (learned update vs noise) [L]:** a new block passes the RMS test at N_div iff

`rms(r_i(N_div) − r_i(0)) ≥ max(2.0 × rms(r_i(0)), 1e-4)`

where `r_i` is the residual-branch output on the FP input set. Passing this at step 0 is impossible by construction (diff = 0).

**Knockout test [L]:** on a frozen DEV subset — the first **16** windows of `dev_exp001.npy`, first **256** bytes of each (sha256 of the dev file `b995fb77c8bd2f29e9093d65cf1b57f8b31945da0d5e05d8e15b4011d18f9f51`) — the arm’s 256-chunk bits/byte is compared to the same metric after zeroing that new block’s `wo` and `fc2`. The block passes iff knockout **raises** loss by **≥ 1e-3 bits/byte**.

**Productive block:** RMS test **and** knockout both pass. **Arm-level gate:** fraction of new blocks productive **≥ 0.50**. Failure ⇒ NO_ADVANCE even if G vs C loss is better.

### 3.4 Function-preservation gate (torch, fail-closed) [L]

The NumPy check already measured max abs logit diff **0.0** for 12→13/24/36. That is not sufficient. **Each arm that claims FP** (`G`, `G_zeroall`) must run the gate **in the torch training code, on the box CPU in fp32, before noise**, on the same frozen FP input set, tolerance **1×10⁻⁵** abs max logit diff. Exceeding the tolerance **auto-aborts** the arm; no manual override. `G_signflip` is expected to fail and is not aborted for that reason; its initial delta is logged.

The gate is fail-closed on non-finite logits and on sequence-count or shape mismatch (it does not use Python `max`, which is fail-open on NaN: `max(0.0, nan) == 0.0`). Negative tests are in `code/fp_gate_tests.py`; output `results/fp_gate_negative_tests.json` (sha256 `7266c9c33d3c5f4460ee5ab05ee85b2ef5ddbdd451c5c600a5acb12742bb5be8`): NaN, Inf, length mismatch, shape mismatch, and empty lists all raise `FP_GATE_FAIL`; a finite identity pair and a proxy-size 2→4 identity grow both pass (max abs 0.0). **PASS = true.**

bf16 / GPU are out of scope for exp001. A future prereg that used them would need a measured identity baseline and a new tolerance; 1×10⁻⁵ would not be reused blindly.

---

## 4. Arms, sequential preflight, 0.9× abort [L]

Per ratio, per seed: **G first**, then **C**, **S**, **G_zeroall**, **G_signflip**. G (and the mirrors, which share G’s registered byte budget) is given `target_bytes` from `CPU_BOX_BUDGETS`. C and S read **G’s train log** for that seed and stop at G’s **logged FLOPs**, never more; registered bytes are only an upper bound (C therefore sees more bytes; S at the grown n_params matches G’s FLOPs with the same byte count if no early stop).

Arms run **one at a time**. Each arm is preceded by a **measured** per-arm preflight (a few fwd+bwd+AdamW steps on a throwaway same-size model) in `run_arm.py`. **PREFLIGHT_ABORT** is raised (fail-closed) if projected train wall **plus applicable reserves** > **0.9 ×** the 3,600 s cap (3,240 s), **or** if the conservative projected peak RSS > **0.9 ×** `MemAvailable` (from `/proc/meminfo`; a missing field is itself fail-closed). Peak RSS is the process high-water mark (`ru_maxrss`, kB) measured around those preflight steps. The full-run projection adds, conservatively, one extra parameter copy (growth / a second resident model), one-layer attention scores at full eval ctx (`n_head × ctx × ctx × 4` bytes), and four residual working buffers through depth (`ctx × d_model × n_layer × 16` bytes). A negative test with injected `MemAvailable = 1` kB must abort. No training starts after an abort.

Phase reserves are **inside** the 3,600 s pilot cap and are **enforced in `train_loop`** (K6): growth+FP+noise **300 s** (growth arms only) and eval + knockout + q3 hook **600 s** (every arm). The training wall passed into the loop is `wall_cap_s − reserves`. If eval needs more than its reserve, finish eval and mark over-cap; do not train further.

Consequently, for every arm, **training + applicable reserves ≤ 0.9 × 3,600 s**. C and S have a 600 s eval reserve, so C training must project to **≤ 2,640 s**. Growth arms have a 900 s reserve, so G training must project to **≤ 2,340 s**. Budgets in §8 are sized to those lines, with planning slack under the abort threshold.

---

## 5. Locked held-out and train corpus [L]

**Held-out (redrawn):** `/workspace/growth/data/heldout_exp001.bin`, 32×2048 B, **32 file clusters**, sha256 `fdaa73481604e76f71b7fca70c91d7369a90e4d396e8ebe80be882c8874dffb6`. Supersedes v1 `0c81811b2ef9e1061627180e6d24d49d50cbeed81bfa0b734c55a901702e9df4` (22 files; 9/32 had a 64-byte overlap with available pretrain-family trees). Composition: see §2.1.

**Corpus guards:** 13-gram vs the eval suite / antonyms / held-out A / names / concepts (v8 protocol). **Load-bearing corpus-vs-corpus guard: zero 64-byte overlap** with available v8-family pretrain trees and with locked train/dev. After redraw: **64-byte 0/32** vs those trees. **32-byte overlap remains a diagnostic: 21/32** windows still share a 32-byte substring with the pretrain-family trees (syntax/boilerplate, not a reject). 13-gram vs the large corpus is saturated (32/32) and is not a reject rule.

**Train / dev subset** (license-clean v8-family local trees; not the exact 350M-byte pin; datasets/gradio missing; no teacher data):

- `train_exp001.npy`: 7,973 × 2049 B = 16,336,677 B, sha256 `3ba0ac5bbb8f5056e1110f22eeb0976865db982aad79dd0b2379739759e7d507`
- `dev_exp001.npy`: 243 windows, sha256 `b995fb77c8bd2f29e9093d65cf1b57f8b31945da0d5e05d8e15b4011d18f9f51`
- 64-byte overlap vs held-out: **0** train, **0** dev.

---

## 6. 3-bit export hook [L]

After a finished arm, `code/export_hook.py` calls the existing converter (`/workspace/gwpub/code/convert.py`, kind `q3g64mse`) with identical settings for G and C (and S if exported). Report fp32 vs q3 bits/byte on the primary 256-byte chunks, and per-tensor rel. Frobenius split **old blocks 0–11 vs new blocks**. Descriptive, not an ADVANCE clause. Duplicated-row / shared g64-group check is recorded if any arm puts copies in the same tensor.

---

## 7. 850M feasibility (unchanged estimate) [H]

This remains a future-hardware note, not an exp001 plan. On an 8 GB 4060, **fp32 Adam will not fit** an ~850M model (~13.6 GB states); bf16 + 8-bit Adam + gradient checkpointing is a later question and is **out of scope here**. Unicorn currently has only ~3.5 GB free (foreign cosyvoice process). exp001 does not train 850M and does not use that GPU. Claims about 850M only after both Stage-1 ratios ADVANCE under this prereg, and only in a new prereg if the device changes.

---

## 8. Measured torch CPU preflight and caps [L]

`results/preflight_torch_cpu.json`, sha256 `97c259eca05fc868e235f0e18461c661279601236f93a4c6ccbf41a75c947874`.
Device: box CPU, **4 threads** (`THREAD_COUNT=4`; `torch.set_num_threads(4)`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`), fwd+bwd+AdamW, T=256 predicted bytes (input length 257), batch=1, 2 warmup + 5 timed steps, CUDA N/A. **Wall** = `time.perf_counter` over timed steps. **CPU-s** = `ru_utime + ru_stime` of this process over the same window. **Peak RSS** = Linux `ru_maxrss` (KB).

**Torch build** (same interpreter `/workspace/venv_wm/bin/python` that produced the preflight; recorded in `results/torch_build_cpu.json`, sha256 `b8f4558e28bdc00fbe04fedb96c0937ab3b0d7247ccfb5ee49cc78a1c0cbb5f0`):

- `torch.__version__` = **2.15.0+cpu** (`USE_CUDA=0`, COMMIT_SHA `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`)
- MKL: Intel oneAPI Math Kernel Library **2024.2** (Product Build 20240605); `torch.backends.mkl.is_available() = True`; `BLAS_INFO=mkl`, `LAPACK_INFO=mkl`
- oneDNN / MKL-DNN: Intel MKL-DNN **v3.13.3** (Git Hash `b38f97eb6f2173513112f15a0481750b8a66e901`); `USE_MKLDNN=ON`; `mkldnn.is_available() = True`, enabled
- OpenMP 4.5 (`USE_OPENMP=ON`); GCC 13.3; CPU capability **AVX512**; NNPACK enabled

| L | unique params | s/step wall | s/step CPU | CPU/wall | B/s (wall) | peak RSS |
|---|---|---|---|---|---|---|
| 12 (85M) | 85,169,664 | **0.696 s** | 2.749 s | 3.95 | **367.70** | 1,830,088 KB (~1.79 GiB) |
| 24 (170M) | 170,141,184 | **1.119 s** | 4.441 s | 3.97 | **228.70** | 3,373,408 KB (~3.22 GiB) |
| 36 (255M) | 255,112,704 | **1.606 s** | 6.379 s | 3.97 | **159.41** | 4,798,312 KB (~4.58 GiB) |

All three fit in the 15.6 GiB box with headroom. This **replaces** the unmeasured 6× NumPy-prefill heuristic (78 B/s at 85M).

**Derived caps** (pilot 3,600 s; hard 21,600 s). Abort if projected wall to finish the FLOP budget **> 0.9 × cap** (3,240 s / 19,440 s):

| L | pilot bytes | pilot steps | 0.9-cap bytes | hard bytes | hard steps |
|---|---|---|---|---|---|
| 12 | 1,323,714 | 5,170 | 1,191,342 | 7,942,286 | 31,024 |
| 24 | 823,323 | 3,216 | 740,990 | 4,939,939 | 19,296 |
| 36 | 573,870 | 2,241 | 516,483 | 3,443,222 | 13,450 |

**Why G cannot use its own 3,600 s byte cap on this CPU box [L].** Incremental FLOP match means C (L=12) must present `bytes_G × n_params_G / n_params_C` bytes. If G (L=24) used its raw pilot cap of 823,323 bytes, C would need ≈ 1,644,647 bytes, which exceeds C’s 3,600 s capacity (1,323,714 bytes). C would hit the wall cap before matching G’s FLOPs. The same holds for ratio B. Therefore G’s presented-byte budget is **bounded by C’s throughput**.

**Planning rule for CPU-box incremental match [L] (P1).** Training + applicable reserves must be ≤ 0.9 × 3,600 s = 3,240 s for **every** arm. C’s abort line is therefore **2,640 s** of training (3,240 − 600). G’s abort line is **2,340 s** (3,240 − 900). G’s presented bytes are the largest multiple of 256 such that C’s projected wall at FLOP match is **≤ 2,520 s** (~120 s slack under 2,640 s so a K5 re-measure cannot trip the abort). C and S stop at G’s logged FLOPs and **never exceed** them (a step that would overshoot is not taken); registered bytes are only an upper bound; the reserved train wall is a further stop. These budgets are frozen in `code/constants.py` as `CPU_BOX_BUDGETS` (and the proxy byte constants in §9) and in this prereg.

**Registered CPU-box pilot budgets** (from the measured rates above):

| ratio | G presented bytes (multiple of 256) | G steps | G projected wall | C bytes at FLOP match | C projected wall | S bytes (= G) |
|---|---|---|---|---|---|---|
| A (L=24) | **463,616** | 1,811 | 2,027 s | 926,153 | 2,519 s | 463,616 |
| B (L=36) | **309,248** | 1,208 | 1,940 s | 926,305 | 2,519 s | 309,248 |

These CPU-box budgets are the registered exp001 budgets. They are not to be rescaled onto a GPU inside this experiment. A different device would require a new prereg.

The CLI entry point is `code/run_arm.py` (see §10). `--smoke-proxy` exercises the full path on the proxy-size model and is **not** an arm.

---

## 9. Secondary full-pipeline proxy [L if started; own NO_ADVANCE]

Tiny GoldWorm layout, d=64, 4 heads, FFN×4, ctx=256. L=2: **115,328** params; L=4: **214,144**. Same grow operator, FP gate, split LR, seeds, cluster stats.

- **Pretrain** the L=2 proxy on the locked train set for a **fixed 2,000,000 presented bytes** (F0 = 6 × 115,328 × 2,000,000).
- Grow 2→4, FP gate, noise, continue for a registered post-grow byte count B1 (default also 2,000,000 unless PLAN_PASS changes only that number). F1 = 6 × 214,144 × B1.
- **C_proxy:** continue L=2 for incremental FLOPs = **G’s logged F1** (never more than the log; planned F1 is only an upper bound). Planned C bytes ≈ 3,714,048 if B1 completes in full (largest multiple of 256 ≤ 214,144/115,328 × 2,000,000).
- **S_proxy:** from-scratch L=4 with **actual total FLOPs F0+F1 from the pretrain and G logs**. Planned presented bytes = `(115328/214144)×2,000,000 + B1`. With B1=2,000,000 the exact match is 3,076,923.076… bytes; **register 3,076,864** (12,019 steps; the largest multiple of 256 that does not exceed the total-FLOP target) as an **upper bound** only. Residual FLOP fraction < 2×10⁻⁵.

These proxy byte counts finish in seconds on the box CPU and therefore already satisfy P1 (train + reserves ≪ 3,240 s). They are not scaled down further.

**Proxy decision rule / NO_ADVANCE [L for the brute-force claim only]:** the load-bearing proxy comparison is **grown vs S_proxy** (matched **total** FLOPs including the 2e6-byte pretrain). Same stats as §2.1, implemented by `code/analyze.py --proxy`: cluster bootstrap, seed 77701, B=10,000, per-file mean over seeds 11/22/33, **negative** difference (grown better) in all 3 seeds, CI upper bound **≤ −0.01 bpb**, and the productive-block gate must pass. Failure ⇒ `PROXY_NO_ADVANCE`, no growth-beats-brute-force claim. The proxy, when authorized, also runs on the box CPU in fp32. It does not start in this session (only the proxy-size smoke test in §10, and the analyzer’s read of those smoke JSONs in §2.3, have been run).

---

## 10. Frozen training code [L]

Implementation: `/workspace/growth/code/`. The frozen CLI `run_arm.py` is the single entry point for training arms and is exactly what runs (no post-freeze edits). Real arms require `--i-understand-this-starts-an-arm` and are refused if `--device` is not `cpu`. `--smoke-proxy` runs the shortened full path and is not an arm. Official post-arm analysis is `code/analyze.py` (§2.3), which is in this freeze.

**Fail-closed FP gate (PhaseMirror).** `grow.max_abs_logit_diff` checks `torch.isfinite` on both logit sets first, asserts equal sequence counts and identical shapes (`zip(..., strict=True)`), reduces with `torch.maximum` (not Python `max`), and the gate aborts with `if not (mx <= tol)`.

**Negative tests.** `code/fp_gate_tests.py` → `results/fp_gate_negative_tests.json` (sha256 `7266c9c33d3c5f4460ee5ab05ee85b2ef5ddbdd451c5c600a5acb12742bb5be8`). All required failure cases raised `FP_GATE_FAIL`; identity cases passed. Suite **PASS**.

**Complete run path.** `run_G` does: grow → FP gate (before noise) → noise → step-0 traces → **measured per-arm preflight** (K5) → train with `target_bytes` from the registered budget (K2), cosine length from that budget (K3, no 1e9), and `train_wall = wall_cap − reserves` (K6), with an N_div snapshot → `divergence_report` vs step 0 at N_div → per-new-block DEV knockout (ε = 1e-3 bpb) + productive-block report → held-out eval (256-byte primary + 2048 descriptive, cropped to model ctx when needed) → q3g64mse gap report. `run_C` and `run_S` do the same eval/export path, read **G’s logged FLOPs**, and never exceed them (K7). `G_zeroall` and `G_signflip` share this path (signflip skips the FP abort). Proxy runners share the same functions.

**Proxy-size smoke test** (not an arm; official seeds unused, seed 7; ~15 s wall on the box CPU). Command: `python code/run_arm.py --smoke-proxy`. Output: `results/smoke_proxy/summary.json`, sha256 `377d41376a85178a6be5a4d147c8c933d9fcba8ec1d3fc6c80a49d6714862945`. Result: `ok=true`. Per arm (path check, not a scientific result):

| arm | steps | FP gate | N_div | notes |
|---|---|---|---|---|
| proxy_pretrain | 8 | n/a | n/a | checkpoint written |
| G | 60 | pass (max abs 0.0) | captured at 50 | n_params 214,144; FLOPs 19,735,511,040 |
| C | 111 | n/a | n/a | FLOPs 19,662,962,688 **≤** G (never more) |
| S | 60 | n/a | n/a | FLOPs **equal** G 19,735,511,040 |
| G_zeroall | 16 | pass | captured | in-proj verified exactly zero; mode `identity_zero_all` |
| G_signflip | 16 | expected fail | captured | no FP abort |

K5: a synthetic 10 s cap with 900 s reserves raised **PREFLIGHT_ABORT** (projected train+reserves 902.28 s > 0.9×cap 9.00 s), recorded in the existing `--smoke-proxy` summary (not rerun for v2.4; unofficial load of those smoke checkpoints still succeeds under the new fail-closed loader). K7 early-stop: G stopped at 5 steps (FLOPs 1,644,625,920); C 1,594,294,272 ≤ G; S exactly 1,644,625,920. Held-out 256-byte eval and q3g64 gaps were written for every trained arm (official converter correctly refused the proxy SHA).

**RAM preflight negative test** (not an arm; `python code/run_arm.py --test-ram-abort`). Injected `MemAvailable = 1` kB, large wall cap so time cannot be the trigger. Raised **PREFLIGHT_ABORT** (projected peak RSS 342,277 kB > 0.9×MemAvailable 0.90 kB). Output: `results/smoke_proxy/ram_preflight_abort/preflight.json`, sha256 `d038f74d16e95525e5579cc2905bc7bc7dcf77307285e11987505756f9304ef4`.

**v8 checkpoint load check** (not an arm). `load_v8` and `run_arm.load_or_empty` (unofficial path) both loaded `upstream_v8_c0dc0f13/model.safetensors` (sha256 `8a7b5716ab340a672bc307772fd3e8387beee4793c62e1f256c31b8c70aea0a7`; 85,169,664 unique params, tied embeddings). A 32-token forward pass was finite. A planted unexpected key `not_a_real_param` raised. The existing smoke proxy-pretrain checkpoint (tok_emb only) also still loads. Output: `results/checkpoint_load_check.json`, sha256 `892eccf6a5f66c0ae1db01ab897e446bdaf895bbfc7d0efc4fbc02d0636f8e18`.

Manifest: `results/CODE_MANIFEST.json`, sha256 **`976c01bf3f77316b0cde98454f111a504bfe8252ac76646f55b9f97d94cf108a`**. Supersedes `89af0138f5e1a08ce19d7764ac1709db75d47e69bd1bddcb5f0fbb5f3ab82487` (the v2.3 freeze, archived at `archive/CODE_MANIFEST_v2.3_89af0138.json`). Code files (14) plus the negative-test, smoke, torch-build, preflight, analyzer-test, smoke-read, checkpoint-load, and RAM-abort artifacts:

| path | bytes | sha256 |
|---|---|---|
| `__init__.py` | 68 | `b56b272ab03ab7b80d4fd498f56f6f0058ef0d8342120c23e07fd09d59e9e392` |
| `analyze.py` | 31553 | `946862386cc8014d635c4b1989d727424cf26ec39840545bde9338ca0e05ee1c` |
| `constants.py` | 4256 | `b1a07bf731d45ba358808d22fa12f6e82689e34ad69c1c21a6c530172997c746` |
| `export_hook.py` | 4500 | `cf83a6ece7ea5d8f1a3432c28ca26a584965329b9de95309906e007eccbfbb9c` |
| `fp_gate_tests.py` | 4419 | `6e5de9a70474391df66709bc61c2c7fbeb913f8ddf3701fe9541d085d2829512` |
| `goldworm_v8.py` | 2590 | `4d2fa8ef07ef035bbe469cd72356c89485e6919ad52b8c78e7618cbf95f96d5d` |
| `grow.py` | 7344 | `f1f12417ee1ebcfa1d4230dbff9e3adb724d3232473b77f6d83e9038d562c2b1` |
| `gwlm_v6.py` | 6427 | `2aea5ab1313a8f4c85956ff560cbf9470e7930b034f16ae2f84ab3333e1abf55` |
| `metrics.py` | 11069 | `3c16e88a791490179899b7831fd648f62a2cfd8c17bbb6cacaa715405b78445a` |
| `model_io.py` | 5556 | `90c51cd75887407a701b186868a46fd670600933db1433cd98b8003848672f8c` |
| `preflight.py` | 7309 | `301e234224f32ac94a878ec25188cb3ae4043f218d044ab3d92ca27aec319b36` |
| `run_arm.py` | 37277 | `831bc1ed12e8382ae344b57671b156e1db0c2bc6198aa290a55ea813710a7bc6` |
| `run_gpu_preflight.sh` | 542 | `4147edcbccd2fcec6513049bbfc4da704200bcadc7d7e4f1cbba4a5d7ab2e077` |
| `train.py` | 7679 | `f27040d6e995bb74f02b95e59d76a03d8ae0a8a436f20e7aa7650d2349d5dfd0` |

Analyzer artifacts in the same manifest: `results/analyze_synthetic_tests.json` (8825 B, sha256 `6157185ea21e57b012a2a52c15908b43e2288f00613cd7317f58362ada5cbd7c`) and `results/analyze_smoke.json` (8973 B, sha256 `2e91276ba98fadc15dd418bbcfcb13972de6a951c69209550bfd80b0413e79b4`). New this freeze: `results/checkpoint_load_check.json` (1186 B, sha256 `892eccf6a5f66c0ae1db01ab897e446bdaf895bbfc7d0efc4fbc02d0636f8e18`) and `results/smoke_proxy/ram_preflight_abort/preflight.json` (775 B, sha256 `d038f74d16e95525e5579cc2905bc7bc7dcf77307285e11987505756f9304ef4`). Previous artifacts (FP negative tests, smoke summary, torch build, CPU preflight) are unchanged from the v2.2 / v2.3 freeze.

Independent code review is required **before the first arm**. After review, any code change is an erratum and a new manifest.

---

## 11. Lean, prior FP (NumPy), licenses

Lean 4.32.1, `lean/FpGrowth.lean`, axioms propext only, algebra not floats. NumPy FP check: 0.0 max abs logit diff on 12→13/24/36 (`results/fp_check.json`, sha256 `d59514533c691783cdfec9c8499c7c15ff4a3e820ba3cfb8f54149b4d11a3acc`). Licenses: model MIT; train BSD-3/Apache as in the train manifest; no distillation in exp001.

---

## 12. Unresolved / PLAN_PASS

1. **Independent review** of this re-frozen code in §10 (P1 and K4–K7 remain closed; uni’s two fail-open analyzer points, unexpected checkpoint keys, and the registered RAM preflight abort are now in the freeze), then first arm **G ratio A** only, on the box CPU in fp32, via `code/run_arm.py`.
2. **GPU / bf16** are out of scope for exp001 (unicorn has ~3.5 GB free under a foreign cosyvoice process). No GPU preflight will be run under this prereg.
3. Official proxy start still waits for PLAN_PASS (and uni, if still required). Proxy byte budgets still satisfy P1 without scaling (they finish in seconds). The proxy-size smoke test has already executed every arm type plus the K5 abort and K7 early-stop paths, and is not an arm. `analyze.py --smoke` has been run on those smoke JSONs as a format check only (§2.3). The full `--smoke-proxy` path was not rerun for v2.4; the existing smoke checkpoints still load under the fail-closed loader, and the new RAM abort was tested standalone.
4. Official eval1/eval2 files and the exact 350M-byte pinned corpus are still absent; the locked subset and 16-task secondary stand.

No training arm has been started.
