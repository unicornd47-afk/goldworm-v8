# GoldWorm v8

Private snapshot of GoldWorm v8 code and training scripts: the v8.x world-model experiments and the byte-level language-model training code.

**Status:** source and experiment plans. This repository is private and is not a public release. Model weights, checkpoints and training datasets are not included.

## Licence

No licence file is attached at the repository root. The world-model tree under `worldmodel/` had no licence in the source we copied.

The language-model snapshot under `lm/upstream/` already carried an **MIT** licence in that source tree (`lm/upstream/LICENSE`). That licence applies to that subtree as found, not to the rest of this repository. Business Source License 1.1 was **not** present in these sources, so it is not added here.

## What is in this snapshot

| Path | Role |
| --- | --- |
| `worldmodel/v8.1/` | World-model v8.1 code, plans, configs, hashes, and small JSON results |
| `worldmodel/v8.2/` | World-model v8.2, including errata |
| `worldmodel/v8.3/` | Current world-model v8.3 code, plans, patches, probe notes |
| `lm/exp001-code/` | Training and analysis scripts used for the exp001 growth run |
| `lm/upstream/` | Upstream v8 language-model source and training scripts (no weights) |
| `lm/PREREG_exp001_v2.4.md` | Preregistration text for that experiment |

## Deliberately excluded

- Model weights and checkpoints (`*.pt`, `*.pth`, `*.safetensors`, `*.bin`, `*.gguf`, `*.ckpt`)
- Large arrays and datasets (`*.npy`, `*.npz`, held-out/train binaries)
- Hugging Face caches and third-party licence dumps next to the upstream checkout
- Virtualenvs and `__pycache__`

JSON ledgers and small result sidecars that are not weight files are kept. `.npy` / `.npz` run-output arrays are not.
