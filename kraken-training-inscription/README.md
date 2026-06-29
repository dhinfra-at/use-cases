# Kraken HTR cluster demo — Latin stone inscriptions

A small, self-contained **JupyterHub** use case for trying [Kraken](https://kraken.re)
HTR training on the cluster. You fine-tune a text-recognition model for Latin stone
inscriptions and compare four training recipes — in a few minutes on a GPU.

## Quickstart
1. Open this folder in **JupyterHub** (request a **GPU** session for speed; CPU works too, slower).
2. Open **`htr_cluster_demo.ipynb`** and run the cells top to bottom.

The notebook uses Kraken, fetches the base model, and trains — nothing else to set up.

## What's inside
- `htr_cluster_demo.ipynb` — the walkthrough: setup → data → 4 recipes → comparison.
- `data/real/`  — **50** annotated EDH inscriptions (image + PAGE-XML).
- `data/synth/` — **500** synthetic stone images (image + PAGE-XML).

Everything the notebook generates (compiled datasets, trained models) is written into
`arrows/`, `models/`, … and is git-ignored.

## The four recipes
Each fine-tunes the same base model and is **validated on real inscriptions only**
(synthetic data is used for training, never for scoring):

| recipe | training data |
|---|---|
| **real only** | 40 real |
| **pretrain → finetune** | pretrain on 500 synth, then finetune on 40 real |
| **mixed** | 40 real + 500 synth together |
| **weighted mixed** | 40 real ×5 + 500 synth |

Deliberately *plain & simple*: one train/val split, no cross-validation.

## Licence
- **Code / notebook:** MIT — see [`LICENSE`](LICENSE).
- **Data:** the 50 real inscriptions are © *Epigraphic Database Heidelberg*, reused under
  **CC BY-SA 4.0**; the 500 synthetic images are released under the same licence.
  See [`data/LICENSE.txt`](data/LICENSE.txt).

## Credits
Real inscription data: [Epigraphic Database Heidelberg](https://edh.ub.uni-heidelberg.de)
(CC BY-SA 4.0). Base model: CATMuS-Print Large. Engine: [Kraken](https://kraken.re).
