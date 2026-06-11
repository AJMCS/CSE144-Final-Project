# CSE 144 Final Project

100-class image classification using a frozen-backbone ensemble of Perception Encoder (PE) and SigLIP 2, plus SigLIP zero-shot text matching.

## Setup

```bash
pip install -r requirements.txt
hf auth login   # paste your HuggingFace token when prompted
```

## Training

Two notebooks each train a small linear head on top of a frozen vision backbone:

| Notebook | Backbone | Output |
|----------|----------|--------|
| `pe_single_h52.ipynb` | Perception Encoder ViT-L/336 (`vit_pe_core_large_patch14_336.fb`) | `PE_model_h52.pth`, `pe_probs_h52.npy` |
| `project-SigLIP2-head.ipynb` | SigLIP 2 ViT-SO400M/378 (`vit_so400m_patch14_siglip_378.webli`) | `SigLIP2_full_model_h52.pth`, `siglip2_probs_h52.npy`, `text_probs_h52.npy` |

Both backbones stay frozen; only a dropout + linear head trains, so each notebook runs in just a few minutes on a GPU.

Run `pe_single_h52.ipynb` first, then `project-SigLIP2-head.ipynb`. Open each and run all cells top to bottom.

`project-PE-head.ipynb` is an optional 3-seed PE ensemble variant for pushing accuracy further.

## Inference

The last cell of `project-SigLIP2-head.ipynb` combines all three prediction sets into `submission.csv`:

```
combined = 0.1 * pe_probs + 0.5 * siglip2_probs + 0.4 * text_probs
```

PE leads, SigLIP 2 vision adds diversity, and SigLIP zero-shot text matching breaks ties.

Upload `submission.csv` to Kaggle.

## Model Weights

Download pretrained weights from Google Drive: [link](https://drive.google.com/drive/folders/1elpQwSoS2eV-4HmJo0Ouptkmvxp0wykk?usp=drive_link)

Place the `.pth` files in the root project directory before running inference.

## Kaggle Leaderboard

![Kaggle leaderboard](/kaggle_leaderboard.png)

## Requirements

- Python 3.10+
- PyTorch 2.0+
- See `requirements.txt` for full list
