# CSE 144 Final Project

100-class image classification using DINOv3 (Meta, 2025) with transfer learning.

## Setup

```bash
pip install -r requirements.txt
hf auth login   # paste your HuggingFace token when prompted
```

## Training

Open `project-DINO.ipynb` and run all cells top to bottom. Change `SEED` in cell 2 for each run:

| Run | SEED |
|-----|------|
| 1   | 44   |
| 2   | 45   |
| 3   | 46   |
| 4   | 47   |
| 5   | 48   |

Restart the kernel between runs. Each run saves a checkpoint: `dino_best_model_{SEED}.pth`

Training takes roughly 15-25 minutes per run on Apple Silicon (MPS).

## Inference

Once all 5 checkpoints exist, run cell 14 in `project-DINO.ipynb`. It loads all 5 models, runs 6 forward passes per test image (1 clean + 5 augmented), averages the predictions, and writes `submission.csv`.

Upload `submission.csv` to Kaggle.

## Model Weights

Download pretrained weights from Google Drive: [link]

Place the `.pth` files in the root project directory before running inference.

## Kaggle Leaderboard

![Kaggle leaderboard](leaderboard_screenshot.png)

## Requirements

- Python 3.10+
- PyTorch 2.0+
- See `requirements.txt` for full list
