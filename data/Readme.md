# Dataset

This project uses the **HAM10000 (Human Against Machine with 10000 training images)** dataset for skin lesion classification.

The dataset is not included in this repository because of its size and licensing considerations.

## Download

Kaggle: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

## Expected directory structure

```text
data/
├── raw/
└── processed/
    ├── train/
    ├── val/
    └── test/
```

Use `src/prepare_dataset.py` to create the processed train/validation/test split from the raw dataset.
`
