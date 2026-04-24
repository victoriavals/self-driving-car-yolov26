# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Self-driving car **instance segmentation** using **YOLOv26** (Ultralytics, released Jan 2026).
Urban street scenes — vehicles, pedestrians, roads, buildings — segmented from dashcam images.

## Stack

- Python 3.12.10
- `ultralytics >= 8.4.41` (YOLOv26 support)
- PyTorch with CUDA 12.4 (`pip install torch --index-url https://download.pytorch.org/whl/cu124`)
- OpenCV, NumPy, Pillow, scikit-learn, matplotlib

## Notebook Pipeline (run in order)

| # | File | Input | Output |
|---|---|---|---|
| 01 | [notebooks/01_data_audit.ipynb](notebooks/01_data_audit.ipynb) | Raw images + masks | `audit_results.json` |
| 02 | [notebooks/02_preprocessing.ipynb](notebooks/02_preprocessing.ipynb) | Raw data + `audit_results.json` | `dataset/` dir + `dataset.yaml` |
| 03 | [notebooks/03_training.ipynb](notebooks/03_training.ipynb) | `dataset/` | `runs/segment/yolo26s_run1/weights/best.pt` |
| 04 | [notebooks/04_evaluation.ipynb](notebooks/04_evaluation.ipynb) | `best.pt` + test set | mAP, Dice Score, `evaluation_results.json` |
| 05 | [notebooks/05_hyperparameter_tuning.ipynb](notebooks/05_hyperparameter_tuning.ipynb) | `dataset/` | `runs/segment/yolo26s_final/weights/best.pt` |

## Raw Dataset Structure

```
images_prepped_train-*/images_prepped_train/        # 398 RGB PNG images (480×360)
images_prepped_test-*/images_prepped_test/           # 101 RGB PNG images (480×360)
annotations_prepped_train-*/annotations_prepped_train/  # 367 grayscale masks
annotations_prepped_test-*/annotations_prepped_test/    # 101 grayscale masks
```

**Key data facts:**
- Grayscale masks: pixel value = semantic class ID (0 = background)
- 398 train images but only 367 masks → 31 orphan images (automatically skipped in preprocessing)
- Preprocessing uses only matched image-mask pairs; 80/20 train/val split from matched pairs
- Test set (101 pairs) is held out and never used during training

## Annotation → YOLO Label Conversion

`02_preprocessing.ipynb` converts grayscale masks to YOLO segmentation polygon format:
- Each foreground class contour → `<class_id> x1 y1 x2 y2 ... xn yn` (normalized 0–1)
- Background (pixel 0) is excluded from label files
- Contours smaller than 100 px² are filtered as noise
- Output lives in `dataset/labels/{train,val,test}/`

**Important**: After running `01_data_audit.ipynb`, update the `"class_names"` field in `audit_results.json`
with real semantic names before running notebook 02. Pixel value → class name must be set manually.

## Model

- Default: `yolo26s-seg.pt` (Small) — best balance for ~294 training images
- YOLOv26 key features: NMS-free inference, Progressive Loss Balancing, Small-Target-Aware Label Assignment
- Training uses AMP (mixed precision), AdamW optimizer, cosine LR decay

## Evaluation Metrics

- **mAP@50 / mAP@50-95**: from `model.val()` on test split
- **Dice Score Coefficient**: custom implementation — YOLO instance masks merged into semantic mask, compared pixel-wise vs. GT mask
- **Dice Loss**: `1 - mean_DSC`
- Results saved to `evaluation_results.json`

## Hyperparameter Tuning

Two strategies in `05_hyperparameter_tuning.ipynb`:
1. **Auto-tune** (`model.tune()`): evolutionary search over ~25 params, 50 trials × 30 epochs
2. **Manual grid search**: sweeps `lr0`, `weight_decay`, `mosaic` — faster for targeted investigation

Best config is used for final full-epoch retraining (`yolo26s_final`).

## Key Constants (shared across notebooks)

| Constant | Value | Where set |
|---|---|---|
| `IMG_H, IMG_W` | 360, 480 | from `audit_results.json` |
| `CLASS_MAP` | pixel_val → YOLO class ID | `02_preprocessing.ipynb` |
| `CLASS_NAMES` | loaded from `dataset.yaml` | all eval/tune notebooks |
| `RUN_NAME` | `yolo26s_run1` | `03_training.ipynb` (must match in `04`) |
| `CONF_THRESHOLD` | 0.25 | `04_evaluation.ipynb` |
