# Project 2 – Rare Defects

A small PyTorch classifier demonstrates why accuracy is misleading when faulty items are rare. In the scikit-learn digits dataset, digit `9` is treated as a faulty part and all other digits as OK.

## Objectives
- Create a stratified 70/30 train/test split.
- Retain only 20 faulty (`9`) examples in the training set.
- Compare an always-OK baseline, an unweighted neural network and a class-weighted neural network.
- Report accuracy, faulty-class recall, faulty-class precision and confusion matrices.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python rare_defects.py
```
Generated files are saved in `artifacts/`: `results.csv`, `REPORT.md` and `confusion_matrices.png`.

## Model
A fully connected network: 64 input values → 32 ReLU units → 2 output classes. Pixels are divided by 16. The weighted loss uses balanced class weights: total training examples / (2 × class count).

## Limitations
- This is a teaching simulation using 8×8 handwritten digit images, not real factory photographs.
- Only 20 faulty examples are available during training.
- A single train/test split may produce variable results on other seeds.
- Thresholds and costs should be chosen with domain experts; false alarms and missed defects have different operational costs.
