# Setup Instructions

## Environment

Recommended: Python 3.10 or 3.11. Python 3.12 may also work with recent package versions.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Smoke Test

This repository includes a tiny sample dataset to verify that the code runs.

```bash
python src/data.py --sample --n-train 24 --n-val 12 --n-test 12
python src/run_all.py --sample-mode
```

Expected outputs:

```text
results/metrics.csv
figures/*confusion_matrix.png
figures/*training_curve.png
```

## Real RoNLI Data

Download the official RoNLI data:

```bash
python src/data.py --download --n-train 6000 --n-val 1000 --n-test 1000
```

This creates course-scale CSV splits in `data/processed/`.

## Baseline Experiments

```bash
python src/train_tfidf.py --model dummy --setting majority
python src/train_tfidf.py --model logreg --setting tfidf_10k --max-features 10000 --ngram-max 2 --C 1.0
python src/train_tfidf.py --model logreg --setting tfidf_30k_tuned --max-features 30000 --ngram-max 2 --C 2.0
```

## Custom MLP

```bash
python src/train_mlp.py --setting tfidf_mlp --epochs 10 --max-features 5000
```


## Error Analysis

```bash
python src/error_analysis.py --predictions results/logreg_tfidf_30k_tuned_test_predictions.csv --model logreg_tfidf_30k_tuned
```
