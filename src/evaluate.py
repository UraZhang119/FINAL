from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from constants import FIGURES, LABEL_NAMES, RESULTS


def compute_metrics(y_true, y_pred, prefix: str = "") -> Dict[str, float]:
    return {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def save_classification_report(y_true, y_pred, model_name: str) -> None:
    RESULTS.mkdir(exist_ok=True, parents=True)
    names = [LABEL_NAMES.get(i, str(i)) for i in sorted(set(y_true) | set(y_pred))]
    report = classification_report(y_true, y_pred, target_names=names, zero_division=0)
    (RESULTS / f"{model_name}_classification_report.txt").write_text(report, encoding="utf-8")


def save_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    FIGURES.mkdir(exist_ok=True, parents=True)
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels([LABEL_NAMES.get(i, str(i)) for i in labels], rotation=45, ha="right")
    ax.set_yticklabels([LABEL_NAMES.get(i, str(i)) for i in labels])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix: {model_name}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(FIGURES / f"{model_name}_confusion_matrix.png", dpi=200)
    plt.close(fig)


def append_metrics(row: Dict, path: Path = RESULTS / "metrics.csv") -> None:
    RESULTS.mkdir(exist_ok=True, parents=True)
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, new], ignore_index=True)
        df = df.drop_duplicates(subset=["model", "split", "setting"], keep="last")
    else:
        df = new
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="CSV with columns y_true,y_pred")
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--setting", default="default")
    args = parser.parse_args()
    df = pd.read_csv(args.predictions)
    y_true, y_pred = df["y_true"], df["y_pred"]
    row = {"model": args.model, "split": args.split, "setting": args.setting, **compute_metrics(y_true, y_pred)}
    append_metrics(row)
    save_classification_report(y_true, y_pred, args.model)
    save_confusion_matrix(y_true, y_pred, args.model)
    print(row)


if __name__ == "__main__":
    main()
