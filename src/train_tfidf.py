from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from constants import DATA_PROCESSED, MODELS, RESULTS
from evaluate import append_metrics, compute_metrics, save_classification_report, save_confusion_matrix


def load_processed():
    return {s: pd.read_csv(DATA_PROCESSED / f"{s}.csv") for s in ["train", "validation", "test"]}


def build_model(model_name: str, max_features: int, ngram_range=(1, 2), C: float = 1.0):
    if model_name == "dummy":
        return DummyClassifier(strategy="most_frequent")
    clf = LogisticRegression(max_iter=1000, C=C, class_weight="balanced", n_jobs=-1)
    if model_name == "svm":
        clf = LinearSVC(C=C, class_weight="balanced")
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, lowercase=True)),
        ("clf", clf),
    ])


def train_and_eval(model_name: str, max_features: int, ngram_max: int, C: float, setting: str):
    splits = load_processed()
    train, val, test = splits["train"], splits["validation"], splits["test"]
    model = build_model(model_name, max_features=max_features, ngram_range=(1, ngram_max), C=C)
    start = time.time()
    model.fit(train["text"], train["label"])
    train_time = time.time() - start

    MODELS.mkdir(exist_ok=True, parents=True)
    RESULTS.mkdir(exist_ok=True, parents=True)
    joblib.dump(model, MODELS / f"{model_name}_{setting}.joblib")

    for split_name, df in [("validation", val), ("test", test)]:
        t0 = time.time()
        pred = model.predict(df["text"])
        inference_time = time.time() - t0
        out_pred = pd.DataFrame({
            "guid": df.get("guid", pd.Series(range(len(df)))),
            "sentence1": df["sentence1"],
            "sentence2": df["sentence2"],
            "y_true": df["label"],
            "y_pred": pred,
        })
        pred_path = RESULTS / f"{model_name}_{setting}_{split_name}_predictions.csv"
        out_pred.to_csv(pred_path, index=False)
        row = {
            "model": model_name,
            "split": split_name,
            "setting": setting,
            "train_time_sec": train_time,
            "inference_time_sec": inference_time,
            "examples_per_sec": len(df) / max(inference_time, 1e-9),
            "max_features": max_features,
            "ngram_max": ngram_max,
            "C": C,
            **compute_metrics(df["label"], pred),
        }
        append_metrics(row)
        save_classification_report(df["label"], pred, f"{model_name}_{setting}_{split_name}")
        save_confusion_matrix(df["label"], pred, f"{model_name}_{setting}_{split_name}")
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["dummy", "logreg", "svm"], default="logreg")
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--setting", default="default")
    args = parser.parse_args()
    train_and_eval(args.model, args.max_features, args.ngram_max, args.C, args.setting)


if __name__ == "__main__":
    main()
