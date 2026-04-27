from __future__ import annotations

import argparse
import pandas as pd
import matplotlib.pyplot as plt

from constants import RESULTS, FIGURES, LABEL_NAMES


def lexical_overlap(a: str, b: str) -> float:
    sa = set(str(a).lower().split())
    sb = set(str(b).lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--model", default="model")
    args = p.parse_args()
    df = pd.read_csv(args.predictions)
    df["correct"] = df["y_true"] == df["y_pred"]
    df["overlap"] = [lexical_overlap(a, b) for a, b in zip(df["sentence1"], df["sentence2"])]
    df["true_label_name"] = df["y_true"].map(LABEL_NAMES)
    df["pred_label_name"] = df["y_pred"].map(LABEL_NAMES)

    RESULTS.mkdir(exist_ok=True, parents=True)
    FIGURES.mkdir(exist_ok=True, parents=True)
    failures = df[~df["correct"]].sort_values("overlap", ascending=False).head(30)
    failures.to_csv(RESULTS / f"{args.model}_top_failure_cases.csv", index=False)

    summary = df.groupby("true_label_name").agg(
        n=("correct", "size"),
        accuracy=("correct", "mean"),
        mean_overlap=("overlap", "mean"),
    ).reset_index()
    summary.to_csv(RESULTS / f"{args.model}_error_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df[df.correct]["overlap"], alpha=0.6, label="correct")
    ax.hist(df[~df.correct]["overlap"], alpha=0.6, label="wrong")
    ax.set_xlabel("Lexical overlap between sentence1 and sentence2")
    ax.set_ylabel("Count")
    ax.set_title(f"Error Analysis: {args.model}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / f"{args.model}_overlap_error_analysis.png", dpi=200)
    print(summary)


if __name__ == "__main__":
    main()
