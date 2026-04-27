from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample-mode", action="store_true")
    args = p.parse_args()
    py = sys.executable
    if args.sample_mode:
        run([py, "src/data.py", "--sample", "--n-train", "24", "--n-val", "12", "--n-test", "12"])
    run([py, "src/train_tfidf.py", "--model", "dummy", "--setting", "majority"])
    run([py, "src/train_tfidf.py", "--model", "logreg", "--setting", "tfidf_10k", "--max-features", "10000", "--ngram-max", "2", "--C", "1.0"])
    run([py, "src/train_tfidf.py", "--model", "logreg", "--setting", "tfidf_30k_tuned", "--max-features", "30000", "--ngram-max", "2", "--C", "2.0"])
    run([py, "src/train_mlp.py", "--setting", "tfidf_mlp", "--epochs", "5", "--max-features", "3000"])
    run([py, "src/error_analysis.py", "--predictions", "results/logreg_tfidf_30k_tuned_test_predictions.csv", "--model", "logreg_tfidf_30k_tuned"])


if __name__ == "__main__":
    main()
