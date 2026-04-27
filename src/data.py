from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from constants import DATA_RAW, DATA_PROCESSED, RANDOM_SEED, SPLIT_URLS


def clean_text(text: str) -> str:
    """Basic Romanian-compatible text cleaning while preserving diacritics."""
    text = str(text).replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def download_ronli(force: bool = False) -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for split, url in SPLIT_URLS.items():
        out = DATA_RAW / f"{split}.json"
        if out.exists() and not force:
            print(f"[skip] {out} already exists")
            continue
        print(f"[download] {url}")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        out.write_bytes(r.content)
        print(f"[saved] {out} ({out.stat().st_size/1e6:.2f} MB)")


def _read_json_records(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Be permissive in case a user exports as {"data": [...]}.
        data = data.get("data", data.get("examples", []))
    df = pd.DataFrame(data)
    rename = {}
    if "premise" in df.columns and "sentence1" not in df.columns:
        rename["premise"] = "sentence1"
    if "hypothesis" in df.columns and "sentence2" not in df.columns:
        rename["hypothesis"] = "sentence2"
    if "gold_label" in df.columns and "label" not in df.columns:
        rename["gold_label"] = "label"
    df = df.rename(columns=rename)
    required = {"sentence1", "sentence2", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns {missing} in {path}")
    df = df[[c for c in ["guid", "sentence1", "sentence2", "label"] if c in df.columns]].copy()
    df["sentence1"] = df["sentence1"].map(clean_text)
    df["sentence2"] = df["sentence2"].map(clean_text)
    df["label"] = df["label"].astype(int)
    df["text"] = df["sentence1"] + " [SEP] " + df["sentence2"]
    return df.dropna(subset=["sentence1", "sentence2", "label"])


def load_splits(raw_dir: Path = DATA_RAW, sample: bool = False) -> Dict[str, pd.DataFrame]:
    if sample:
        raw_dir = Path(__file__).resolve().parents[1] / "data" / "sample"
    paths = {split: raw_dir / f"{split}.json" for split in ["train", "validation", "test"]}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset files. Run `python src/data.py --download` or use `--sample`. Missing: "
            + ", ".join(missing)
        )
    return {split: _read_json_records(path) for split, path in paths.items()}


def make_course_scale_subset(
    n_train: int = 6000,
    n_val: int = 1000,
    n_test: int = 1000,
    sample: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Create smaller stratified splits for fast course-scale experiments."""
    splits = load_splits(sample=sample)
    out = {}
    for name, n in [("train", n_train), ("validation", n_val), ("test", n_test)]:
        df = splits[name]
        if len(df) <= n:
            out[name] = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        else:
            out[name], _ = train_test_split(
                df,
                train_size=n,
                stratify=df["label"],
                random_state=RANDOM_SEED,
            )
            out[name] = out[name].reset_index(drop=True)
    return out


def save_processed_splits(splits: Dict[str, pd.DataFrame]) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for split, df in splits.items():
        out = DATA_PROCESSED / f"{split}.csv"
        df.to_csv(out, index=False)
        print(f"[saved] {out} shape={df.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download official RoNLI JSON files")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sample", action="store_true", help="Use bundled tiny sample data")
    parser.add_argument("--n-train", type=int, default=6000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    args = parser.parse_args()
    if args.download:
        download_ronli(force=args.force)
    splits = make_course_scale_subset(args.n_train, args.n_val, args.n_test, sample=args.sample)
    save_processed_splits(splits)


if __name__ == "__main__":
    main()
