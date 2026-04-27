from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"

LABEL_NAMES = {
    0: "entailment",
    1: "contradiction",
    2: "neutral_related",
    3: "neutral_unrelated",
}

# The original repo exposes these files in dataset/datasets/.
RONLI_RAW_BASE = "https://raw.githubusercontent.com/Eduard6421/RONLI/main/dataset/datasets"
SPLIT_URLS = {
    "train": f"{RONLI_RAW_BASE}/train.json",
    "validation": f"{RONLI_RAW_BASE}/validation.json",
    "test": f"{RONLI_RAW_BASE}/test.json",
}

RANDOM_SEED = 372
