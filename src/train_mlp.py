from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from constants import DATA_PROCESSED, FIGURES, MODELS, RESULTS, RANDOM_SEED
from evaluate import append_metrics, compute_metrics, save_classification_report, save_confusion_matrix


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_processed():
    return {s: pd.read_csv(DATA_PROCESSED / f"{s}.csv") for s in ["train", "validation", "test"]}


def make_loader(X, y, batch_size: int, shuffle: bool):
    x_tensor = torch.tensor(X.toarray() if hasattr(X, "toarray") else X, dtype=torch.float32)
    y_tensor = torch.tensor(np.asarray(y), dtype=torch.long)
    return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)


def evaluate_model(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            labels.extend(yb.numpy().tolist())
    return np.array(labels), np.array(preds)


def plot_curves(history, setting: str):
    FIGURES.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss"], label="train loss")
    ax.plot(history["val_macro_f1"], label="val macro F1")
    ax.set_xlabel("Epoch")
    ax.set_title(f"MLP Training Curve ({setting})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / f"mlp_{setting}_training_curve.png", dpi=200)
    plt.close(fig)


def train_mlp(args):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    splits = load_processed()
    train, val, test = splits["train"], splits["validation"], splits["test"]

    vectorizer = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, args.ngram_max), lowercase=True)
    X_train = vectorizer.fit_transform(train["text"])
    X_val = vectorizer.transform(val["text"])
    X_test = vectorizer.transform(test["text"])

    train_loader = make_loader(X_train, train["label"], args.batch_size, True)
    val_loader = make_loader(X_val, val["label"], args.batch_size, False)
    test_loader = make_loader(X_test, test["label"], args.batch_size, False)

    num_classes = int(max(train["label"].max(), val["label"].max(), test["label"].max()) + 1)
    model = MLPClassifier(X_train.shape[1], args.hidden_dim, num_classes, args.dropout).to(device)
    class_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=train["label"].to_numpy())
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=1, factor=0.5)

    history = {"train_loss": [], "val_macro_f1": []}
    best_f1, best_state, patience_left = -1, None, args.patience
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())
        y_val, p_val = evaluate_model(model, val_loader, device)
        val_f1 = compute_metrics(y_val, p_val)["macro_f1"]
        scheduler.step(val_f1)
        history["train_loss"].append(float(np.mean(losses)))
        history["val_macro_f1"].append(val_f1)
        print(f"epoch={epoch+1} train_loss={history['train_loss'][-1]:.4f} val_macro_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered")
                break
    train_time = time.time() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    plot_curves(history, args.setting)

    MODELS.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), MODELS / f"mlp_{args.setting}.pt")
    joblib.dump(vectorizer, MODELS / f"mlp_{args.setting}_tfidf.joblib")

    for split_name, df, loader in [("validation", val, val_loader), ("test", test, test_loader)]:
        t0 = time.time()
        y_true, y_pred = evaluate_model(model, loader, device)
        inference_time = time.time() - t0
        pd.DataFrame({
            "guid": df.get("guid", pd.Series(range(len(df)))),
            "sentence1": df["sentence1"],
            "sentence2": df["sentence2"],
            "y_true": y_true,
            "y_pred": y_pred,
        }).to_csv(RESULTS / f"mlp_{args.setting}_{split_name}_predictions.csv", index=False)
        row = {
            "model": "mlp",
            "split": split_name,
            "setting": args.setting,
            "device": str(device),
            "train_time_sec": train_time,
            "inference_time_sec": inference_time,
            "examples_per_sec": len(df) / max(inference_time, 1e-9),
            "max_features": args.max_features,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            **compute_metrics(y_true, y_pred),
        }
        append_metrics(row)
        save_classification_report(y_true, y_pred, f"mlp_{args.setting}_{split_name}")
        save_confusion_matrix(y_true, y_pred, f"mlp_{args.setting}_{split_name}")
        print(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-features", type=int, default=5000)
    p.add_argument("--ngram-max", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--setting", default="default")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    train_mlp(args)


if __name__ == "__main__":
    main()
