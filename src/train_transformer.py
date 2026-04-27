from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback

from constants import DATA_PROCESSED, RESULTS, RANDOM_SEED
from evaluate import append_metrics, compute_metrics, save_classification_report, save_confusion_matrix


class NLIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.df = df.reset_index(drop=True)
        self.enc = tokenizer(
            self.df["sentence1"].tolist(),
            self.df["sentence2"].tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = self.df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device) if self.class_weights is not None else None)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def metric_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return compute_metrics(labels, preds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="distilbert-base-multilingual-cased")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-train", type=int, default=2000)
    p.add_argument("--max-val", type=int, default=500)
    p.add_argument("--max-test", type=int, default=500)
    p.add_argument("--setting", default="default")
    p.add_argument("--no-class-weights", action="store_true")
    args = p.parse_args()

    train = pd.read_csv(DATA_PROCESSED / "train.csv").sample(frac=1, random_state=RANDOM_SEED).head(args.max_train)
    val = pd.read_csv(DATA_PROCESSED / "validation.csv").sample(frac=1, random_state=RANDOM_SEED).head(args.max_val)
    test = pd.read_csv(DATA_PROCESSED / "test.csv").sample(frac=1, random_state=RANDOM_SEED).head(args.max_test)

    num_labels = int(max(train.label.max(), val.label.max(), test.label.max()) + 1)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

    train_ds = NLIDataset(train, tokenizer, args.max_length)
    val_ds = NLIDataset(val, tokenizer, args.max_length)
    test_ds = NLIDataset(test, tokenizer, args.max_length)

    class_weights = None
    if not args.no_class_weights:
        cw = compute_class_weight("balanced", classes=np.arange(num_labels), y=train["label"].to_numpy())
        class_weights = torch.tensor(cw, dtype=torch.float32)

    training_args = TrainingArguments(
        output_dir=f"models/transformer_{args.setting}",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=25,
        seed=RANDOM_SEED,
        report_to="none",
        save_total_limit=1,
    )
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=metric_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
        class_weights=class_weights,
    )

    start = time.time()
    trainer.train()
    train_time = time.time() - start

    for split_name, df, ds in [("validation", val, val_ds), ("test", test, test_ds)]:
        t0 = time.time()
        out = trainer.predict(ds)
        inference_time = time.time() - t0
        pred = np.argmax(out.predictions, axis=1)
        pd.DataFrame({
            "guid": df.get("guid", pd.Series(range(len(df)))),
            "sentence1": df["sentence1"],
            "sentence2": df["sentence2"],
            "y_true": df["label"].to_numpy(),
            "y_pred": pred,
        }).to_csv(RESULTS / f"transformer_{args.setting}_{split_name}_predictions.csv", index=False)
        row = {
            "model": "transformer",
            "split": split_name,
            "setting": args.setting,
            "pretrained_model": args.model_name,
            "train_time_sec": train_time,
            "inference_time_sec": inference_time,
            "examples_per_sec": len(df) / max(inference_time, 1e-9),
            "lr": args.lr,
            "class_weights": not args.no_class_weights,
            **compute_metrics(df["label"], pred),
        }
        append_metrics(row)
        save_classification_report(df["label"], pred, f"transformer_{args.setting}_{split_name}")
        save_confusion_matrix(df["label"], pred, f"transformer_{args.setting}_{split_name}")
        print(row)


if __name__ == "__main__":
    main()
