#!/usr/bin/env python
import json, time
from argparse import ArgumentParser
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tao import TAOTreeClassifier  # assumes tao.py is importable
import numpy as np

def main():
    p = ArgumentParser()
    p.add_argument("--output", default="metrics.json")
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--passes", type=int, default=5)
    args = p.parse_args()

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = TAOTreeClassifier(max_depth=args.max_depth, max_passes=args.passes, random_state=42)

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = clf.predict(X_test)
    infer_time = time.perf_counter() - t0

    acc = float(accuracy_score(y_test, preds))
    metrics = {
        "train_time_seconds": train_time,
        "inference_time_seconds": infer_time,
        "test_accuracy": acc,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "classes": int(np.unique(y).size),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
