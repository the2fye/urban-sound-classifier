# trains a classical ml model on hand built audio features (dog_bark / drilling / street_music)
# usage: python train_model.py --data UrbanSound_Filtered

import argparse
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import load_audio, extract_features, feature_names

TEST_FOLDS = [9, 10]  # hold out whole folds so slices from the same recording don't leak into the test set


def featurize_one(path):
    try:
        return extract_features(load_audio(path))
    except Exception as e:
        print(f"skipping {path}: {e}")
        return None


def build_features(data_dir, cache_path):
    meta = pd.read_csv(os.path.join(data_dir, "filtered_metadata.csv"))
    audio_dir = os.path.join(data_dir, "audio")

    # only keep rows whose wav actually exists
    meta["path"] = meta["slice_file_name"].apply(lambda f: os.path.join(audio_dir, f))
    missing = ~meta["path"].apply(os.path.exists)
    if missing.any():
        print(f"warning: {missing.sum()} files in the csv weren't found, dropping them")
    meta = meta[~missing].reset_index(drop=True)

    # reuse cached features if they match (extraction is the slow part)
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        same_files = len(cached) == len(meta) and (
            cached["slice_file_name"].to_numpy(dtype=object) == meta["slice_file_name"].to_numpy(dtype=object)
        ).all()
        if same_files:
            print(f"loaded cached features from {cache_path}")
            return cached

    print(f"extracting features from {len(meta)} files (takes a few minutes the first time)...")
    feats = Parallel(n_jobs=-1, verbose=5)(delayed(featurize_one)(p) for p in meta["path"])

    keep = [f is not None for f in feats]
    meta = meta[keep].reset_index(drop=True)
    X = pd.DataFrame(np.vstack([f for f in feats if f is not None]), columns=feature_names())
    df = pd.concat([meta[["slice_file_name", "fold", "classID", "class"]], X], axis=1)
    df.to_csv(cache_path, index=False)
    print(f"saved features to {cache_path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="UrbanSound_Filtered", help="folder with audio/ and filtered_metadata.csv")
    parser.add_argument("--out", default="model", help="where to save the model and results")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = build_features(args.data, os.path.join(args.out, "features.csv"))
    cols = feature_names()

    # drop any rows with nan/inf in case a clip was corrupted
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols).reset_index(drop=True)
    print("\nclass counts:\n", df["class"].value_counts().to_string())

    train = df[~df["fold"].isin(TEST_FOLDS)]
    test = df[df["fold"].isin(TEST_FOLDS)]
    X_train = train[cols].to_numpy()
    y_train = train["class"].to_numpy(dtype=object)
    g_train = train["fold"].to_numpy()
    X_test = test[cols].to_numpy()
    y_test = test["class"].to_numpy(dtype=object)
    print(f"\ntrain: {len(train)} clips (folds 1-8) | test: {len(test)} clips (folds {TEST_FOLDS})")

    # the scaler lives inside each pipeline so it only ever gets fit on training data
    candidates = {
        "random_forest": RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1),
        # calibrated wrapper gives the svm predict_proba for the app
        "svm_rbf": CalibratedClassifierCV(SVC(kernel="rbf", C=10, gamma="scale"), ensemble=False),
        "knn": KNeighborsClassifier(n_neighbors=7),
    }

    # pick the model with leave-one-fold-out cross validation on the training folds only
    cv_scores = {}
    print("\nleave-one-fold-out cv on training folds:")
    for name, clf in candidates.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        scores = cross_val_score(pipe, X_train, y_train, groups=g_train, cv=LeaveOneGroupOut(), n_jobs=-1)
        cv_scores[name] = float(scores.mean())
        print(f"  {name:14s} {scores.mean():.3f} +/- {scores.std():.3f}")

    best = max(cv_scores, key=cv_scores.get)
    print(f"\nbest model: {best}")
    model = Pipeline([("scaler", StandardScaler()), ("clf", candidates[best])])
    model.fit(X_train, y_train)

    # final check on the held out folds
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\ntest accuracy: {acc:.3f}\n")
    print(classification_report(y_test, preds, digits=3))

    labels = list(model.classes_)
    cm = confusion_matrix(y_test, preds, labels=labels)
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap="Blues", colorbar=False)
    plt.title(f"{best} - test folds {TEST_FOLDS} (acc {acc:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "confusion_matrix.png"), dpi=150)

    # save everything the app needs
    joblib.dump(model, os.path.join(args.out, "audio_model.joblib"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({
            "best_model": best,
            "cv_accuracy": cv_scores,
            "test_folds": TEST_FOLDS,
            "test_accuracy": acc,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "classes": labels,
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(y_test, preds, output_dict=True),
        }, f, indent=2)
    print(f"saved model, metrics and confusion matrix to {args.out}/")


if __name__ == "__main__":
    main()
