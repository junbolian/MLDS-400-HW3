# src/train.py
import pathlib
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "src" / "data"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

TRAIN = DATA_DIR / "train.csv"
TEST = DATA_DIR / "test.csv"
GENDER = DATA_DIR / "gender_submission.csv"
SINGLE = DATA_DIR / "titanic.csv"

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

def ensure_data():
    have_kaggle_any = any(p.exists() for p in [TRAIN, TEST, GENDER])
    if not have_kaggle_any and not SINGLE.exists():
        print("[train] No local Titanic data detected. Running download_data.py ...")
        import subprocess, sys as _sys
        subprocess.check_call([_sys.executable, str(ROOT / "src" / "download_data.py")])

    if not (TRAIN.exists() or SINGLE.exists()):
        print("[train] No data found after attempting download.", file=sys.stderr)
        sys.exit(1)

def _prep_X(df: pd.DataFrame, age_median=None, fare_median=None):
    X = df[FEATURES].copy()
    X["Sex"] = (X["Sex"] == "male").astype(int)
    if age_median is None:
        age_median = X["Age"].median()
    if fare_median is None:
        fare_median = X["Fare"].median()
    X["Age"] = X["Age"].fillna(age_median)
    X["Fare"] = X["Fare"].fillna(fare_median)
    return X, age_median, fare_median

def run_kaggle_mode():
    print("[train] Using Kaggle format (train.csv [+ test.csv]).")
    df_tr = pd.read_csv(TRAIN)

    y = df_tr["Survived"].astype(int)
    X, age_median, fare_median = _prep_X(df_tr)

    clf = LogisticRegression(max_iter=1000)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    print(f"[train] Holdout accuracy (from train.csv split): {acc:.4f}")

    dump(clf, ART / "model.joblib")
    print(f"[train] Saved model to {ART / 'model.joblib'}")

    if TEST.exists():
        df_te = pd.read_csv(TEST)
        Xsub, _, _ = _prep_X(df_te, age_median=age_median, fare_median=fare_median)
        pred = clf.predict(Xsub)
        sub = pd.DataFrame({"PassengerId": df_te["PassengerId"], "Survived": pred.astype(int)})
        out = ART / "submission.csv"
        sub.to_csv(out, index=False)
        print(f"[train] Wrote Kaggle-style predictions to {out}")

def run_single_file_mode():
    print("[train] Using single-file fallback (titanic.csv).")
    df = pd.read_csv(SINGLE)
    y = df["Survived"].astype(int)
    X, _, _ = _prep_X(df)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    print(f"[train] Test accuracy: {acc:.4f}")
    dump(clf, ART / "model.joblib")
    print(f"[train] Saved model to {ART / 'model.joblib'}")

def main():
    ensure_data()
    if TRAIN.exists():
        run_kaggle_mode()
    else:
        run_single_file_mode()

if __name__ == "__main__":
    main()
