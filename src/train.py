# src/train.py
# Part 3 requirements:
# - Load train.csv (prefer Kaggle format)
# - Explore/adjust data with clear print logs (STEP 14–18)
# - Build a logistic regression model
# - Report training accuracy
# - Predict on test.csv and export submission (Kaggle test has no labels, so we print holdout accuracy from train split)
# - Work both locally and in Docker
# - Robust to weird CSV encodings/separators and column name variants

import pathlib
import sys
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import csv

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "src" / "data"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

TRAIN = DATA_DIR / "train.csv"
TEST = DATA_DIR / "test.csv"
SINGLE = DATA_DIR / "titanic.csv"

EXPECTED_LOWER = ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]

def step(n, msg):
    print(f"[STEP {n}] {msg}", flush=True)

def ensure_data():
    have_kaggle = TRAIN.exists() or TEST.exists()
    have_single = SINGLE.exists()
    if not have_kaggle and not have_single:
        step(14, "No local data detected. Will run download_data.py to fetch a single-file fallback.")
        import subprocess, sys as _sys
        subprocess.check_call([_sys.executable, str(ROOT / "src" / "download_data.py")])
    else:
        present = [p.name for p in [TRAIN, TEST, SINGLE] if p.exists()]
        step(14, f"Detected local data files: {present}")

# ---------- Robust CSV reading + column normalization ----------

def _read_csv_robust(path):
    """
    Super-robust CSV reader:
    1) Try sep=None (sniff) with multiple encodings.
    2) If that fails, try explicit delimiters across encodings.
    3) If still failing, use csv.Sniffer() to guess delimiter per encoding.
    Supports weird Excel exports (UTF-16 + tab/comma) and pipe-delimited files.
    """
    import io, csv

    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1", "utf-16", "utf-16le", "utf-16be", "gbk"]
    seps = [None, ",", ";", "\t", "|"]   # None lets pandas sniff with python engine
    last_err = None

    # Try pandas sniffing first with multiple encodings
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, sep=None, engine="python", skipinitialspace=True)
            if df.shape[1] >= 3:
                return df
        except Exception as e:
            last_err = e

    # Try explicit delimiters x encodings
    for enc in encodings:
        for sep in seps[1:]:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine="python", skipinitialspace=True)
                if df.shape[1] >= 3:
                    return df
            except Exception as e:
                last_err = e

    # Use csv.Sniffer on a small sample per encoding to pick delimiter, then read with pandas
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="replace") as f:
                sample = f.read(4096)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t,")
                sep = dialect.delimiter
            except Exception:
                sep = ","  # fallback
            df = pd.read_csv(path, encoding=enc, sep=sep, engine="python", skipinitialspace=True)
            if df.shape[1] >= 3:
                return df
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Failed to read {path} with extended encodings/separators. Last error: {last_err}")

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip spaces, drop BOM, unify to lowercase, replace non-alnum with underscores,
    and fix common alias variants to expected names.
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.replace("\ufeff", "", regex=False)          # BOM
        .str.replace(r"[^\w]+", "_", regex=True)         # non-alnum -> underscore
        .str.lower()
    )
    alias = {
        "p_class": "pclass",
        "ticket_class": "pclass",
        "sex_": "sex",
        "no_of_siblings_spouses": "sibsp",
        "siblings_spouses": "sibsp",
        "no_of_parents_children": "parch",
        "parents_children": "parch",
        "embark": "embarked",
    }
    df.rename(columns={k: v for k, v in alias.items() if k in df.columns}, inplace=True)
    return df

# ---------- Data loaders (Kaggle-preferred, fallback to single-file) ----------

def load_train_dataframe():
    if TRAIN.exists():
        step(15, "Using Kaggle train.csv as the training set (robust loader).")
        df = _read_csv_robust(TRAIN)
    elif SINGLE.exists():
        step(15, "Using single-file fallback titanic.csv as the training set (robust loader).")
        df = _read_csv_robust(SINGLE)
    else:
        print("[ERROR] No available training data.", file=sys.stderr)
        sys.exit(1)

    df = _normalize_columns(df)

    if "survived" not in df.columns:
        step(15, f"Columns detected: {list(df.columns)}")
        raise KeyError("Expected column 'Survived' not found after normalization.")

    keep_lower = ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    missing = [c for c in keep_lower if c not in df.columns]
    if missing:
        step(15, f"Columns detected: {list(df.columns)}")
        raise KeyError(f"Missing expected columns after normalization: {missing}")

    df = df[keep_lower].copy()
    # Rename back to TitleCase expected by downstream code
    df.rename(columns={
        "survived": "Survived",
        "pclass": "Pclass",
        "sex": "Sex",
        "age": "Age",
        "sibsp": "SibSp",
        "parch": "Parch",
        "fare": "Fare",
        "embarked": "Embarked"
    }, inplace=True)
    return df

def load_test_dataframe_like_train():
    if TEST.exists():
        step(16, "Loading Kaggle test.csv for predictions (robust loader).")
        df = _read_csv_robust(TEST)
        df = _normalize_columns(df)
        keep_lower = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
        missing = [c for c in keep_lower if c not in df.columns]
        if missing:
            step(16, f"Columns detected in test.csv: {list(df.columns)}")
            raise KeyError(f"Missing expected test columns after normalization: {missing}")
        df = df[keep_lower].copy()
        df.rename(columns={
            "pclass": "Pclass",
            "sex": "Sex",
            "age": "Age",
            "sibsp": "SibSp",
            "parch": "Parch",
            "fare": "Fare",
            "embarked": "Embarked"
        }, inplace=True)
        return df
    elif SINGLE.exists():
        step(16, "Single-file mode has no separate test.csv; will rely on holdout validation.")
        return None
    else:
        return None

# ---------- Model pipeline ----------

def build_pipeline():
    """
    Logistic regression with basic preprocessing:
    - Numeric: median imputation
    - Categorical: most_frequent imputation + OneHot
    """
    numeric = ["Age", "SibSp", "Parch", "Fare", "Pclass"]
    categorical = ["Sex", "Embarked"]

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical),
        ],
        remainder="drop"
    )

    clf = LogisticRegression(max_iter=1000, n_jobs=None)
    pipe = Pipeline([
        ("pre", pre),
        ("clf", clf)
    ])
    return pipe, numeric + categorical

# ---------- Main ----------

def main():
    # === STEP 14: Data presence and environment check ===
    ensure_data()

    # === STEP 15: Light EDA / cleaning ===
    df = load_train_dataframe()
    step(15, f"Train shape: {df.shape}")
    step(15, f"Missing values (head):\n{df.isna().sum().head()}")

    y = df["Survived"].astype(int)
    X = df.drop(columns=["Survived"])

    # === STEP 16: Feature engineering and model assembly ===
    pipe, feature_cols = build_pipeline()
    step(16, f"Feature columns used: {feature_cols}")

    # === STEP 17: Fit on full train and report training accuracy ===
    step(17, "Fitting logistic regression (max_iter=1000).")
    pipe.fit(X, y)
    train_pred = pipe.predict(X)
    train_acc = accuracy_score(y, train_pred)
    step(17, f"Training accuracy: {train_acc:.4f}")

    # === STEP 18: Holdout validation from train (since Kaggle test has no labels) ===
    step(18, "Performing 80/20 holdout split on train.csv.")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe2, _ = build_pipeline()
    pipe2.fit(Xtr, ytr)
    holdout_acc = accuracy_score(yte, pipe2.predict(Xte))
    step(18, f"Holdout accuracy: {holdout_acc:.4f}")

    # Save model
    model_path = ART / "model.joblib"
    dump(pipe2, model_path)
    step(18, f"Saved model to: {model_path}")

    # Predict on Kaggle test.csv if present and export submission
    test_df = load_test_dataframe_like_train()
    if test_df is not None and TEST.exists():
        step(18, "Generating predictions for Kaggle test.csv and writing artifacts/submission.csv.")
        sub_pred = pipe2.predict(test_df).astype(int)
        raw_test = pd.read_csv(TEST)
        submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Survived": sub_pred})
        out = ART / "submission.csv"
        submission.to_csv(out, index=False)
        step(18, f"Wrote submission file: {out} (Note: official test has no labels, so no test accuracy here.)")
    else:
        step(18, "Kaggle test.csv not found. Skipping submission export.")

    print("\n[SUMMARY]")
    print(f"  Train accuracy : {train_acc:.4f}")
    print(f"  Holdout accuracy: {holdout_acc:.4f}")
    if (ART / "submission.csv").exists():
        print(f"  Submission path : {ART / 'submission.csv'}")
    print(f"  Model path      : {model_path}")

if __name__ == "__main__":
    main()
