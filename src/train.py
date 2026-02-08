import json
import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA_PATH = Path("data/weatherAUS.csv")
OUT_DIR = Path("results")
MODEL_DIR = Path("models")

def load_data(path: Path) -> pd.DataFrame:

    if not path.exists():
        raise FileExistsError(...)
    return pd.read_csv(path)

def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()


    numeric_transform = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    
    categorial_transform = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transform, numeric_features),
            ("cat", categorial_transform,categorical_features),
        ]
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")

    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])
    return clf

def main(target_col: str = "RainTomorrow") -> None:
    df = load_data(DATA_PATH)
    df.columns = df.columns.str.strip()  # safety

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found. Columns: {df.columns.tolist()}")

    # ---- Clean target values robustly ----
    before = len(df)

    # Normalize strings: "No ", " YES", "no" -> "No"/"Yes"
    df[target_col] = df[target_col].astype(str).str.strip().str.lower()

    # Keep only valid labels
    df = df[df[target_col].isin(["yes", "no"])].copy()

    # Map to 1/0
    y = df[target_col].map({"no": 0, "yes": 1})

    # Final safety (should already be clean)
    mask = y.notna()
    df = df.loc[mask].copy()
    y = y.loc[mask]

    print(f"Rows before target cleaning: {before}")
    print(f"Rows after target cleaning : {len(df)}")
    print("Target distribution (0=No, 1=Yes):", y.value_counts().to_dict())

    # Features
    X = df.drop(columns=[target_col])

    # Train/test split (stratify keeps class ratio same in train/test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = build_pipeline(X_train)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    metrics = {
        "target": target_col,
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "report": classification_report(y_test, preds, digits=4),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2),encoding="utf-8")
    joblib.dump(clf, MODEL_DIR / f"model_{target_col}.joblib")

    print("✅ Saved:", OUT_DIR / "metrics.json")
    print("✅ Saved:", MODEL_DIR / f"model_{target_col}.joblib")
    print("\n=== Classification Report ===")
    print(metrics["report"])
    print(f"\nAccuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main("RainTomorrow")



