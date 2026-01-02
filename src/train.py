import argparse
import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, default=None, help="Path to local training CSV (overrides SM_CHANNEL_TRAIN)")
    parser.add_argument("--model-dir", type=str, default=None, help="Directory to store the trained model")
    parser.add_argument("--max-iter", type=int, default=200, help="Maximum iterations for Logistic Regression")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data to use for validation")
    return parser.parse_args()


def load_training_data(train_path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(train_path)
    features = df.drop(columns=["target"])
    labels = df["target"].astype(int)
    return features, labels


def train_model(X: pd.DataFrame, y: pd.Series, max_iter: int) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=max_iter,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def save_model(model: Pipeline, model_dir: str) -> None:
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")


def main():
    args = parse_args()

    train_path = args.train_data or os.environ.get("SM_CHANNEL_TRAIN")
    if train_path is None:
        raise ValueError("Training data path must be provided via --train-data or SM_CHANNEL_TRAIN")

    model_dir = args.model_dir or os.environ.get("SM_MODEL_DIR", "./model")

    X, y = load_training_data(train_path)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)

    model = train_model(X_train, y_train, max_iter=args.max_iter)
    val_accuracy = model.score(X_val, y_val)
    print(f"Validation accuracy: {val_accuracy:.4f}")

    save_model(model, model_dir)


if __name__ == "__main__":
    main()
