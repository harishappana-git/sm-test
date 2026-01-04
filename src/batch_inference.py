import argparse
import pandas as pd
import joblib


def parse_args():
    parser = argparse.ArgumentParser(description="Run local batch predictions using a trained model.")
    parser.add_argument("--model-path", required=True, help="Path to model.joblib produced by training")
    parser.add_argument("--input", required=True, help="Path to CSV with feature columns")
    parser.add_argument("--output", required=True, help="Path to write predictions CSV")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input)
    model = joblib.load(args.model_path)

    predictions = model.predict(df)
    probabilities = model.predict_proba(df)

    output = pd.DataFrame({"predicted_class": predictions})
    for idx in range(probabilities.shape[1]):
        output[f"prob_class_{idx}"] = probabilities[:, idx]

    output.to_csv(args.output, index=False)
    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
