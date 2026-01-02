import io
import json
from typing import Any

import joblib
import pandas as pd


def model_fn(model_dir: str):
    model_path = f"{model_dir}/model.joblib"
    model = joblib.load(model_path)
    return model


def input_fn(request_body: bytes, content_type: str):
    if content_type == "application/json":
        payload = json.loads(request_body.decode("utf-8"))
        data = payload.get("instances") or payload
        return pd.DataFrame(data, columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ])

    if content_type in ("text/csv", "text/x-libsvm"):
        return pd.read_csv(io.BytesIO(request_body))

    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data: pd.DataFrame, model):
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    return {"predicted_class": predictions.tolist(), "probabilities": probabilities.tolist()}


def output_fn(prediction: Any, accept: str):
    if accept == "application/json" or accept is None:
        return json.dumps(prediction), "application/json"

    if accept == "text/csv":
        df = pd.DataFrame(prediction)
        csv_str = df.to_csv(index=False)
        return csv_str, "text/csv"

    raise ValueError(f"Unsupported Accept type: {accept}")
