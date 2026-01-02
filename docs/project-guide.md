# Project Guide: Training, Deployment, and Batch Inference on SageMaker

This guide shows how to run the included sample project end-to-end using AWS SageMaker. It covers local validation, SageMaker training jobs, real-time endpoints, and batch transform for offline inference. For focused walkthroughs, see `docs/training-job.md` (training) and `docs/endpoint-deployment.md` (real-time deployment).

## 1) Prerequisites
- Complete the AWS setup in `docs/aws-setup.md` (roles, bucket, profile).
- Python 3.10+, AWS CLI, and the SageMaker Python SDK installed locally:
  ```bash
  pip install -r requirements.txt
  ```
- An S3 bucket (referred to as `s3://<bucket>` below) and an execution role ARN.

## 2) Prepare data
- The repository includes `data/iris.csv` (features and labels) and `data/sample-batch.csv` (feature-only rows for inference).
- Upload training and batch input data to S3:
  ```bash
  aws s3 cp data/iris.csv s3://<bucket>/data/iris/train.csv --profile sagemaker-basic
  aws s3 cp data/sample-batch.csv s3://<bucket>/batch-input/sample-batch.csv --profile sagemaker-basic
  ```

## 3) Local dry-run (optional)
- Validate the training script locally:
  ```bash
  python src/train.py --train-data data/iris.csv --model-dir ./model-output
  ls model-output  # Contains model.joblib
  ```
- Run local batch predictions:
  ```bash
  python src/batch_inference.py \
    --model-path ./model-output/model.joblib \
    --input data/sample-batch.csv \
    --output predictions.csv
  ```

## 4) Launch a SageMaker training job
Use the SageMaker Python SDK to submit a managed training job with the provided entry point (`src/train.py`). You can also run the CLI helper `python src/sagemaker_jobs.py train ...` as described in `docs/training-job.md`.

```python
import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

region = "us-east-1"
session = sagemaker.Session()
role = "arn:aws:iam::<ACCOUNT_ID>:role/SageMakerExecutionRole"
bucket = "<bucket>"
train_s3 = f"s3://{bucket}/data/iris/train.csv"
model_output = f"s3://{bucket}/models/iris"

estimator = SKLearn(
    entry_point="src/train.py",
    role=role,
    instance_type="ml.m5.large",
    framework_version="1.2-1",
    py_version="py3",
    source_dir="src",
    hyperparameters={"max_iter": 200},
    output_path=model_output,
)

estimator.fit({"train": train_s3})
print("Training job name:", estimator.latest_training_job.name)
```

Notes:
- `src/train.py` expects a channel named `train` (default when using the dictionary input above).
- Adjust `instance_type` and `max_iter` to control cost and training time.

## 5) Deploy a real-time endpoint
After training finishes, deploy the model as a real-time HTTPS endpoint.

```python
from sagemaker.sklearn.model import SKLearnModel

model = SKLearnModel(
    model_data=estimator.model_data,
    role=role,
    entry_point="inference.py",
    source_dir="src",
    framework_version="1.2-1",
    py_version="py3",
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.t3.medium",
    endpoint_name="iris-realtime-endpoint",
)

response = predictor.predict({"instances": [[5.1, 3.5, 1.4, 0.2]]})
print(response)
```

- The deployment uses `src/inference.py` for `model_fn`, `input_fn`, `predict_fn`, and `output_fn` hooks.
- Delete the endpoint to avoid ongoing charges:
  ```bash
  predictor.delete_endpoint()
  ```

## 6) Run batch inference (Batch Transform)
Batch Transform lets you perform offline, large-scale inference without keeping an endpoint running.

```python
transformer = estimator.transformer(
    instance_count=1,
    instance_type="ml.m5.large",
    strategy="SingleRecord",
    output_path=f"s3://{bucket}/batch-output/",
)

transformer.transform(
    data=f"s3://{bucket}/batch-input/sample-batch.csv",
    content_type="text/csv",
    split_type="Line",
)
transformer.wait()

print("Batch output located at:", transformer.output_path)
```

- `sample-batch.csv` contains only features (no labels); predictions are returned in CSV format.
- Retrieve the batch results:
  ```bash
  aws s3 cp s3://<bucket>/batch-output/ ./batch-results/ --recursive --profile sagemaker-basic
  ```

## 7) Clean up resources
- Stop endpoints and delete CloudWatch logs when finished:
  ```bash
  aws sagemaker delete-endpoint --endpoint-name iris-realtime-endpoint --profile sagemaker-basic
  ```
- Delete training artifacts and batch outputs from S3 to avoid storage costs.

## 8) Troubleshooting tips
- Inspect CloudWatch logs for training and transform jobs for stack traces and metrics.
- Confirm IAM role permissions if jobs fail to download data or write artifacts.
- Ensure the training and inference `framework_version` match (`1.2-1` for scikit-learn in this example).
- If using a VPC, confirm your subnets have outbound internet or necessary VPC endpoints (S3, ECR, CloudWatch Logs).
