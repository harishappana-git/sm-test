# SageMaker Endpoint Deployment Guide

This document walks through deploying a trained Iris model to a real-time SageMaker endpoint and exercising it with sample payloads.

## Prerequisites
- A completed SageMaker training job for this project (see `docs/training-job.md`).
- IAM role ARN, S3 model artifact URI, and region details.
- Dependencies installed locally:
  ```bash
  pip install -r requirements.txt
  ```

## Deploy with the helper script
Deploy directly from the CLI using `src/sagemaker_jobs.py`.

```bash
python src/sagemaker_jobs.py deploy \
  --role-arn arn:aws:iam::<ACCOUNT_ID>:role/SageMakerExecutionRole \
  --training-job-name <previous-training-job-name> \
  --endpoint-name iris-realtime-endpoint \
  --instance-type ml.t3.medium \
  --region us-east-1
```

Notes:
- Replace `--training-job-name` with `--model-artifact s3://<bucket>/models/iris/<job>/output/model.tar.gz` if you prefer to deploy from an artifact URI.
- The script publishes a `SKLearnModel` using `src/inference.py` for request/response handling.

## Deploy from the SageMaker SDK directly
```python
from sagemaker.sklearn.model import SKLearnModel

role = "arn:aws:iam::<ACCOUNT_ID>:role/SageMakerExecutionRole"
model_artifact = "s3://<bucket>/models/iris/<job>/output/model.tar.gz"

model = SKLearnModel(
    model_data=model_artifact,
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

print("Endpoint:", predictor.endpoint_name)
```

## Invoke the endpoint
Use JSON payloads to request predictions.

```bash
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name iris-realtime-endpoint \
  --body '{"instances": [[5.1, 3.5, 1.4, 0.2]]}' \
  --content-type application/json \
  --accept application/json \
  response.json --profile sagemaker-basic
cat response.json
```

The response includes `predicted_class` and class probabilities as defined in `src/inference.py`.

## Clean up
- Remove the endpoint to stop charges:
  ```bash
  aws sagemaker delete-endpoint --endpoint-name iris-realtime-endpoint --profile sagemaker-basic
  ```
- Delete associated endpoint configurations and models if created separately.
