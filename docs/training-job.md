# SageMaker Training Job Guide

Use this guide to launch a managed SageMaker training job for the Iris sample project.

## Prerequisites
- Complete AWS setup in `docs/aws-setup.md` (role, bucket, credentials).
- Install project dependencies locally:
  ```bash
  pip install -r requirements.txt
  ```
- Upload training data:
  ```bash
  aws s3 cp data/iris.csv s3://<bucket>/data/iris/train.csv --profile sagemaker-basic
  ```

## Submit a training job with the helper script
The repository includes `src/sagemaker_jobs.py` to simplify training submissions.

```bash
python src/sagemaker_jobs.py train \
  --role-arn arn:aws:iam::<ACCOUNT_ID>:role/SageMakerExecutionRole \
  --train-s3-uri s3://<bucket>/data/iris/train.csv \
  --output-s3-uri s3://<bucket>/models/iris/ \
  --instance-type ml.m5.large \
  --max-iter 200 \
  --region us-east-1
```

Key details:
- `--output-s3-uri` receives the model artifacts produced as `model.tar.gz`.
- Use `--job-name` to override the auto-generated job name.
- Logs stream to CloudWatch; monitor metrics from the SageMaker console.

## Submit using the SageMaker SDK inline
If you prefer to work directly in a notebook or Python REPL, use the snippet below.

```python
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

role = "arn:aws:iam::<ACCOUNT_ID>:role/SageMakerExecutionRole"
bucket = "<bucket>"
train_s3 = f"s3://{bucket}/data/iris/train.csv"
output_prefix = f"s3://{bucket}/models/iris"

estimator = SKLearn(
    entry_point="train.py",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    framework_version="1.2-1",
    py_version="py3",
    source_dir="src",
    hyperparameters={"max_iter": 200},
    output_path=output_prefix,
)

estimator.fit({"train": train_s3})
print("Training job:", estimator.latest_training_job.name)
print("Model data:", estimator.model_data)
```

## After training
- Note the `model_data` S3 URI from the SDK output; you will pass this to deployment.
- Training outputs are versioned per job under the specified S3 prefix.
- Stop or clean up log groups if you created custom ones for the job.
