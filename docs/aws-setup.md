# AWS Setup Guide for SageMaker

Follow these steps to prepare an AWS account for running the training and inference workflows in this repository.

## 1) Account and region preparation
- Choose an AWS region that supports SageMaker (for example, `us-east-1`).
- Create or select an AWS account with sufficient service quotas for SageMaker training, endpoints, and S3 storage.
- Configure a local AWS profile using the AWS CLI:
  ```bash
  aws configure --profile sagemaker-basic
  # Provide Access Key ID, Secret Access Key, default region, and output format
  ```

## 2) IAM roles and permissions
- **SageMaker execution role**: Create a role named `SageMakerExecutionRole` with the following policies:
  - `AmazonSageMakerFullAccess` (or the more restrictive `AmazonSageMakerCanvasFullAccess` + specific permissions for your workflow).
  - `AmazonS3FullAccess` (or a scoped-down policy to the specific S3 buckets you will use).
  - Optional: `CloudWatchLogsFullAccess` to troubleshoot training jobs and endpoints.
- **Studio/Notebook access (optional)**: If using SageMaker Studio, ensure your user profile role can assume the execution role above.
- Record the execution role ARN for later use (for example, `arn:aws:iam::<ACCOUNT_ID>:role/SageMakerExecutionRole`).

## 3) Networking
- Verify your account has a default VPC with public subnets. SageMaker will use these subnets unless you configure VPC isolation.
- For VPC-only deployments, allow outbound internet via a NAT gateway or VPC endpoints for S3, ECR, and CloudWatch Logs.
- If you use private subnets, ensure the security group allows HTTPS egress for model artifact downloads and metrics.

## 4) S3 buckets
- Create (or reuse) an S3 bucket for training data and model artifacts:
  ```bash
  aws s3 mb s3://<your-bucket-name> --profile sagemaker-basic
  ```
- Create logical prefixes for organization:
  - `data/iris/` for training CSVs.
  - `models/iris/` for model artifacts exported by training jobs.
  - `batch-input/` and `batch-output/` for batch transform jobs.

## 5) ECR (only if using custom containers)
- If you later build custom images, create an ECR repository:
  ```bash
  aws ecr create-repository --repository-name sagemaker-basic --profile sagemaker-basic
  ```
- Authenticate Docker to ECR before pushing images:
  ```bash
  aws ecr get-login-password --region <region> --profile sagemaker-basic \
    | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
  ```

## 6) Local tooling
- Install the following locally:
  - AWS CLI v2
  - Python 3.10+ and `pip`
  - Docker (optional; required only for custom containers)
- Install the Python dependencies from this repo for local runs:
  ```bash
  pip install -r requirements.txt
  ```

## 7) Quotas and cost controls
- Check SageMaker service quotas for training and endpoint instance types you plan to use (for example, `ml.t3.medium` or `ml.m5.large`).
- Set up AWS Budgets and cost alerts.
- Optionally enable SageMaker Endpoint Auto Scaling if you plan to keep real-time endpoints active.

## 8) Validating access
- Confirm you can list SageMaker resources in the chosen region:
  ```bash
  aws sagemaker list-training-jobs --max-items 5 --profile sagemaker-basic
  ```
- Confirm S3 access:
  ```bash
  aws s3 ls s3://<your-bucket-name>/ --profile sagemaker-basic
  ```

You are now ready to follow the project guide to launch training and deployment jobs.
