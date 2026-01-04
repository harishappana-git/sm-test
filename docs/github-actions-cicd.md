# GitHub Actions CI/CD (Continuous Training + Deployment)

This repo includes two GitHub Actions workflows:

- `CI` (`.github/workflows/ci.yml`): Runs a local smoke test (train + batch inference) on every PR and push when `src/`, `data/`, or `requirements.txt` changes.
- `Continuous Train & Deploy (SageMaker)` (`.github/workflows/continuous-train-deploy.yml`): On every push to `main` (when `src/` or `data/` changes), uploads the training CSV to S3, launches a SageMaker training job, deploys/updates a real-time endpoint, and invokes it once as a smoke test.

## Required GitHub Secrets

Set these under **Repo → Settings → Secrets and variables → Actions → Secrets**:

- `AWS_REGION`: e.g. `us-east-1`
- `S3_BUCKET`: S3 bucket name (no `s3://` prefix)
- Auth (choose one):
  - **Recommended (OIDC)**: `AWS_ROLE_TO_ASSUME`
  - **Access keys**: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
- `SAGEMAKER_ROLE_ARN`: SageMaker *execution role* ARN used by training and endpoint deployment.
  - If omitted, the workflow falls back to `AWS_ROLE_TO_ASSUME` (only works if that role can be assumed by both GitHub OIDC and `sagemaker.amazonaws.com`).

## Optional GitHub Variables

Set these under **Repo → Settings → Secrets and variables → Actions → Variables**:

- `ENDPOINT_NAME` (default: `iris-realtime-endpoint`)
- `TRAIN_INSTANCE_TYPE` (default: `ml.m5.large`)
- `ENDPOINT_INSTANCE_TYPE` (default: `ml.t3.medium`)
- `MAX_ITER` (default: `200`)

You can also override `ENDPOINT_NAME`, instance types, and `MAX_ITER` via **Run workflow** (manual `workflow_dispatch`) in GitHub.

## IAM notes (high level)

The identity used by GitHub Actions must be able to:

- Write to your S3 bucket/prefix (uploads `data/iris.csv`).
- Create/describe SageMaker training jobs and create/update endpoints.
- Pass the SageMaker execution role (`iam:PassRole` on `SAGEMAKER_ROLE_ARN`).

## What gets uploaded/created

- Training data upload: `s3://$S3_BUCKET/data/iris/$GITHUB_SHA/train.csv`
- Training outputs: `s3://$S3_BUCKET/models/iris/$GITHUB_SHA/` (and SageMaker-managed job subpaths)

