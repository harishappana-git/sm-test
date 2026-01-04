# SageMaker Basic Project Template

This repository provides a minimal end-to-end template for training and deploying a machine learning model on AWS SageMaker. It includes:

- A sample dataset (Iris) for classification experiments.
- Training and inference scripts compatible with the SageMaker Python SDK.
- Batch transform utilities.
- Detailed guides for AWS account setup and running the project.

## Repository layout

- `data/`: Sample datasets for quick experiments.
- `docs/`: Step-by-step guides for AWS setup and project execution.
- `src/`: Training and inference entry points for SageMaker jobs.
- `requirements.txt`: Python dependencies for local runs.

## Quick start

1. Install dependencies locally (Python 3.10+ recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Explore the guides:
   - AWS setup prerequisites: `docs/aws-setup.md`
   - Training job walkthrough: `docs/training-job.md`
   - Endpoint deployment walkthrough: `docs/endpoint-deployment.md`
   - End-to-end flow (including batch transform): `docs/project-guide.md`
   - GitHub Actions CI/CD (continuous training + deployment): `docs/github-actions-cicd.md`
3. Run a local training dry-run:
   ```bash
   python src/train.py \
     --train-data data/iris.csv \
     --model-dir ./model-output
   ```
4. Create a batch prediction locally using the saved model:
   ```bash
   python src/batch_inference.py \
     --model-path ./model-output/model.joblib \
     --input data/sample-batch.csv \
     --output predictions.csv
   ```

## Next steps

Follow `docs/project-guide.md` to launch SageMaker training, deploy an endpoint, and run batch transform jobs using the provided scripts. Use `src/sagemaker_jobs.py` if you prefer ready-made CLI commands for training and deployment.
