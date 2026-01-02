import argparse
import time

import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel


def get_session(region: str | None) -> sagemaker.Session:
    boto_session = boto3.Session(region_name=region)
    return sagemaker.Session(boto_session=boto_session)


def submit_training_job(
    role_arn: str,
    train_s3_uri: str,
    output_s3_uri: str,
    instance_type: str,
    max_iter: int,
    region: str | None,
    job_name: str | None,
) -> SKLearn:
    session = get_session(region)
    resolved_job_name = job_name or f"iris-logreg-{int(time.time())}"

    estimator = SKLearn(
        entry_point="train.py",
        source_dir="src",
        role=role_arn,
        instance_type=instance_type,
        instance_count=1,
        framework_version="1.2-1",
        py_version="py3",
        hyperparameters={"max_iter": max_iter},
        output_path=output_s3_uri,
        sagemaker_session=session,
    )

    estimator.fit({"train": train_s3_uri}, job_name=resolved_job_name)

    print(f"Training job completed: {estimator.latest_training_job.name}")
    print(f"Model artifacts: {estimator.model_data}")

    return estimator


def deploy_realtime_endpoint(
    role_arn: str,
    endpoint_name: str,
    instance_type: str,
    region: str | None,
    model_artifact: str | None = None,
    training_job_name: str | None = None,
):
    session = get_session(region)

    if training_job_name:
        estimator = SKLearn.attach(training_job_name, sagemaker_session=session)
        model_data = estimator.model_data
        framework_version = estimator.framework_version
        py_version = estimator.py_version
    elif model_artifact:
        model_data = model_artifact
        framework_version = "1.2-1"
        py_version = "py3"
    else:
        raise ValueError("Provide either --training-job-name or --model-artifact to deploy an endpoint")

    model = SKLearnModel(
        model_data=model_data,
        role=role_arn,
        entry_point="inference.py",
        source_dir="src",
        framework_version=framework_version,
        py_version=py_version,
        sagemaker_session=session,
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
    )

    print(f"Endpoint deployed: {predictor.endpoint_name}")
    return predictor


def parse_args():
    parser = argparse.ArgumentParser(description="Submit SageMaker training jobs and deploy endpoints.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Submit a SageMaker training job")
    train.add_argument("--role-arn", required=True, help="IAM role ARN for SageMaker execution")
    train.add_argument("--train-s3-uri", required=True, help="S3 URI to training CSV (e.g., s3://bucket/path/train.csv)")
    train.add_argument("--output-s3-uri", required=True, help="S3 prefix for model artifacts (e.g., s3://bucket/models/)")
    train.add_argument("--instance-type", default="ml.m5.large", help="Instance type for training (default: ml.m5.large)")
    train.add_argument("--max-iter", type=int, default=200, help="Max iterations for Logistic Regression")
    train.add_argument("--region", default=None, help="AWS region (falls back to default profile region)")
    train.add_argument("--job-name", default=None, help="Optional custom training job name")

    deploy = subparsers.add_parser("deploy", help="Deploy a trained model to a real-time endpoint")
    deploy.add_argument("--role-arn", required=True, help="IAM role ARN for SageMaker execution")
    deploy.add_argument("--endpoint-name", required=True, help="Name for the SageMaker endpoint")
    deploy.add_argument("--instance-type", default="ml.t3.medium", help="Instance type for the endpoint (default: ml.t3.medium)")
    deploy.add_argument("--region", default=None, help="AWS region (falls back to default profile region)")
    deploy.add_argument("--model-artifact", help="S3 URI to model.tar.gz (alternative to --training-job-name)")
    deploy.add_argument("--training-job-name", help="Existing SageMaker training job name to attach and deploy")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "train":
        submit_training_job(
            role_arn=args.role_arn,
            train_s3_uri=args.train_s3_uri,
            output_s3_uri=args.output_s3_uri,
            instance_type=args.instance_type,
            max_iter=args.max_iter,
            region=args.region,
            job_name=args.job_name,
        )
    elif args.command == "deploy":
        deploy_realtime_endpoint(
            role_arn=args.role_arn,
            endpoint_name=args.endpoint_name,
            instance_type=args.instance_type,
            region=args.region,
            model_artifact=getattr(args, "model_artifact", None),
            training_job_name=getattr(args, "training_job_name", None),
        )


if __name__ == "__main__":
    main()
