from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import h5py
import numpy as np
import boto3
import mnist_model  # Your model file
import mlflow
import mlflow.pytorch

# Constants
S3_BUCKET = "ai22m020-models"
HDF5_BUCKET = "ai22m020-datasets"
LOCAL_H5_PATH = "/home/ubuntu/data/"
MODEL_SAVE_PATH = "/home/ubuntu/trained_models/"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)


def download_latest_h5_from_s3(**kwargs):
    conf = kwargs.get("dag_run").conf if kwargs.get("dag_run") else {}
    dataset_name = conf.get("dataset_name", "default_value")
    binarization_number = conf.get("binarization", "-1")
    if binarization_number == "0":
        binarization = False
    else:
        binarization = True
    # CHECK WHY BINARIZATION PARAM IS NOT TAKEN FROM REUQEST BODY, IT IS AWLAYS FALSE TO MODEL IS OVERWRITTEN!

    if binarization:
        dataset_name = f"{dataset_name}_bin"
    else:
        dataset_name = f"{dataset_name}_reg"
    s3_client = boto3.client("s3")
    objects = s3_client.list_objects_v2(Bucket=HDF5_BUCKET, Prefix=dataset_name)

    h5_files = [obj['Key'] for obj in objects.get("Contents", []) if obj['Key'].endswith(".h5")]
    if not h5_files:
        raise FileNotFoundError("No HDF5 files found in S3 bucket")

    latest_file = sorted(h5_files)[-1]  # Get the latest by name versioning
    local_path = os.path.join(LOCAL_H5_PATH, os.path.basename(latest_file))
    s3_client.download_file(HDF5_BUCKET, latest_file, local_path)

    print(f"Downloaded {latest_file} to {local_path}")
    kwargs["ti"].xcom_push(key="hdf5_file", value=local_path)
    kwargs["ti"].xcom_push(key="binarization", value=binarization)

def train_model(**kwargs):
    h5_file = kwargs["ti"].xcom_pull(task_ids="download_hdf5", key="hdf5_file")
    binarization = kwargs["ti"].xcom_pull(task_ids="download_hdf5", key="binarization")
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"File not found: {h5_file}")

    with h5py.File(h5_file, 'r') as hf:
        images = hf['images'][:]
        labels = hf['labels'][:]

    images = images.astype(np.float32) / 255.0  # Normalize to [0,1]
    images = (images - 0.1307) / 0.3081  # Further normalize
    images = torch.tensor(images).unsqueeze(1)  # Add channel
    labels = torch.tensor(labels)

    dataset = torch.utils.data.TensorDataset(images, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = mnist_model.Net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 1
    final_loss = 0.0

    with mlflow.start_run(run_name="mnist-training") as run:
        mlflow.log_param("epochs", num_epochs)
        mlflow.log_param("learning_rate", 1.0)
        mlflow.log_param("gamma", 0.7)
        mlflow.log_param("batch_size", 64)

        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(loader)
            mlflow.log_metric("loss", avg_loss, step=epoch)
            final_loss = avg_loss
            scheduler.step()

        if binarization:
            base_name = "mnist_bin"
        else:
            base_name = "mnist_reg"

        s3_client = boto3.client("s3")
        existing_files = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=base_name)
        existing_versions = []
        if "Contents" in existing_files:
            for obj in existing_files["Contents"]:
                filename = obj["Key"]
                if filename.startswith(base_name) and filename.endswith(".pt"):
                    parts = filename.replace(".pt", "").split("_v")
                    if len(parts) == 2 and parts[1].isdigit():
                        existing_versions.append(int(parts[1]))

        new_version = max(existing_versions, default=0) + 1

        new_pt_filename = f"{base_name}_v{new_version}.pt"

        local_model_path = os.path.join(MODEL_SAVE_PATH, new_pt_filename)

        torch.save(model.state_dict(), local_model_path)
        print(f"Model saved to {local_model_path}")

        # Log the model file as artifact
        mlflow.log_artifact(local_model_path, artifact_path="models")

        kwargs["ti"].xcom_push(key="model_file", value=local_model_path)

        # Optional: Save HDF5 dataset file path and final loss
        mlflow.log_param("hdf5_file", h5_file)
        mlflow.log_metric("final_loss", final_loss)

        # Store run ID if needed later
        # kwargs["ti"].xcom_push(key="mlflow_run_id", value=run.info.run_id)


def upload_model_to_s3(**kwargs):
    model_path = kwargs["ti"].xcom_pull(task_ids="train_model", key="model_file")
    # run_id = kwargs["ti"].xcom_pull(task_ids="train_model", key="mlflow_run_id")

    s3_client = boto3.client("s3")
    s3_key = os.path.basename(model_path)
    s3_client.upload_file(model_path, S3_BUCKET, s3_key)
    s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
    print(f"Uploaded model to {s3_uri}")

    # if run_id:
    #     mlflow.set_tracking_uri("http://0.0.0.0:5000")  # or your MLflow URI
    #     mlflow.set_experiment("default")
    #     with mlflow.start_run(run_id=run_id):
    #         mlflow.set_tag("s3_model_path", s3_uri)


# DAG Definition
default_args = {"owner": "airflow", "start_date": datetime.now()}
with DAG("train_base_model_ec2", default_args=default_args, schedule_interval=None, catchup=False) as dag:
    download_task = PythonOperator(
        task_id="download_hdf5",
        python_callable=download_latest_h5_from_s3,
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    upload_task = PythonOperator(
        task_id="upload_model",
        python_callable=upload_model_to_s3,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    trigger_train_dag = TriggerDagRunOperator(
        task_id="train_art",
        trigger_dag_id="train_defence_models_ec2",
        wait_for_completion=True,
        conf={
            "model_name": "{{ ti.xcom_pull(task_ids='train_model', key='model_file') | basename }}"
        },
    )

    download_task >> train_task >> upload_task >> trigger_train_dag
