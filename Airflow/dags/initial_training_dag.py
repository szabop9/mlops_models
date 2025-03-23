from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
import h5py
import numpy as np
import boto3
import mnist_model  # Your model file

# Constants
S3_BUCKET = "ai22m020-models"
HDF5_BUCKET = "ai22m020-datasets"
LOCAL_H5_PATH = "/home/ubuntu/data/"
MODEL_SAVE_PATH = "/home/ubuntu/trained_models/"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def download_latest_h5_from_s3(dataset_name, **kwargs):
    hdf5_files = [f for f in os.listdir(LOCAL_H5_PATH) if f.startswith(dataset_name) and f.endswith(".h5")]
    if not hdf5_files:
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
    else:
        hdf5_files.sort(reverse=True)
        latest_file = hdf5_files[0]
        local_hdf5_path = os.path.join(LOCAL_H5_PATH, latest_file)
        kwargs["ti"].xcom_push(key="hdf5_file", value=local_hdf5_path)

def train_model(**kwargs):
    h5_file = kwargs["ti"].xcom_pull(task_ids="download_hdf5", key="hdf5_file")
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

    model.train()
    for epoch in range(10):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

    local_model_path = os.path.join(MODEL_SAVE_PATH, "mnist_cnn.pt")
    torch.save(model.state_dict(), local_model_path)
    print(f"Model saved to {local_model_path}")

    kwargs["ti"].xcom_push(key="model_file", value=local_model_path)

def upload_model_to_s3(**kwargs):
    model_path = kwargs["ti"].xcom_pull(task_ids="train_model", key="model_file")
    s3_client = boto3.client("s3")
    s3_client.upload_file(model_path, S3_BUCKET, os.path.basename(model_path))
    print(f"Uploaded model to s3://{S3_BUCKET}/{os.path.basename(model_path)}")


# DAG Definition
default_args = {"owner": "airflow", "start_date": datetime.now()}
with DAG("train_base_model_ec2", default_args=default_args, schedule_interval=None, catchup=False) as dag:

    download_task = PythonOperator(
        task_id="download_hdf5",
        python_callable=download_latest_h5_from_s3,
        op_kwargs={"dataset_name": "mnist"},
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

    download_task >> train_task >> upload_task
