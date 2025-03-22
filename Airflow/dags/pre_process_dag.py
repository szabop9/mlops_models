from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import h5py
import boto3
import numpy as np
from PIL import Image

# AWS S3 Bucket Name
S3_BUCKET = "ai22m020-datasets"

# EC2 Image Dataset Path (Ensure this path exists)
DATASET_PATH = "/home/ubuntu/dev/mlops_models/Airflow/data"  # Update based on EC2 directory

# Ensure data directories exist
LOCAL_SAVE_DIR = "/home/ubuntu/data/"
os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)
# test 3

def load_images_and_labels(dataset_name, **kwargs):
    """Loads images from the local EC2 directory and saves them into an HDF5 file with a unique versioned name."""

    dataset_path = os.path.join(DATASET_PATH, dataset_name, "classes")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist.")

    print(f"Loading images from {dataset_path}...")

    # Get class labels dynamically (folders named [0-9])
    class_labels = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    print(f"Detected classes: {class_labels}")

    images, labels = [], []

    for label in class_labels:
        class_path = os.path.join(dataset_path, label)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path).convert("L")  # Convert to grayscale
            img = img.resize((28, 28))  # Ensure correct size
            images.append(np.array(img))
            labels.append(int(label))

    images = np.array(images, dtype=np.uint8)  # Convert to NumPy array
    labels = np.array(labels, dtype=np.uint8)

    print(f"Total images loaded: {len(images)}")

    # Determine a unique file name with versioning
    s3_client = boto3.client("s3")
    existing_files = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=dataset_name)
    existing_versions = []

    if "Contents" in existing_files:
        for obj in existing_files["Contents"]:
            filename = obj["Key"]
            if filename.startswith(dataset_name) and filename.endswith(".h5"):
                parts = filename.replace(".h5", "").split("_v")
                if len(parts) == 2 and parts[1].isdigit():
                    existing_versions.append(int(parts[1]))

    new_version = max(existing_versions, default=0) + 1
    new_hdf5_filename = f"{dataset_name}_v{new_version}.h5"
    hdf5_file_path = os.path.join(LOCAL_SAVE_DIR, new_hdf5_filename)

    print(f"Saving HDF5 file as {new_hdf5_filename}")

    # Save dataset to HDF5 file
    with h5py.File(hdf5_file_path, "w") as hf:
        hf.create_dataset("images", data=images)
        hf.create_dataset("labels", data=labels)

    print(f"HDF5 file saved at {hdf5_file_path}")

    # Push the file path to XCom for use in the next task
    kwargs["ti"].xcom_push(key="hdf5_filename", value=new_hdf5_filename)


def upload_to_s3(**kwargs):
    """Uploads the versioned HDF5 file to an S3 bucket."""

    s3_client = boto3.client("s3")

    # Get the file name from XCom
    ti = kwargs["ti"]
    hdf5_filename = ti.xcom_pull(task_ids="load_images", key="hdf5_filename")

    if not hdf5_filename:
        raise ValueError("No HDF5 file name found in XCom.")

    local_file_path = os.path.join(LOCAL_SAVE_DIR, hdf5_filename)
    s3_key = hdf5_filename  # Use the same name for S3 storage

    # Upload file
    s3_client.upload_file(local_file_path, S3_BUCKET, s3_key)

    print(f"Uploaded {local_file_path} to s3://{S3_BUCKET}/{s3_key}")


# Define DAG
default_args = {"owner": "airflow", "start_date": datetime(2024, 3, 1)}

with DAG("convert_images_to_hdf5", default_args=default_args, schedule_interval=None) as dag:
    load_task = PythonOperator(
        task_id="load_images",
        python_callable=load_images_and_labels,
        op_kwargs={"dataset_name": "mnist"},
    )

    upload_task = PythonOperator(
        task_id="upload_to_s3",
        python_callable=upload_to_s3,
    )

    load_task >> upload_task  # Ensure HDF5 file is created before uploading
