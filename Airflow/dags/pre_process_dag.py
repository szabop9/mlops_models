from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import numpy as np
import h5py
from PIL import Image

# Define the base data directory
DATA_BASE_DIR = "../data"


# Function to load PNG images and labels dynamically
def load_images_and_labels(dataset_name, **kwargs):
    dataset_name = "mnist"
    dataset_path = os.path.join(DATA_BASE_DIR, dataset_name, "classes")
    dataset_path = "/mnt/c/dev/sagemaker_airflow_mlflow/AutomatedDefenceML/Airflow/data/mnist/classes"

    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path '{dataset_path}' does not exist.")

    images = []
    labels = []

    # Dynamically find class folders inside "classes"
    class_folders = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

    print(f"Detected classes: {class_folders}")

    for label, class_name in enumerate(class_folders):
        class_folder = os.path.join(dataset_path, class_name)

        for filename in os.listdir(class_folder):
            if filename.endswith(".png"):
                img_path = os.path.join(class_folder, filename)
                img = Image.open(img_path).convert("L")  # Convert to grayscale
                img = img.resize((28, 28))  # Ensure 28x28 resolution
                img_array = np.array(img, dtype=np.uint8)  # Convert to NumPy array

                images.append(img_array)
                labels.append(label)  # Dynamically assigned based on detected class order

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.uint8)

    # Save images and labels to XCom for the next task
    kwargs["ti"].xcom_push(key="images", value=images.tolist())
    kwargs["ti"].xcom_push(key="labels", value=labels.tolist())


# Function to save images & labels as an HDF5 file
def save_to_hdf5(dataset_name, **kwargs):
    images_list = kwargs["ti"].xcom_pull(task_ids="load_images", key="images")
    labels_list = kwargs["ti"].xcom_pull(task_ids="load_images", key="labels")

    if images_list is None or labels_list is None:
        raise ValueError("No data received from XCom!")

    # Convert back to NumPy arrays
    images = np.array(images_list, dtype=np.uint8)
    labels = np.array(labels_list, dtype=np.uint8)

    h5_filename = "mnist_v2.0.h5"
    h5_path = os.path.join("/mnt/c/dev/sagemaker_airflow_mlflow/AutomatedDefenceML/Airflow/data", h5_filename)

    with h5py.File(h5_path, "w") as hf:
        hf.create_dataset("images", data=images)
        hf.create_dataset("labels", data=labels)

    print(f"HDF5 file saved at: {h5_path}")


# Define the DAG
default_args = {"owner": "airflow", "start_date": datetime(2024, 2, 1)}

with DAG(
        "convert_images_to_hdf5",
        default_args=default_args,
        schedule=None,  # Trigger manually
) as dag:
    load_task = PythonOperator(
        task_id="load_images",
        python_callable=load_images_and_labels,
        op_kwargs={"dataset_name": "{{ dag_run.conf.get('dataset_name', 'mnist') }}"},
    )

    save_task = PythonOperator(
        task_id="save_hdf5",
        python_callable=save_to_hdf5,
        op_kwargs={"dataset_name": "{{ dag_run.conf.get('dataset_name', 'mnist') }}"},
    )

    load_task >> save_task  # Ensure load runs before save
