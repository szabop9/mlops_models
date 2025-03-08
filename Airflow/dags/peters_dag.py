from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests
import torch
import os
import mnist_model
from torchvision import datasets, transforms

# GitHub Repo Info
GITHUB_REPO = "szabop9/mlops_models"
GITHUB_FOLDER = "Foolbox/FoolPyTorch/models"
GITHUB_TOKEN = "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"
MODEL_STORAGE_PATH = "/tmp/models"  # Local storage for downloaded models

# Ensure model directory exists
os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)


def download_model_from_github(**kwargs):
    """Download the selected model from GitHub and save it locally."""
    #model_name = kwargs["dag_run"].conf.get("model_name")  # User input from Airflow UI
    model_name = "mnist_cnn.pt"
    if not model_name:
        raise ValueError("No model name provided!")

    url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/master/{GITHUB_FOLDER}/{model_name}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        model_path = os.path.join(MODEL_STORAGE_PATH, model_name)
        with open(model_path, "wb") as f:
            f.write(response.content)
        print(f"Model {model_name} downloaded successfully.")
        return model_path
    else:
        raise Exception(f"Failed to download model: {response.text}")


def load_and_evaluate_model(**kwargs):
    """Load the model and evaluate it on adversarial examples."""
    model_path = kwargs["ti"].xcom_pull(task_ids="download_model")

    if not model_path or not os.path.exists(model_path):
        raise ValueError(f"Model file not found at {model_path}")

    # Load the model
    model = mnist_model.Net()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Load MNIST image
    dataset_path = "../../data"
    images = datasets.MNIST(dataset_path, download=True).data[98:99].to('cpu').float() / 255.0
    images = transforms.Normalize((0.1307,), (0.3081,))(images)
    images = images.unsqueeze(1)
    labels = datasets.MNIST(dataset_path, download=True).targets[98:99].to('cpu')

    # Example attack (replace with Foolbox attack if needed)
    with torch.no_grad():
        predictions = model(images)
        predicted_label = predictions.argmax(1).item()

    print(f"PREDICTED LABEL: {predicted_label}")


# Define DAG
default_args = {"owner": "airflow", "start_date": datetime(2024, 2, 1)}

with DAG("select_and_load_model", default_args=default_args, schedule_interval=None) as dag:
    download_task = PythonOperator(
        task_id="download_model",
        python_callable=download_model_from_github,
        provide_context=True
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=load_and_evaluate_model,
        provide_context=True
    )

    download_task >> evaluate_task  # Ensure model is downloaded before evaluation
