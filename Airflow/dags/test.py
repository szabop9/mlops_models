from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import numpy as np
import h5py
from PIL import Image

# Define the base data directory
DATA_BASE_DIR = "../data"

dataset_name = "mnist"
dataset_path = os.path.join(DATA_BASE_DIR, dataset_name, "classes")
dataset_path = "../data/mnist/classes"

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



if images is None or labels is None:
    raise ValueError("No data received from XCom!")

h5_filename = f"{dataset_name}.h5"
h5_path = os.path.join("../data", h5_filename)

with h5py.File(h5_path, "w") as hf:
    hf.create_dataset("images", data=images)
    hf.create_dataset("labels", data=labels)

print(f"HDF5 file saved at: {h5_path}")