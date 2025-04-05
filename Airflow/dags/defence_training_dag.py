import argparse
from torch.utils.data import TensorDataset, DataLoader
import boto3
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

from art.attacks.evasion import FastGradientMethod
from art.defences.trainer import AdversarialTrainer
from art.estimators.classification import PyTorchClassifier

# from deeprobust.image.defense.fgsmtraining import FGSMtraining

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method


MODEL_SAVE_PATH = "/home/ubuntu/trained_models/"
S3_BUCKET = "ai22m020-models"


def train_art_defence_model(**kwargs):
    conf = kwargs.get("dag_run").conf if kwargs.get("dag_run") else {}
    model_name = conf.get("model_name")
    h5_name = conf.get("h5_name")
    print(f"Training ART defence on model: {model_name}")
    print(f"With data: {h5_name}")

    model = Net()

    state_dict = torch.load(
        model_name, map_location=torch.device('cpu'))

    model.load_state_dict(state_dict)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Wrap the model in an ART PyTorchClassifier
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
    )

    # Load the HDF5 file
    with h5py.File(h5_name, "r") as hf:
        images = hf["images"][:]
        labels = hf["labels"][:]

    # Preprocess the data (normalize just like torchvision)
    images = images.astype(np.float32) / 255.0
    images = (images - 0.1307) / 0.3081
    images = torch.tensor(images).unsqueeze(1)  # Add channel dimension
    labels = torch.tensor(labels)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    x, y = dataloader_to_numpy(loader)

    # Define FGSM attack
    attack = FastGradientMethod(estimator=classifier, eps=4)

    # Create adversarial trainer
    adv_trainer = AdversarialTrainer(classifier, attacks=attack, ratio=0.5)

    # Train with adversarial training
    adv_trainer.fit(x, y, nb_epochs=1)

    classifier = adv_trainer.classifier

    base_name = "art/art_defense_model"

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

    # Save the trained model
    classifier.save(f"{base_name}_v{new_version}", MODEL_SAVE_PATH)

    s3_client.upload_file(f"{MODEL_SAVE_PATH}{base_name}_v{new_version}.model", S3_BUCKET, new_pt_filename)
    s3_uri = f"s3://{S3_BUCKET}/{new_pt_filename}"
    print(f"Uploaded model to {s3_uri}")

    kwargs["ti"].xcom_push(key="base_model", value=model_name)
    kwargs["ti"].xcom_push(key="h5_file", value=h5_name)


def train_deeprobust_defence_model(**kwargs):
    device = torch.device("cpu")

    model_path = kwargs["ti"].xcom_pull(task_ids="train_art_model", key="base_model")
    h5_name = kwargs["ti"].xcom_pull(task_ids="train_art_model", key="h5_file")

    # model = Net()
    #
    # state_dict = torch.load(
    #     model_path, map_location=torch.device('cpu'))
    #
    # model.load_state_dict(state_dict)
    #
    # # Load the HDF5 file
    # with h5py.File(h5_name, "r") as hf:
    #     images = hf["images"][:]
    #     labels = hf["labels"][:]
    #
    # # Preprocess the data (normalize just like torchvision)
    # images = images.astype(np.float32) / 255.0
    # images = (images - 0.1307) / 0.3081
    # images = torch.tensor(images).unsqueeze(1)  # Add channel dimension
    # labels = torch.tensor(labels)
    #
    # # Create a single TensorDataset
    # dataset = TensorDataset(images, labels)
    #
    # # Create one DataLoader that returns (data, label) tuples
    # loader = DataLoader(dataset, batch_size=64, shuffle=True)
    #
    # # x, y = dataloader_to_numpy(loader)
    #
    # f = FGSMtraining(model, device)
    # defense_model = f.generate(loader, loader, epoch_num=1)
    #
    # base_name = "deeprobust/deeprobust_defense_model"
    # s3_client = boto3.client("s3")
    #
    # existing_files = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=base_name)
    # existing_versions = []
    # if "Contents" in existing_files:
    #     for obj in existing_files["Contents"]:
    #         filename = obj["Key"]
    #         if filename.startswith(base_name) and filename.endswith(".pt"):
    #             parts = filename.replace(".pt", "").split("_v")
    #             if len(parts) == 2 and parts[1].isdigit():
    #                 existing_versions.append(int(parts[1]))
    #
    # new_version = max(existing_versions, default=0) + 1
    #
    # new_pt_filename = f"{base_name}_v{new_version}.pt"
    #
    # # Save the trained model
    # torch.save(defense_model.state_dict(), f"{MODEL_SAVE_PATH}{base_name}_v{new_version}")
    #
    # s3_client.upload_file(f"{MODEL_SAVE_PATH}{base_name}_v{new_version}.pt", S3_BUCKET, new_pt_filename)
    # s3_uri = f"s3://{S3_BUCKET}/{new_pt_filename}"
    # print(f"Uploaded model to {s3_uri}")

    kwargs["ti"].xcom_push(key="base_model", value=model_path)
    kwargs["ti"].xcom_push(key="h5_file", value=h5_name)




def train_cleverhans_defence_model(**kwargs):
    device = torch.device("cpu")

    model_path = kwargs["ti"].xcom_pull(task_ids="train_art_model", key="base_model")
    h5_name = kwargs["ti"].xcom_pull(task_ids="train_art_model", key="h5_file")

    model = Net()

    state_dict = torch.load(
        model_path, map_location=torch.device('cpu'))

    model.load_state_dict(state_dict)

    # Load the HDF5 file
    with h5py.File(h5_name, "r") as hf:
        images = hf["images"][:]
        labels = hf["labels"][:]

    # Preprocess the data (normalize just like torchvision)
    images = images.astype(np.float32) / 255.0
    images = (images - 0.1307) / 0.3081
    images = torch.tensor(images).unsqueeze(1)  # Add channel dimension
    labels = torch.tensor(labels)

    # Create a single TensorDataset
    dataset = TensorDataset(images, labels)

    # Create one DataLoader that returns (data, label) tuples
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epsilon = 0.3
    num_epochs = 1  # Number of adversarial training epochs

    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # Generate adversarial examples
            images_adv = generate_adversarial_examples(model, images, labels, epsilon)

            # Combine clean and adversarial examples (50% each)
            images_combined = torch.cat([images, images_adv])
            labels_combined = torch.cat([labels, labels])

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images_combined)
            loss = criterion(outputs, labels_combined)

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    base_name = "cleverhans/cleverhans_defense_model"
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

    # Save the trained model
    torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}{base_name}_v{new_version}.pt")

    s3_client.upload_file(f"{MODEL_SAVE_PATH}{base_name}_v{new_version}.pt", S3_BUCKET, new_pt_filename)
    s3_uri = f"s3://{S3_BUCKET}/{new_pt_filename}"
    print(f"Uploaded model to {s3_uri}")


# Define DAG
default_args = {"owner": "airflow", "start_date": datetime.now()}

with DAG("train_defence_models_ec2", default_args=default_args, schedule_interval=None) as dag:
    train_art_task = PythonOperator(
        task_id="train_art_model",
        python_callable=train_art_defence_model
    )

    train_deeprobust_task = PythonOperator(
        task_id="train_deeprobust_model",
        python_callable=train_deeprobust_defence_model
    )

    train_cleverhans_task = PythonOperator(
        task_id="train_cleverhans_model",
        python_callable=train_cleverhans_defence_model
    )

    trigger_eval_dag = TriggerDagRunOperator(
        task_id="eval_models_task",
        trigger_dag_id="evaluate_models_ec2",
        wait_for_completion=False,
        conf={
            "h5_file": "{{ ti.xcom_pull(task_ids='train_art_task', key='h5_file') }}"
        },
    )

    train_art_task >> train_deeprobust_task >> train_cleverhans_task# >> upload_task


# ----------------------------------------------------------------------------------------
def generate_adversarial_examples(model, images, labels, epsilon):
    images_adv = fast_gradient_method(model, images, epsilon, norm=float("inf"))
    return images_adv

def dataloader_to_numpy(data_loader):
    all_data = []
    all_labels = []
    for data, labels in data_loader:
        all_data.append(data.numpy())  # Convert to numpy
        all_labels.append(labels.numpy())
    x = np.concatenate(all_data, axis=0)  # Combine all batches
    y = np.concatenate(all_labels, axis=0)
    return x, y


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=3064, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "models/mnist_cnn2.pt")
