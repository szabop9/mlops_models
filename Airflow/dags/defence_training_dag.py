import argparse

import boto3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os

from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

from art.attacks.evasion import FastGradientMethod
from art.defences.trainer import AdversarialTrainer
from art.estimators.classification import PyTorchClassifier


MODEL_SAVE_PATH = "/home/ubuntu/trained_models/"
S3_BUCKET = "ai22m020-models"

def train_art_defence_model(**kwargs):
    conf = kwargs.get("dag_run").conf if kwargs.get("dag_run") else {}
    model_name = conf.get("model_name")
    print(f"Training ART defence on model: {model_name}")

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

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    x, y = dataloader_to_numpy(train_loader)

    # Define FGSM attack
    attack = FastGradientMethod(estimator=classifier, eps=0.1)

    # Create adversarial trainer
    adv_trainer = AdversarialTrainer(classifier, attacks=attack, ratio=0.5)

    # Train with adversarial training
    adv_trainer.fit(x, y, nb_epochs=1)

    classifier = adv_trainer.classifier

    # Save the trained model
    classifier.save("fgsm_art_defense_model.pt", MODEL_SAVE_PATH)

    s3_client = boto3.client("s3")
    s3_key = "art/art_defence_model.pt"
    s3_client.upload_file(f"{MODEL_SAVE_PATH}fgsm_art_defense_model.pt", S3_BUCKET, s3_key)

    s3_uri = f"s3://{S3_BUCKET}/{s3_key}"

    print(f"Uploaded model to {s3_uri}")

# def train_deeprobust_defence_model():
#     print("SOMETHING1")
#
# def train_cleverhans_defence_model():
#     print("SOMETHING2")
#
# def upload_defence_model():
#     print("SOMETHING3")

# Define DAG
default_args = {"owner": "airflow", "start_date": datetime.now()}

with DAG("train_defence_models_ec2", default_args=default_args, schedule_interval=None) as  dag:

    train_art_task = PythonOperator(
        task_id="train_art_model",
        python_callable=train_art_defence_model
    )

    # train_deeprobust_task = PythonOperator(
    #     task_id="train_deeprobust_model",
    #     python_callable=train_deeprobust_defence_model
    # )
    #
    # train_cleverhans_task = PythonOperator(
    #     task_id="train_cleverhans_model",
    #     python_callable=train_cleverhans_defence_model
    # )
    #
    # upload_task = PythonOperator(
    #     task_id="upload_defence_models",
    #     python_callable=upload_defence_model,
    #     trigger_rule=TriggerRule.ALL_SUCCESS
    # )

    train_art_task #>> train_deeprobust_task >> train_deeprobust_task >> upload_task


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

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
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

