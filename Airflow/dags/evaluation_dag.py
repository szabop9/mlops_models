import argparse
from datetime import datetime

import foolbox as fb

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import boto3
import os

S3_BUCKET = "ai22m020-models"
MODEL_SAVE_PATH = "/home/ubuntu/eval/"

def evaluate_models(**kwargs):
    conf = kwargs.get("dag_run").conf if kwargs.get("dag_run") else {}
    h5_file = conf.get("h5_file")

    art_path = conf.get("art_path")
    deeprobust_path = conf.get("deeprobust_path")
    cleverhans_path = conf.get("cleverhans_path")




    s3_client = boto3.client("s3")

    art_local_path = os.path.join(MODEL_SAVE_PATH, os.path.basename(art_path))
    deeprobust_local_path = os.path.join(MODEL_SAVE_PATH, os.path.basename(deeprobust_path))
    ch_local_path = os.path.join(MODEL_SAVE_PATH, os.path.basename(cleverhans_path))

    s3_client.download_file(S3_BUCKET, art_path, art_local_path)
    s3_client.download_file(S3_BUCKET, deeprobust_path, deeprobust_local_path)
    s3_client.download_file(S3_BUCKET, cleverhans_path, ch_local_path)

    art_model = Net()
    cleverhans_model = Net()
    deeprobust_model = Net()

    state_dict_art = torch.load(
        art_local_path, map_location=torch.device('cpu'))
    state_dict_deeprobust = torch.load(
        deeprobust_local_path, map_location=torch.device('cpu'))
    state_dict_cleverhans = torch.load(
        ch_local_path, map_location=torch.device('cpu'))

    art_model.load_state_dict(state_dict_art)
    deeprobust_model.load_state_dict(state_dict_deeprobust)
    cleverhans_model.load_state_dict(state_dict_cleverhans)

    art_model.eval()
    deeprobust_model.eval()
    cleverhans_model.eval()

    with h5py.File(h5_file, 'r') as hf:
        images = hf['images'][:300]
        labels = hf['labels'][:300]

    images = images.astype(np.float32) / 255.0  # Normalize to [0,1]
    images = (images - 0.1307) / 0.3081  # Further normalize
    images = torch.tensor(images).unsqueeze(1)  # Add channel
    labels = torch.tensor(labels)

    # Evaluation on clean images
    with torch.no_grad():
        pred_art = art_model(images).argmax(1)
        pred_deeprobust = deeprobust_model(images).argmax(1)
        pred_ch = cleverhans_model(images).argmax(1)

    acc_art = (pred_art == labels).sum().item() / len(labels)
    acc_deeprobust = (pred_deeprobust == labels).sum().item() / len(labels)
    acc_ch = (pred_ch == labels).sum().item() / len(labels)

    print("Clean image accuracy:")
    print(f"ART: {acc_art * 100:.2f}% | DeepRobust: {acc_deeprobust * 100:.2f}% | CleverHans: {acc_ch * 100:.2f}%")

    fmodel_art = fb.PyTorchModel(art_model, bounds=(-1, 3))
    fmodel_deeprobust = fb.PyTorchModel(deeprobust_model, bounds=(-1, 3))
    fmodel_cleverhans = fb.PyTorchModel(cleverhans_model, bounds=(-1, 3))

    epsilons = [0.00, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    attack = fb.attacks.LinfPGD()

    _, _, success_art = attack(fmodel_art, images, labels, epsilons=epsilons)
    _, _, success_deeprobust = attack(fmodel_deeprobust, images, labels, epsilons=epsilons)
    _, _, success_cleverhans = attack(fmodel_cleverhans, images, labels, epsilons=epsilons)

    acc_art = (~success_art).float().mean().item()
    acc_deeprobust = (~success_deeprobust).float().mean().item()
    acc_cleverhans = (~success_cleverhans).float().mean().item()

    print("Adversarial Evaluation (mean over epsilons):")
    print(f"ART Model Accuracy:        {acc_art * 100:.2f}%")
    print(f"DeepRobust Model Accuracy: {acc_deeprobust * 100:.2f}%")
    print(f"CleverHans Model Accuracy: {acc_cleverhans * 100:.2f}%")





default_args = {"owner": "airflow", "start_date": datetime.now()}
with DAG("evaluate_models_ec2", default_args=default_args, schedule_interval=None, catchup=False) as dag:
    eval_models_task = PythonOperator(
        task_id="evaluate_models",
        python_callable=evaluate_models,
    )

    eval_models_task

#----------------

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

