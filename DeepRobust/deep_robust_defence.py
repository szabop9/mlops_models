import os
import sys
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from deeprobust.image.attack.fgsm import FGSM
from deeprobust.image.defense.fgsmtraining import FGSMtraining
import deeprobust.image.netmodels.train_model as trainmodel
import mnist_model  # Import your MNIST model class

# Add DeepRobust to path
sys.path.append('DeepRobust/build/lib')

# Define dataset paths
dataset_path = '../data'
model_save_path = 'trained_models/trained_fgsm_model.pth'

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load or initialize model
cnn = mnist_model.Net().to(device)
if os.path.exists("trained_models/mnist_cnn.pt"):
    cnn.load_state_dict(torch.load("trained_models/mnist_cnn.pt", map_location=device))
else:
    batch_size = 64
    test_batch_size = 1000
    epochs = 1
    lr = 1.0
    gamma = 0.7

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transform),
        batch_size=test_batch_size, shuffle=False
    )

    optimizer = optim.Adadelta(cnn.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    for epoch in range(1, epochs + 1):
        mnist_model.train(10, False, cnn, device, train_loader, optimizer, epoch)
        mnist_model.test(cnn, device, test_loader)
        scheduler.step()

    torch.save(cnn.state_dict(), "trained_models/mnist_cnn.pt")

# Load or train adversiarially trained model
if os.path.exists(model_save_path):
    cnn2 = mnist_model.Net().to(device)
    cnn2.load_state_dict(torch.load(model_save_path))
    print("Model loaded from disk.")
    train_loader, test_loader = trainmodel.feed_dataset('MNIST', dataset_path)
else:
    train_loader, test_loader = trainmodel.feed_dataset('MNIST', dataset_path)
    f = FGSMtraining(cnn, device)
    defense_model = f.generate(train_loader, test_loader, epoch_num=1)
#    torch.save(defense_model.state_dict(), model_save_path)
#    cnn2 = mnist_model.Net().to(device)
#    cnn2.load_state_dict(f.model.state_dict())
    print("Adversarially trained model saved.")

# Evaluate models on a test sample
# xx = datasets.MNIST(dataset_path, download=True).data[998:999].to(device).unsqueeze_(1).float() / 255
# yy = datasets.MNIST(dataset_path, download=True).targets[998:999].to(device)
#
# for i, model in enumerate([cnn, cnn2]):
#     print("With Defense:" if i == 1 else "Without Defense:")
#
#     F1 = FGSM(model, device=device)
#     AdvExArray = F1.generate(xx, yy)
#
#     plt.imshow(AdvExArray.detach()[0].squeeze().cpu().numpy(), cmap="gray")
#     plt.show()
#
#     predict0 = model(xx).argmax(dim=1, keepdim=True)
#     predict1 = model(AdvExArray).argmax(dim=1, keepdim=True)
#
#     print(f"Original prediction: {predict0.item()}")
#     print(f"Attack prediction: {predict1.item()}")
#     print(f"Actual value: {yy.item()}")
