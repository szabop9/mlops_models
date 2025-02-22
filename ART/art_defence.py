from torch.optim.lr_scheduler import StepLR

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from art.defences.trainer import AdversarialTrainer
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
import mnist_model as mnist_model



from AutomatedDefenceML.Foolbox.FoolPyTorch.mnist_model import Net

def dataloader_to_numpy(data_loader):
    all_data = []
    all_labels = []
    for data, labels in data_loader:
        all_data.append(data.numpy())  # Convert to numpy
        all_labels.append(labels.numpy())
    x = np.concatenate(all_data, axis=0)  # Combine all batches
    y = np.concatenate(all_labels, axis=0)
    return x, y

# Define the PyTorch model
model = Net()

if os.path.exists(os.path.join(os.getcwd(), 'trained_models', 'mnist_cnn.pt')):
    if torch.cuda.is_available():
        state_dict = torch.load(
            "trained_models/mnist_cnn.pt")
    else:
        state_dict = torch.load(
            "trained_models/mnist_cnn.pt", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)

    # Set training parameters
    batch_size = 64
    test_batch_size = 1000
    epochs = 1
    lr = 1.0
    gamma = 0.7
    log_interval = 10
    dry_run = False

    # Define datasets and loaders
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

    # Training setup
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    # Train and test
    for epoch in range(1, epochs + 1):
        mnist_model.train(log_interval, dry_run, model, device, train_loader, optimizer, epoch)  # Pass None for args
        mnist_model.test(model, device, test_loader)
        scheduler.step()

    # Save the model
    torch.save(model.state_dict(), "trained_models/mnist_cnn.pt")
    state_dict = torch.load(
        "trained_models/mnist_cnn.pt")
    model.load_state_dict(state_dict)

# Define loss and optimizer
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
adv_trainer.fit(x, y, nb_epochs=5)

# Ensure the "model" directory exists
os.makedirs("models", exist_ok=True)

# Save the trained model
classifier.save("fgsm_art_defense_model", "models")

# Evaluate on clean and adversarial data
# test_data, test_labels = next(iter(test_loader))
# predictions = classifier.predict(test_data.numpy())
# adversarial_examples = attack.generate(x=test_data.numpy())
# adv_predictions = classifier.predict(adversarial_examples)
#
# # Generate adversarial examples
# adversarial_correct = 0
# adversarial_correct2 = 0
# total = 0
# for data, target in test_loader:
#     data, target = data.to('cpu'), target.to('cpu')
#     adversarial_data = attack.generate(x=data.numpy())
#     predictions = classifier.predict(adversarial_data)
#     predictions2 = model(adversarial_data.data)
#     pred_classes = predictions.argmax(axis=1)
#     pred_classes2 = predictions2.argmax(axis=1)
#     adversarial_correct += (pred_classes == target.numpy()).sum()
#     adversarial_correct2 += (pred_classes2 == target.numpy()).sum()
#     total += target.size(0)
#
# print(f"Accuracy on adversarial data: {100. * adversarial_correct / total:.2f}%")
# print(f"Accuracy on adversarial data (with base model): {100. * adversarial_correct2 / total:.2f}%")


