import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
import mnist_model  # Import your MNIST model class

# Load Pretrained Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mnist_model.Net().to(device)
model.load_state_dict(torch.load("trained_models/mnist_cnn.pt", map_location=device))
model.train()  # Set to training mode

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST Dataset
train_dataset = datasets.MNIST("../data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# FGSM attack parameters
epsilon = 0.3  # Adjust as needed


def generate_adversarial_examples(model, images, labels, epsilon):
    images_adv = fast_gradient_method(model, images, epsilon, norm=float("inf"))
    return images_adv


# Adversarial Training Loop
num_epochs = 3  # Number of adversarial training epochs

for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in train_loader:
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

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

# Save the Adversarially Trained Model
torch.save(model.state_dict(), "models/mnist_cleverhans_fgsm.pt")
print("Adversarially trained model with CleverHans saved successfully!")