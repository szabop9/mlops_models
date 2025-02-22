import foolbox as fb
import deeprobust.image.netmodels.CNN as CNN
import torch
from torchvision import datasets, transforms
import mnist_model
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from art.estimators.classification import PyTorchClassifier

models = []
folder_path = "models"
with os.scandir(folder_path) as entries:
    for entry in entries:
        models.append(entry.name)

dataset_path = '../../data'

# Define normalization transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load models
model = mnist_model.Net()
model.load_state_dict(torch.load("models/mnist_cnn.pt", map_location="cpu"))
model.eval()

model2 = mnist_model.Net()
model2.load_state_dict(torch.load("models/mnist_fgsmtraining_0.2.pt", map_location="cpu"))
model2.eval()

model3 = mnist_model.Net()
model3.load_state_dict(torch.load("models/fgsm_art_defense_model.pt", map_location="cpu"))
model3.eval()

model4 = mnist_model.Net()
model4.load_state_dict(torch.load("models/mnist_cleverhans_fgsm.pt", map_location="cpu"))
model4.eval()

# Accuracy Evaluation
images = datasets.MNIST(dataset_path, download=True).data[0:500].to('cpu').float() / 255.0
images = transforms.Normalize((0.1307,), (0.3081,))(images)
images = images.unsqueeze(1)
labels = datasets.MNIST(dataset_path, download=True).targets[0:500].to('cpu')

# Make predictions
with torch.no_grad():
    predictions = model(images).argmax(1)
    predictions2 = model2(images).argmax(1)
    predictions3 = model3(images).argmax(1)
    predictions4 = model4(images).argmax(1)

# Calculate accuracy
accuracy = (predictions == labels).sum().item() / len(labels)
accuracy2 = (predictions2 == labels).sum().item() / len(labels)
accuracy3 = (predictions3 == labels).sum().item() / len(labels)
accuracy4 = (predictions4 == labels).sum().item() / len(labels)

print(
    f"Accuracy of initial model: {accuracy * 100:.2f}%, "
    f"DeepRobust model: {accuracy2 * 100:.2f}%, "
    f"ART model: {accuracy3 * 100:.2f}%, "
    f"CleverHans model: {accuracy4 * 100:.2f}%")

# Prepare Foolbox models
fmodel = fb.PyTorchModel(model, bounds=(-1, 3))
fmodel2 = fb.PyTorchModel(model2, bounds=(-1, 3))
fmodel3 = fb.PyTorchModel(model3, bounds=(-1, 3))
fmodel4 = fb.PyTorchModel(model4, bounds=(-1, 3))

# Define attack
attack = fb.attacks.LinfPGD()
epsilons = [0.00, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]

# Evaluate accury for adv data
# Load and preprocess images and labels
images_eval = datasets.MNIST(dataset_path, download=True).data[0:100].to('cpu').float() / 255.0
images_eval = transforms.Normalize((0.1307,), (0.3081,))(images_eval)
images_eval = images_eval.unsqueeze(1)
labels_eval = datasets.MNIST(dataset_path, download=True).targets[0:100].to('cpu')

# Perform the attack
_, advs_eval, success_eval = attack(fmodel, images_eval, labels_eval, epsilons=epsilons)
_, advs2_eval, success2_eval = attack(fmodel2, images_eval, labels_eval, epsilons=epsilons)
_, advs3_eval, success3_eval = attack(fmodel3, images_eval, labels_eval, epsilons=epsilons)
_, advs4_eval, success4_eval = attack(fmodel4, images_eval, labels_eval, epsilons=epsilons)
accuracy_eval = [0,0,0,0]
for i in range(100):
    for j in range(len(epsilons)):
        original_image = images[0].squeeze().numpy()
        success_pic_eval = success_eval.H[i, j].item()
        success_pic2_eval = success2_eval.H[i, j].item()
        success_pic3_eval = success3_eval.H[i, j].item()
        success_pic4_eval = success4_eval.H[i, j].item()

        if not success_pic_eval:
            accuracy_eval[0] += 1
        if not success_pic2_eval:
            accuracy_eval[1] += 1
        if not success_pic3_eval:
            accuracy_eval[2] += 1
        if not success_pic4_eval:
            accuracy_eval[3] += 1


print(
    "Evaluation of model inferred on adv data:"
    f"Accuracy of initial model: {(accuracy_eval[0]/(100*12)) * 100:.2f}%, "
    f"DeepRobust model: {(accuracy_eval[1]/(100*12)) * 100:.2f}%, "
    f"ART model: {(accuracy_eval[2]/(100*12)) * 100:.2f}%, "
    f"CleverHans model: {(accuracy_eval[3]/(100*12)) * 100:.2f}%")


# Load and preprocess images and labels
images = datasets.MNIST(dataset_path, download=True).data[98:99].to('cpu').float() / 255.0
images = transforms.Normalize((0.1307,), (0.3081,))(images)
images = images.unsqueeze(1)
labels = datasets.MNIST(dataset_path, download=True).targets[98:99].to('cpu')

# Perform the attack
_, advs, success = attack(fmodel, images, labels, epsilons=epsilons)
_, advs2, success2 = attack(fmodel2, images, labels, epsilons=epsilons)
_, advs3, success3 = attack(fmodel3, images, labels, epsilons=epsilons)
_, advs4, success4 = attack(fmodel4, images, labels, epsilons=epsilons)

# Visualization and evaluation loop
pred1 = pred2 = pred3 = pred4 = 0
for i in range(len(epsilons)):
    original_image = images[0].squeeze().numpy()
    success_pic = success.H[0, i].item()
    success_pic2 = success2.H[0, i].item()
    success_pic3 = success3.H[0, i].item()
    success_pic4 = success4.H[0, i].item()

    adversarial_image = advs[i][0].squeeze().numpy()
    adversarial_image2 = advs2[i][0].squeeze().numpy()
    adversarial_image3 = advs3[i][0].squeeze().numpy()
    adversarial_image4 = advs4[i][0].squeeze().numpy()

    with torch.no_grad():
        predicted_label = model(advs[i]).argmax(1).item()
        predicted_label2 = model2(advs2[i]).argmax(1).item()
        predicted_label3 = model3(advs3[i]).argmax(1).item()
        predicted_label4 = model4(advs4[i]).argmax(1).item()

    plt.figure(figsize=(16, 4))
    plt.subplot(1, 5, 1)
    plt.title("Original Image")
    plt.imshow(original_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.title(f"Init, Success: {success_pic} Pred.: {predicted_label}")
    plt.imshow(adversarial_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.title(f"DR, Success: {success_pic2} Pred.: {predicted_label2}")
    plt.imshow(adversarial_image2, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.title(f"ART, Success: {success_pic3} Pred.: {predicted_label3}")
    plt.imshow(adversarial_image3, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.title(f"CH, Success: {success_pic4} Pred.: {predicted_label4}")
    plt.imshow(adversarial_image4, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
