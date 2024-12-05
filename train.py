import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

# Paths
train_dir = "../data/Images/train"  # Update to your train directory path
weights_output_path = "trained_weights.pth"

# Data transformations
transform = transforms.Compose([
    transforms.Grayscale(),  # Ensure images are grayscale
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize grayscale images
])

# Load dataset
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define model (Transfer Learning with ResNet18)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust for grayscale
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))  # Update for number of classes
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Lower learning rate for better fine-tuning

# Training loop
epochs = 10
for epoch in range(epochs):
    print(f"Start epoch {epoch + 1}/{epochs}")
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

# Save model weights
torch.save(model.state_dict(), weights_output_path)
print(f"Model weights saved to {weights_output_path}")
