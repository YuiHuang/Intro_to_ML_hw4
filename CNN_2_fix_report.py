import argparse
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from collections import Counter
import os
import random
import sys
from tqdm import tqdm

# Define a Simple CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 12 * 12, 128),  # Adjusted for 48x48 input
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Training script for emotion classification.")
    parser.add_argument("--id", type=str, required=True, help="Unique identifier for the submission.")
    args = parser.parse_args()

    # Paths and settings
    train_dir = "../data/Images/train"
    id = args.id
    output_path = f"../submissions/{id}/"
    os.makedirs(output_path, exist_ok=True)
    output_file = f"{output_path}log_{id}.txt"

    # Redirect stdout to a file
    original_stdout = sys.stdout
    with open(output_file, "w") as log_file:
        sys.stdout = log_file

        # Hyperparameters
        batch_size = 64
        epochs = 500
        lr = 0.001

        # Data transformations
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Dataset and DataLoader
        full_dataset = datasets.ImageFolder(train_dir, transform=transform)
        indices = list(range(len(full_dataset)))
        random.seed(42)
        random.shuffle(indices)

        train_split = int(0.8 * len(indices))
        train_idx, val_idx = indices[:train_split], indices[train_split:]

        train_loader = DataLoader(
            full_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx), num_workers=2
        )
        val_loader = DataLoader(
            full_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx), num_workers=2
        )

        # Model, Loss, and Optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleCNN(num_classes=len(full_dataset.classes)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training Loop
        best_val_loss = float("inf")
        for epoch in range(epochs):
            # Training Phase
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

            avg_train_loss = running_loss / len(train_loader)

            # Validation Phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / len(val_idx) * 100

            # Logging
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            print(classification_report(all_labels, all_preds, target_names=full_dataset.classes, digits=4, zero_division=0))

            torch.save(model.state_dict(), f"{output_path}{epoch+1}.pth")

            # Save the Best Model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"New best model saved with Val Loss: {avg_val_loss:.4f}")

    sys.stdout = original_stdout  # Restore stdout
