import argparse
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
import os
import time
from tqdm import tqdm
import random
import sys

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="Training script for emotion classification.")
    parser.add_argument("--id", type=str, required=True, help="Unique identifier for the submission.")
    args = parser.parse_args()

    # print("test 1")

    # Paths and settings
    train_dir = "../data/Images/train"  # Update path
    id = args.id
    output_path = f"../submissions/{id}/"
    os.makedirs(os.path.dirname(f"../submissions/{id}"), exist_ok=True)
    os.makedirs(os.path.dirname(f"{output_path}log_{id}.txt"), exist_ok=True)
    output_file = f"{output_path}log_{id}.txt"
    output_file_handle = open(output_file, "w")
    original_stdout = sys.stdout  # Save the original stdout
    sys.stdout = output_file_handle  # Redirect stdout to the file

    batch_size = 128
    epochs = 50
    subset_fraction = 0.8  # Use 80% of the data in each epoch
    random_seed = int(time.time())
    warm_up_epochs = 5

    # Data transformations
    transform = transforms.Compose([
        transforms.Grayscale(),  # Grayscale input
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),  # Augmentation
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize
    ])

    # Dataset and stratified split
    full_dataset = datasets.ImageFolder(train_dir, transform=transform)
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=[sample[1] for sample in full_dataset.samples], random_state=random_seed
    )

    # Debug class distribution
    def calculate_class_distribution(dataset, indices):
        class_counts = Counter(dataset[i][1] for i in indices)
        return {full_dataset.classes[k]: v for k, v in class_counts.items()}

    print("Class distribution for Full Dataset:", calculate_class_distribution(full_dataset, indices))
    print("Class distribution for Validation Dataset:", calculate_class_distribution(full_dataset, val_idx))

    # Define custom CNN model
    class CustomCNN(nn.Module):
        def __init__(self, num_classes):
            super(CustomCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 6 * 6, 256),  # Adjust dimensions based on input size (48x48)
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    # Initialize train and validation loaders
    def create_subset_sampler(dataset, train_indices, subset_fraction):
        subset_size = int(len(train_indices) * subset_fraction)
        subset_indices = random.sample(train_indices, subset_size)  # Randomly sample the subset
        return SubsetRandomSampler(subset_indices)

    # Initialize train_loader with subset sampling
    train_sampler = create_subset_sampler(full_dataset, train_idx, subset_fraction)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )

    # Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomCNN(num_classes=len(full_dataset.classes))
    model = model.to(device)

    # Loss function (Focal Loss)
    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2, weight=None):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.weight = weight

        def forward(self, inputs, targets):
            ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)
            pt = torch.exp(-ce_loss)
            return self.alpha * (1 - pt) ** self.gamma * ce_loss

    class_weights = torch.tensor([1.0 / count for count in calculate_class_distribution(full_dataset, indices).values()])
    class_weights = class_weights.to(device)
    criterion = FocalLoss(weight=class_weights)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-2)
    scheduler = CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-2, step_size_up=10, mode='triangular')

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_sampler = create_subset_sampler(full_dataset, train_idx, subset_fraction)
        train_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation
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

        # Debug logs
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Classification report
        report = classification_report(all_labels, all_preds, target_names=full_dataset.classes, digits=4, zero_division=0)
        print(f"Classification Report:\n{report}")

        # Save model
        torch.save(model.state_dict(), os.path.join(output_path, f"{epoch + 1}.pth"))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best model saved with Val Loss: {avg_val_loss:.4f}")

    print("Training complete")

    # Restore the original stdout
    sys.stdout = original_stdout
    output_file_handle.close()  # Close the file

    # Confirmation
    print(f"Output has been saved to {output_file}")
