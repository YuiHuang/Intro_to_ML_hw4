import argparse
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np
import random
import sys
import os

# Argument parser
parser = argparse.ArgumentParser(description="Inference script for emotion classification.")
parser.add_argument("--id", type=str, required=True, help="Unique identifier for the submission.")
args = parser.parse_args()

# Paths
train_dir = "../data/Images/train"
id = args.id
output_path = f"../submissions/{id}/"
os.makedirs(os.path.dirname(f"../submissions/{id}"), exist_ok=True)
os.makedirs(os.path.dirname(f"{output_path}log_{id}.txt"), exist_ok=True)

# print(f"{output_path}{1}.pth")

# File to save the output
output_file = f"{output_path}log_{id}.txt"
# Open the file and redirect stdout
output_file_handle = open(output_file, "w")
original_stdout = sys.stdout  # Save the original stdout
sys.stdout = output_file_handle  # Redirect stdout to the file


def calculate_class_distribution(dataset, dataset_name):
    class_counts = {class_name: 0 for class_name in full_dataset.classes}
    for _, label in dataset:
        class_name = full_dataset.classes[label]
        class_counts[class_name] += 1
    print(f"\nClass distribution for {dataset_name}:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} images")
    return class_counts


# Data transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load full dataset
full_dataset = datasets.ImageFolder(train_dir, transform=transform)
# calculate_class_distribution(full_dataset, "Original Dataset")

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(full_dataset.targets), y=full_dataset.targets)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# Define model
print(f"cuda is available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Sequential(
    nn.BatchNorm1d(model.fc.in_features),  # Batch Normalization
    nn.Dropout(p=0.5),                     # Regularization after normalization
    nn.Linear(model.fc.in_features, 256),  # Intermediate Fully Connected Layer
    nn.ReLU(),                             # Non-linearity
    nn.Dropout(p=0.3),                     # Regularization after activation
    nn.Linear(256, len(full_dataset.classes))  # Final Fully Connected Layer
)
model = model.to(device)

# Fine-tune deeper layers
for param in model.layer2.parameters():
    param.requires_grad = True
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True


# adjust learning rate
mx_lr = 0.002
weight_decay = 1e-2
warm_up_epochs = 10
lower_lr_patience = 2
factor = 0.8

# data selection
train_fraction = 0.8
subset_fraction = 0.5
batch_size = 64

# Training loop
epochs = 1000
best_val_loss = float('inf')
early_stop_patience = 1000


early_stopping_counter = 0
seed = int(id[5:])

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
optimizer = optim.Adam(model.parameters(), lr=mx_lr, weight_decay=weight_decay)

# Scheduler with warm-up and ReduceLROnPlateau
def lr_lambda(epoch):
    if epoch < warm_up_epochs:
        return epoch / warm_up_epochs  # Warm-up phase
    return 1

warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=lower_lr_patience, factor=factor)

# # Randomly split dataset into training and validation sets
# dataset_size = len(full_dataset)
# indices = np.random.permutation(dataset_size)
# train_size = int(train_fraction * dataset_size)
# train_indices, val_indices = indices[:train_size], indices[train_size:]
# train_subset = Subset(full_dataset, train_indices)
# val_subset = Subset(full_dataset, val_indices)

# Stratified split
def stratified_split(dataset, train_fraction, random_seed=seed):
    print(f"seed: {random_seed}")
    targets = [dataset.targets[i] for i in range(len(dataset))]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_fraction, random_state=random_seed)
    train_indices, val_indices = next(splitter.split(np.zeros(len(targets)), targets))
    return train_indices, val_indices

# Stratified splitting
train_indices, val_indices = stratified_split(full_dataset, train_fraction=train_fraction)

# Subsets
train_subset = Subset(full_dataset, train_indices)
val_subset = Subset(full_dataset, val_indices)
# calculate_class_distribution(val_subset, "Validation Dataset")

val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


for epoch in range(epochs):
    print(f"\nStart epoch {epoch + 1}/{epochs}")
    
    # Randomly select a subset of training data
    subset_size = int(len(train_subset) * subset_fraction)
    random.seed(seed + epoch)
    subset_indices = random.sample(range(len(train_subset)), subset_size)
    cur_train_subset = Subset(train_subset, subset_indices)

    # Data loaders
    train_loader = DataLoader(cur_train_subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # print(f"Training on {len(train_subset)} samples, validating on {len(val_subset)} samples.")

    # Training
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
    print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    print(classification_report(y_true, y_pred, target_names=full_dataset.classes))

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Adjust learning rate
    if epoch < warm_up_epochs:
        warmup_scheduler.step()  # Warm-up phase
    else:
        plateau_scheduler.step(val_loss)  # Reduce LR after warm-up

    # Debug: Print learning rate
    print(f"Learning rate for epoch {epoch + 1}: {optimizer.param_groups[0]['lr']:.6f}")

    torch.save(model.state_dict(), f"{output_path}{epoch+1}.pth")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        print(f"New best model with Validation Loss: {best_val_loss:.4f}")
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stop_patience:
            print("Early stopping triggered")
            break

print("Training complete")

# Restore the original stdout
sys.stdout = original_stdout
output_file_handle.close()  # Close the file

# Confirmation
print(f"Output has been saved to {output_file}")