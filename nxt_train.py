import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import numpy as np
import random

# Paths
train_dir = "../data/Images/train"
weights_output_path = "best_weights.pth"

# Data transformations
transform = transforms.Compose([
    transforms.Grayscale(),  # Ensure images are grayscale
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize grayscale images
])

# Load full dataset
full_dataset = datasets.ImageFolder(train_dir, transform=transform)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(full_dataset.targets), y=full_dataset.targets)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# Define model
print(f"cuda is available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.fc.in_features, len(full_dataset.classes))
)
model = model.to(device)

# Fine-tune deeper layers
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True


# adjust learning rate
mx_lr = 0.001
weight_decay=1e-4
warm_up_epochs = 5
lower_lr_patience = 3
factor = 0.5

# data selection
train_fraction = 0.8
subset_fraction = 0.25  # Use 50% of the training data in each epoch
batch_size = 64

# Training loop
epochs = 30
best_val_loss = float('inf')
early_stop_patience = 5


early_stopping_counter = 0

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


for epoch in range(epochs):
    print(f"\nStart epoch {epoch + 1}/{epochs}")
    
    # Randomly split dataset into training and validation sets
    dataset_size = len(full_dataset)
    indices = np.random.permutation(dataset_size)
    train_size = int(train_fraction * dataset_size)
    train_indices, val_indices = indices[:train_size], indices[train_size:]
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    # Randomly select a subset of training data
    subset_size = int(len(train_subset) * subset_fraction)
    subset_indices = random.sample(range(len(train_subset)), subset_size)
    train_subset = Subset(train_subset, subset_indices)

    # Data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

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

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), f"{epoch+1}_{val_loss}.pth")
        print(f"New best model saved with Validation Loss: {best_val_loss:.4f}")
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stop_patience:
            print("Early stopping triggered")
            break

print("Training complete")
