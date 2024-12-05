# Key changes:
# Added print statement to display the learning rate at the end of each epoch

if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms, models
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.utils.class_weight import compute_class_weight
    from tqdm import tqdm
    import numpy as np

    # Paths
    train_dir = "../data/Images/train"
    weights_output_path = "best_weights.pth"

    # Data transformations
    transform_train = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    transform_val = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load dataset and split into training/validation sets
    full_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = transform_val

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(full_dataset.targets), y=full_dataset.targets)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    # Define model
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

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Learning rate scheduler with warm-up
    def lr_lambda(epoch):
        if epoch < 5:
            return epoch / 5  # Warm-up
        return 1  # Default after warm-up

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Training loop
    epochs = 30
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

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}")

        # Validation loop
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

        # Scheduler step
        scheduler.step()
        
        # Debugging: Print the learning rate
        print(f"Learning rate for epoch {epoch + 1}: {scheduler.get_last_lr()[0]:.6f}")
