if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms, models
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.utils.class_weight import compute_class_weight
    from tqdm import tqdm
    import numpy as np
    import os

    # Paths
    train_dir = "../data/Images/train"  # Update to your training data directory
    weights_output_path = "best_weights.pth"

    # Data transformations
    transform_train = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load dataset
    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(train_dataset.targets), y=train_dataset.targets)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    # Define model
    print(f"cuda is available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),  # Dropout before final layer
        nn.Linear(model.fc.in_features, len(train_dataset.classes))
    )
    model = model.to(device)

    # Freeze early layers and fine-tune deeper layers
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.004, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    # Training loop
    epochs = 30  # Increased number of epochs
    best_val_loss = float('inf')
    patience = 5
    early_stopping_counter = 0

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
        print(f"Epoch {epoch + 1}, Loss: {avg_train_loss:.4f}")

        # Scheduler step
        scheduler.step(avg_train_loss)

        # Early stopping (if using validation, this would use val_loss instead)
        if avg_train_loss < best_val_loss:
            best_val_loss = avg_train_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), weights_output_path)  # Save best model
            print(f"New best model saved with Loss: {best_val_loss:.4f}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered")
                break

    print("Training complete")
