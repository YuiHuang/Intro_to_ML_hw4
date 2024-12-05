if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms, models
    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm
    import os

    # Paths
    train_dir = "../data/Images/train"  # Update to your train directory path
    weights_output_path = "best_weights.pth"

    # Data transformations
    transform_train = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load dataset
    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

    # Define model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust for grayscale
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))  # Update for 7 classes
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

    # Save model weights
    torch.save(model.state_dict(), weights_output_path)
    print(f"Model weights saved to {weights_output_path}")
