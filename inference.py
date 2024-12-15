import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import os
import argparse
import pandas as pd
from tqdm import tqdm

# Define the same SimpleCNN model as in train.py
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
            nn.Linear(32 * 12 * 12, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# Argument parser
parser = argparse.ArgumentParser(description="Inference script for emotion classification.")
parser.add_argument("--id", type=str, required=True, help="Unique identifier for the submission.")
parser.add_argument("--tar", type=str, required=True, help="Weights file name (without extension).")
args = parser.parse_args()

# Paths
id = args.id
tar = args.tar
path = f"../submissions/{id}/"
test_dir = "../data/Images/test"  # Update to your test directory path
weights_input_path = f"{path}{tar}.pth"
submission_output_path = f"{path}{id}_{tar}.csv"

# Device setup
print(f"cuda is available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations (same as in training)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load model
model = SimpleCNN(num_classes=7)  # Update the number of classes as in training
model = model.to(device)

# Load weights
print(f"Loading model weights from {weights_input_path}")
model.load_state_dict(torch.load(weights_input_path, map_location=device))
model.eval()

predictions = []
filenames = []

with torch.no_grad():
    for img_name in tqdm(sorted(os.listdir(test_dir))):
        img_path = os.path.join(test_dir, img_name)
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Load and preprocess image
            image = Image.open(img_path).convert("L")  # Convert to grayscale
            image = transform(image).unsqueeze(0).to(device)

            # Forward pass
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

            # Save results
            predictions.append(predicted.item())
            filenames.append(img_name)

# Process filenames to remove extensions
filenames = [fname.rsplit('.', 1)[0] for fname in filenames]

# Save predictions to CSV
print(f"Saving predictions to {submission_output_path}")
output_df = pd.DataFrame({
    "filename": filenames,
    "label": predictions
})
output_df.to_csv(submission_output_path, index=False)
print("Inference complete!")
