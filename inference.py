import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import os
import argparse
import pandas as pd
from tqdm import tqdm

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
model = models.efficientnet_b0(weights=None)  # No pretrained weights for inference
model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)  # Grayscale adjustment
model.classifier[1] = nn.Sequential(
    nn.BatchNorm1d(1280),
    nn.Dropout(p=0.5),
    nn.Linear(1280, 7)  # Update to match the number of classes
)
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

# Process filenames to remove ".jpg"
filenames = [fname.replace('.jpg', '') for fname in filenames]

# Save predictions to CSV
print(f"Saving predictions to {submission_output_path}")
output_df = pd.DataFrame({
    "filename": filenames,
    "label": predictions
})
output_df.to_csv(submission_output_path, index=False)
print("Inference complete!")
