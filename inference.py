import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import os
import pandas as pd

# Argument parser
parser = argparse.ArgumentParser(description="Inference script for emotion classification.")
parser.add_argument("--id", type=str, required=True, help="Unique identifier for the submission.")
parser.add_argument("--tar", type=str, required=True, help="Weights file name (without extension).")
args = parser.parse_args()

# Paths
id = args.id
tar = args.tar
path = ""  # f"../submissions/{id}/"
test_dir = "../data/Images/test"  # Update to your test directory path
weights_input_path = f"{path}{tar}.pth"
submission_output_path = f"{path}{id}_{tar}.csv"

# Data transformations
transform = transforms.Compose([
    transforms.Grayscale(),  # Ensure images are grayscale
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize grayscale images
])

# Custom Dataset for test data
class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_paths = sorted([os.path.join(test_dir, fname) for fname in os.listdir(test_dir) if fname.endswith('.jpg')])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)

# Load dataset
test_dataset = TestDataset(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust for grayscale
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),  # Dropout added during inference
    nn.Linear(model.fc.in_features, 7)  # Update for number of classes
)

# Fix for FutureWarning
state_dict = torch.load(weights_input_path, map_location=device)
model.load_state_dict(state_dict, strict=True)

model = model.to(device)
model.eval()

# Inference and submission generation
submission = []
filenames = []
with torch.no_grad():
    for images, image_filenames in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        submission.extend(predicted.cpu().numpy())
        filenames.extend(image_filenames)

# Process filenames to remove ".jpg"
filenames = [fname.replace('.jpg', '') for fname in filenames]

# Create submission DataFrame
submission_df = pd.DataFrame({
    'filename': filenames,
    'label': submission
})

# Save submission file
submission_df.to_csv(submission_output_path, index=False)
print(f"Submission saved to {submission_output_path}")
