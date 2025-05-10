import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# === CONFIG ===
MODEL_PATH = "labubu_classifier.pth"         # Path to your saved model
DATA_DIR = "dataset"             # Dataset root directory
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === TRANSFORMS (same as training) ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === LOAD DATA ===
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# === LOAD MODEL ===
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # Two classes: labubu vs not_labubu
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# === EVALUATION ===
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# === METRICS ===
print("\n--- Evaluation Results ---")
print("Accuracy: {:.2f}%".format(100 * np.mean(np.array(all_preds) == np.array(all_labels))))
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=dataset.classes))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
