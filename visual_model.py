import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, classification_report
)
import numpy as np
import os

# === CONFIG ===
MODEL_PATH = "labubu_classifier.pth"
DATA_DIR = "dataset"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# === DATASET & LOADER ===
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = dataset.classes  # ['labubu', 'not_labubu']

# === MODEL ===
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# === EVALUATION ===
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# === METRICS ===
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average=None)
recall = recall_score(y_true, y_pred, average=None)
f1 = f1_score(y_true, y_pred, average=None)
cm = confusion_matrix(y_true, y_pred)

# === PRINT METRICS ===
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# === CONFUSION MATRIX ===
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# === BAR CHARTS ===
metrics = {
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}

for metric_name, scores in metrics.items():
    plt.figure(figsize=(6, 4))
    sns.barplot(x=class_names, y=scores)
    plt.title(f"{metric_name} per Class")
    plt.ylim(0, 1)
    plt.ylabel(metric_name)
    plt.tight_layout()
    plt.show()
