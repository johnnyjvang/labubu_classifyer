import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# === CONFIG ===
DATA_DIR = "dataset"  # Must contain subfolders 'labubu_augmented' and 'not_labubu_augmented'
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "labubu_classifier.pth"

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Basic normalization (or use ImageNet mean/std)
])

# === DATASET ===
train_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# === MODEL ===
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
model = model.to(DEVICE)

# === TRAINING ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()

    acc = correct / len(train_dataset)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

# === SAVE THE TRAINED MODEL ===
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
