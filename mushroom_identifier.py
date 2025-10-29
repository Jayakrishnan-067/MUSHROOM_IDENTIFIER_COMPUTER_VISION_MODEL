
#  DEEP LEARNING MINI PROJECT
# MUSHROOM IDENTIFIER COMPUTER VISION MODEL


#  IMPORT NECESSARY LIBRARIES
import os
import zipfile
import random
import shutil
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


#  STEP 1: UPLOAD AND EXTRACT ZIP DATASET

print(""C:\Users\jj268\Downloads\Mushroom Identifier.v1i.multiclass.zip"...")

from google.colab import files
uploaded = files.upload()

for fname in uploaded.keys():
    zip_path = fname
    print(f"✅ Uploaded {fname}")

# Extract ZIP
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(data_dir)
print("✅ Dataset extracted successfully.")


#  STEP 2: AUTO-DETECT OR SPLIT TRAIN/VAL FOLDERS

import glob

potential_train = glob.glob(os.path.join(data_dir, "**/train"), recursive=True)
potential_val = glob.glob(os.path.join(data_dir, "**/val"), recursive=True)

if not potential_train or not potential_val:
    print(" train/val folders not found — creating automatic split...")

    # Find folder containing images/class folders
    all_folders = [f for f in Path(data_dir).glob("*") if f.is_dir()]
    if len(all_folders) == 1:
        root_images = all_folders[0]
    else:
        root_images = Path(data_dir)

    train_path = Path(data_dir) / "train"
    val_path = Path(data_dir) / "val"
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    for class_folder in root_images.glob("*"):
        if not class_folder.is_dir():
            continue
        imgs = list(class_folder.glob("*"))
        random.shuffle(imgs)
        split_idx = int(0.8 * len(imgs))
        (train_path / class_folder.name).mkdir(parents=True, exist_ok=True)
        (val_path / class_folder.name).mkdir(parents=True, exist_ok=True)
        for img in imgs[:split_idx]:
            shutil.copy(img, train_path / class_folder.name / img.name)
        for img in imgs[split_idx:]:
            shutil.copy(img, val_path / class_folder.name / img.name)
    print("✅ Auto-split 80% train / 20% validation done.")
    train_dir = str(train_path)
    val_dir = str(val_path)
else:
    train_dir = potential_train[0]
    val_dir = potential_val[0]
    print(f"✅ Found train folder: {train_dir}")
    print(f"✅ Found val folder: {val_dir}")


#  STEP 3: DATA TRANSFORMS AND LOADERS

train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_ds = datasets.ImageFolder(train_dir, transform=train_transforms)
val_ds = datasets.ImageFolder(val_dir, transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

classes = train_ds.classes
print(f"📊 Classes detected: {classes}")


#  STEP 4: DEFINE CNN MODEL

class MushroomCNN(nn.Module):
    def __init__(self, num_classes):
        super(MushroomCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MushroomCNN(num_classes=len(classes)).to(device)
print("✅ Model initialized.")


// STEP 5: TRAINING SETUP

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10

train_acc, val_acc, train_loss, val_loss = [], [], [], []


# STEP 6: TRAIN THE MODEL

for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss.append(running_loss / len(train_loader))
    train_acc.append(100 * correct / total)

    // Validation
    model.eval()
    val_running_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss.append(val_running_loss / len(val_loader))
    val_acc.append(100 * val_correct / val_total)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Acc: {train_acc[-1]:.2f}% | Val Acc: {val_acc[-1]:.2f}%")


 # STEP 7: VISUALIZE TRAINING

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_acc, label="Train Acc")
plt.plot(val_acc, label="Val Acc")
plt.title("Accuracy per Epoch")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="Val Loss")
plt.title("Loss per Epoch")
plt.legend()
plt.show()


 STEP 8: EVALUATION METRICS

y_true, y_pred = [], []
model.eval()
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("\n CLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred, target_names=classes))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix")
plt.show()


 STEP 9: VISUALIZE SAMPLE PREDICTIONS

model.eval()
images, labels = next(iter(val_loader))
images, labels = images.to(device), labels.to(device)
outputs = model(images)
_, preds = torch.max(outputs, 1)

plt.figure(figsize=(12,6))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(np.transpose(images[i].cpu().numpy(), (1,2,0)))
    plt.title(f"Pred: {classes[preds[i]]}\nTrue: {classes[labels[i]]}")
    plt.axis("off")
plt.show()

print(" Training complete. Mushroom Identifier ready!")
