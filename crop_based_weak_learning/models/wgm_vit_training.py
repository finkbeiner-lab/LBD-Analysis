import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time
from torchvision.datasets import ImageFolder




# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2  # Change this according to your dataset
batch_size = 32
learning_rate = 1e-4
num_epochs = 10

# Data transforms
transform = transforms.Compose([
     transforms.ColorJitter(brightness=.3, contrast=0.3 ,saturation=0.3,  hue=.3),
    transforms.Resize((224, 224)),  # ViT expects 224x224 input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Datasets
train_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/WM_images3/train"
val_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/WM_images3/val"

train_dataset = ImageFolder(root=train_dir, transform=transform)
val_dataset = ImageFolder(root=val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Load pretrained ViT model
model = models.vit_b_16(pretrained=True)  # Use vit_l_16 for larger variant
#weights = ViT_G_16_Weights.IMAGENET1K_V1
#model = vit_g_16(weights=weights)
model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)  # Replace classification head
model = model.to(device)

print("loaded model")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_acc = 100. * correct / total
    print(f"Validation Accuracy: {val_acc:.2f}%\n")
    
torch.save(model.state_dict(), "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/models/wgm_vit_model/vit_classifier_kif_10epoch_16bs.pth")


# Test dataset and loader
test_dir =  "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/WM_images3/test"
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False)
num_classes = len(test_dataset.classes)
model.eval()

# Evaluation
correct = 0
total = 0
class_correct = [0 for _ in range(num_classes)]
class_total = [0 for _ in range(num_classes)]

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(labels.size(0)):
            label = labels[i]
            pred = predicted[i]
            class_total[label] += 1
            if label == pred:
                class_correct[label] += 1

# Accuracy
overall_acc = 100. * correct / total
print(f"\n✅ Overall Test Accuracy: {overall_acc:.2f}%")

# Per-class accuracy
for i, cls in enumerate(test_dataset.classes):
    acc = 100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    print(f"🔹 {cls}: {acc:.2f}%")


