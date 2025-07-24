import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np


# --- 1. Load DINO-v2 Backbone ---

def get_dino_v2_backbone():
    # This returns a model directly, not a state_dict
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")  # ✅ this is already a model
    return backbone


class DINOv2SegmentationModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        features = self.backbone.get_intermediate_layers(x, n=1)[0]  # (B, N, 768)
        feat_size = int(features.shape[1] ** 0.5)
        features = features.permute(0, 2, 1).reshape(B, 768, feat_size, feat_size)  # (B, 768, h, w)
        out = self.decoder(features)  # (B, num_classes, h, w)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)  # upscale to 1022x1022
        return out


# --- 3. Dataset ---
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transforms.Compose([
            transforms.Resize((1022, 1022)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        self.mask_transform = transforms.Resize((1022, 1022), interpolation=Image.NEAREST)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])
        img = self.transform(img)
        mask = self.mask_transform(mask)
        mask1 = np.array(mask)
        print(np.unique(mask1))
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        return img, mask
    
    def __len__(self):
        return len(self.image_paths)

# --- 4. Training Loop ---
def train(model, dataloader, device, num_epochs=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")




# Replace these with your actual paths
image_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/Train_val_LB/train_corrected/image"
mask_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/Train_val_LB/train_corrected/mask"
image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])

dataset = SegmentationDataset(image_paths, mask_paths)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

backbone = get_dino_v2_backbone()
model = DINOv2SegmentationModel(backbone, num_classes=3)  # Update `num_classes` as needed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train(model, dataloader, device, num_epochs=20)

torch.save(model.state_dict(), "dino_v2_segmentation_may30.pth")
