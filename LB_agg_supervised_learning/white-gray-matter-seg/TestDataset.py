import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import numpy as np
import torch

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img_list):
        super(TestDataset, self).__init__()
        self.img_list = img_list
        self.transforms = transforms.Compose([
        transforms.Resize(1024),
        transforms.ToTensor(),
        transforms.Normalize([-0.0601, -0.0667, -0.0616],[0.9118, 0.9479, 0.9625])
        #transforms.Normalize([0.8478, 0.8328, 0.8679], [0.0932, 0.0973, 0.0825])
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    def __len__(self):
        return len(self.img_list)
    
    def crop2img(self, crop):
        crop_array = crop.numpy()
        crop_array = crop_array[:,:,:3]
        img = Image.fromarray(crop_array.astype('uint8'), 'RGB')
        return img
    
    def __getitem__(self, idx):
        img = self.img_list[idx]
        img = self.crop2img(img)
        transformed_img = self.transforms(img)
        transformed_img = torch.unsqueeze(transformed_img, dim=0)
        return transformed_img