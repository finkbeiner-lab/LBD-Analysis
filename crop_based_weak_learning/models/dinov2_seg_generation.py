import torch
import os
import sys
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from skimage.measure import regionprops, label
import pandas as pd


class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None, device=None):
        self.img_csv = pd.read_csv(os.path.join(img_dir,"dataset.csv"))
        self.img_dir = img_dir
        self.transform = transform
        self.device=device

    def __len__(self):
        return len(self.img_csv)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_csv.iloc[idx, 2].split("/")[-1])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image).to(device)
        img_name = self.img_csv.iloc[idx, 2]
        return image, img_name

def get_dino_v2_backbone():
    # This returns a model directly, not a state_dict
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")  # ✅ this is already a model dinov2_vitb14 #dinov2_vitg14
    return backbone

class DINOv2SegmentationModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1), #768
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        features = self.backbone.get_intermediate_layers(x, n=1)[0]  # (B, N, 768)
        feat_size = int(features.shape[1] ** 0.5)
        features = features.permute(0, 2, 1).reshape(B, 768, feat_size, feat_size)  # (B, 768, h, w) #768
        out = self.decoder(features)  # (B, num_classes, h, w)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)  # upscale to 1022x1022
        return out

def extract_features(pred_masks_batch, img_names):
    features = []
    for pred_masks, img_name in zip(pred_masks_batch,img_names):
        classes_pred = np.unique(pred_masks)
        if len(classes_pred)==1:
            continue
        
        for i in sorted(classes_pred)[1:]:
            pred_mask = pred_masks==i
            labeled_mask = label(pred_mask)
            props = regionprops(labeled_mask)
            for region in props:
                feats = {
                    'slide_id':img_name.split("/")[0],
                    'img_name':img_name,
                    'area': region.area,
                    'perimeter': region.perimeter,
                    'eccentricity': region.eccentricity,
                    'major_axis_length': region.major_axis_length,
                    'minor_axis_length': region.minor_axis_length,
                    'solidity': region.solidity,
                    'extent': region.extent,
                    'aspect_ratio': region.major_axis_length / region.minor_axis_length if region.minor_axis_length != 0 else 0,
                    'circularity': (4 * np.pi * region.area) / (region.perimeter ** 2) if region.perimeter > 0 else 0,
                    'bbox_xmin': region.bbox[0],
                    'bbox_xmax': region.bbox[2],
                    'bbox_ymin': region.bbox[1],
                    'bbox_ymax': region.bbox[3],
                    'centroid_x': region.centroid[0],
                    'centroid_y': region.centroid[1],
                    'label':region.label,
                    'class':i
                }
                features.append(feats)
    return features


def run_prediction(model, test_dataloader,output_path):
    features_all = []
    with torch.no_grad():
        for image_batch, img_names in test_dataloader:
            output = model(image_batch)  # (1, num_classes, 1022, 1022)
            pred_masks = output.argmax(dim=1).squeeze(0).cpu().numpy()  # (1022, 1022)
            #print(pred_masks.shape)
            features = extract_features(pred_masks, img_names)
            features_all.extend(features)
    df = pd.DataFrame(features_all)
    print(len(df))
    df.to_csv(output_path)

def load_model(path):
    backbone = get_dino_v2_backbone()
    model = DINOv2SegmentationModel(backbone, num_classes=4)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()  # 🔍 Important for inference
    return model    
    
if __name__ == "__main__":    
    
    #path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/kif_codes/202505082023555epoch_dino_v2_segmentation.pth"
    path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/kif_codes/20250502225532dino_v2_segmentation.pth"
    tile_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/tiles/output/"
    save_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/crop_based/seg_results"
    folders  = glob(os.path.join(tile_dir,"*.svs")) 
    oxford_data = pd.read_csv("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/data_imperial2.csv")
    oxford_slides = oxford_data["slide_id"].values    
    folders = [os.path.join(tile_dir,x) for x in oxford_slides]
    completed = glob(os.path.join(save_dir,"*.csv"))
    completed = [x.split("/")[-1] for x in completed]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = 16
    
    model = load_model(path)

    seg_transform = transform = transforms.Compose([
            transforms.Resize((1022, 1022)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    for i in range(len(folders)): # running one instance starting 13
        if folders[i].split("/")[-1]+".csv" not in completed:
            print("running for", folders[i])
            dataset = TestDataset(folders[i], seg_transform, device)
            test_dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)
            output_path = os.path.join(save_dir, folders[i].split("/")[-1]+".csv")
            run_prediction(model, test_dataloader, output_path )
            print("run complete")


        

