import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time
from torchvision import datasets, transforms, models

import timm
import pandas as pd
import os
from glob import glob
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision import datasets
import pandas as pd
from PIL import Image
from skimage import io, img_as_ubyte
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def plot_curve(train_losses,val_losses,epochs, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1,1+epochs), train_losses, 'bo-', label='Training Loss')
    plt.plot(np.arange(1,1+epochs), val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    # Save the plot to a file
    plt.savefig(save_path, dpi=300)  # or .pdf, .svg, etc.

np.random.seed(0)

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d),
                            "resnet50": models.resnet50(pretrained=False)}

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x
    
    
    
class Dataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        temp_path = self.files_list.iloc[idx, 1]
        label = self.files_list.iloc[idx, 3]
        img = Image.open(temp_path)
        img = transforms.functional.to_tensor(img)
        if self.transform:
            img = self.transform(img)
        return img, label

class ToPIL(object):
    def __call__(self, sample):
        img = sample
        img = transforms.functional.to_pil_image(img)
        return img
    
    
def _get_simclr_pipeline_transform():
    s=1
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 *s, 0.2 * s)
    data_transforms = transforms.Compose([ToPIL(),
                                           transforms.RandomResizedCrop(size=input_shape[0]),
                                           transforms.Resize((input_shape[0],input_shape[1])),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                        #    GaussianBlur(kernel_size=int(0.06 * self.input_shape[0])),
                                            transforms.ToTensor()])
    return data_transforms


def get_train_validation_data_loaders(train_dataset, batch_size,num_workers, valid_size):
    # obtain training indices that will be used for validation
    num_train = len(train_dataset)
    print("num_train",num_train)
    indices = list(range(num_train))
    np.random.shuffle(indices)

    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                num_workers=num_workers, drop_last=True, shuffle=False)
    valid_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler,
                                num_workers=num_workers, drop_last=True) # 
    return train_loader, valid_loader


def get_train_data_loaders(train_dataset, batch_size,num_workers):
    # obtain training indices that will be used for validation
    num_train = len(train_dataset)
    print("num_train",num_train)
    indices = list(range(num_train))
    np.random.shuffle(indices)

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                num_workers=num_workers, drop_last=True, shuffle=False)
    return train_loader


def get_val_data_loaders(val_dataset, batch_size,num_workers):
    # obtain training indices that will be used for validation
    num_val = len(val_dataset)
    print("num_val",num_val)
    indices = list(range(num_val))
    #np.random.shuffle(indices)

    # define samplers for obtaining training and validation batches
    val_sampler = SubsetRandomSampler(indices)

    valid_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                                num_workers=num_workers, drop_last=True) # 
    return valid_loader


class SimCLRClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super(SimCLRClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features, _ = self.encoder(x)
        out = self.classifier(features)
        return out
    
def load_pretrained_simclr_model(model_path):
    
    model = ResNetSimCLR(base_model= "resnet50",out_dim=2 )# .to(self.device)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, device, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for i, (batch_imgs, batch_labels) in enumerate(val_loader):
        batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
        outputs = model(batch_imgs)
        loss = criterion(outputs, batch_labels)
        batch_size = batch_imgs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    avg_loss = total_loss / total_samples
    return avg_loss

def train_classifier(train_loader, val_loader, device, simclr_model, num_classes, num_epochs, model_checkpoints_folder):
    model = SimCLRClassifier(simclr_model, num_classes=num_classes)
    model.to(device)
    #for param in model.encoder.parameters():
    #    param.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    running_loss = 0.0
    total_samples = 0
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 5  # number of epochs to wait
    wait = 0
    for epoch in range(num_epochs):
        model.train()
        for i, (batch_imgs, batch_labels) in enumerate(train_loader):
            batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            #break
            #print(i, batch_imgs.shape, batch_labels.shape, outputs.shape)
            #print("idx", i, "loss", loss.item())
            batch_size = batch_imgs.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
        
        scheduler.step()
        avg_loss = running_loss / total_samples
        train_losses.append(avg_loss)
        val_loss  = evaluate_model(model, device, val_loader,criterion)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Val Loss:{val_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'using_resnet1_model_train_val_new.pth'))
        print('saved')
            # --- Early Stopping Check ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            best_model_state = model.state_dict()  # save best model
        else:
            wait += 1
            print(f"  ↳ No improvement for {wait}/{patience} epochs.")
            if wait >= patience:
                print("Early stopping triggered.")
                break
    return model, train_losses, val_losses


def val_classifier(val_loader, device,model,class_names):
    model.eval()
    all_preds = []
    all_labels = []
    for i, (batch_imgs, batch_labels) in enumerate(val_loader):
        batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
        outputs = model(batch_imgs)
        y_pred =  outputs.argmax(axis=1)
        all_preds.extend(y_pred.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())
    # Report
    print(confusion_matrix(all_labels, all_preds))
    print(classification_report(all_labels, all_preds))
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    print(report)
    #report.to_csv('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/crop_classifier/report.csv')



    
if __name__ == '__main__':
    model_path  = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/feature_extractor/runs/May23_19-38-08_kif-gh200-02.gladstone.internal/checkpoints/model.pth"
    #model_path  = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/feature_extractor/runs/May30_10-28-40_kif-gh200-04.gladstone.internal/checkpoints/model.pth"
    patches = glob(os.path.join("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/antibodies_data/tiled_slide/*/20.0","*.jpeg"))
    df  = pd.DataFrame(patches, columns = ["path"])
    df["class"] = df['path'].apply(lambda l: "DLB" if l.find("DLB")!=-1 else "PDD")
    df["label"]=df['class'].apply(lambda l: 1 if l=="DLB" else 0)
    #df.to_csv("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/crop_classifier/all_patches_oxford_updated.csv")
    input_shape = (224,224,3)
    batch_size = 32
    num_workers = 8
    valid_size =  0.3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_transforms = _get_simclr_pipeline_transform()
    train_dataset = Dataset(csv_file='/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/crop_classifier/train.csv', transform=data_transforms)
    val_dataset = Dataset(csv_file='/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/crop_classifier/val.csv', transform=data_transforms)

    train_loader = get_train_data_loaders(train_dataset, batch_size,num_workers)
    valid_loader  = get_val_data_loaders(val_dataset, batch_size,num_workers)

    #train_loader, valid_loader = get_train_validation_data_loaders(train_dataset, batch_size, num_workers, valid_size)
    simclr_model = load_pretrained_simclr_model(model_path)
    
    num_classes = 2
    num_epochs = 40
    model_checkpoints_folder = '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/crop_classifier/saved_models'
    model, train_losses, val_losses = train_classifier(train_loader, valid_loader, device, simclr_model, num_classes, num_epochs, model_checkpoints_folder)
    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'using_resnet1_model_train_val_new.pth'))
    print('final saved')
    
    class_names = ["PDD","DLB"]
    val_classifier(valid_loader, device,model,class_names)
    
    
    save_path =  '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/crop_classifier/using_resnet1_model_train_val_new_loss_curve.png'
    plot_curve(train_losses,val_losses,len(train_losses), save_path)
    
    
    
    

