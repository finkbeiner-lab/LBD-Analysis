import torch
import tqdm
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import sys
sys.path.append(os.path.join(os.getcwd(), *tuple(['..'])))
from features import build_features


def compute_mean_std(train_loader, dataset_size,image_size):
    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    # loop through images
    for inputs in train_loader:
        s = inputs[0].squeeze(0)
        psum    += s.sum(axis        = [0, 2, 3])
        psum_sq += (s ** 2).sum(axis = [0, 2, 3])
        
    count =  dataset_size* image_size * image_size

    # mean and std
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    # output
    print('mean: '  + str(total_mean))
    print('std:  '  + str(total_std))
    return total_mean, total_std

def compute_mean_std_lbd(train_loader, dataset_size,image_size):
    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    # loop through images
    for inputs, targets in train_loader:
        #print(len(inputs))
        #print(inputs[0].shape)
        s = inputs[0].unsqueeze(0)
        #print(s.shape)
        psum    += s.sum(axis        = [0, 2, 3])
        psum_sq += (s ** 2).sum(axis = [0, 2, 3])
        
    count =  dataset_size* image_size * image_size

    # mean and std
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    # output
    print('mean: '  + str(total_mean))
    print('std:  '  + str(total_std))
    return total_mean, total_std


collate_fn = lambda _: tuple(zip(*_))
data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
dataset_train_location = '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/Train_val_LB/train_negclass_allBR'
train_dataset = build_features.LBD_Dataset(dataset_train_location, data_transforms,["images","labels"])
train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=4,
            collate_fn=collate_fn)

compute_mean_std_lbd(train_data_loader, 510 , 1024)