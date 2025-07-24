
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import copy
from glob import glob
import pandas as pd
from PIL import Image
import torchdata.datapipes as dp
import random
from torch.utils.data.backward_compatibility import worker_init_fn

class WGM_dataset:
    def __init__(self, csv_file, data_transforms, data_type, batch_size):
        self.csv_file = csv_file
        self.data_transforms = data_transforms
        self.batch_size = batch_size
        self.data_type = data_type
        
        
    def open_image(self, inputs):
        _ , wsi_name, img_path, label = inputs
        img = Image.open(img_path)
        return wsi_name, img, int(label)

    def apply_train_transforms(self, inputs):
        _, x, y = inputs
        return self.data_transforms["train"](x), y

    def apply_val_transforms(self, inputs):
        wsi_name, x, y = inputs
        return wsi_name, self.data_transforms["val"](x), y

    def build_data_pipe(self):
        new_dp = dp.iter.FileOpener([self.csv_file])
        new_dp = new_dp.parse_csv(skip_lines=1)
        # returns tuples like ('0','filename', 'filepath', 'label')
        if self.data_type == "train":
            new_dp = new_dp.shuffle()
        
        new_dp = new_dp.sharding_filter()
        # important to use sharding_filter after (not before) shuffling -For the data source that needs to be sharded, it is crucial to add Shuffler before ShardingFilter to ensure data are globally shuffled before being split into shards. Otherwise, each worker process would always process the same shard of data for all epochs. And, it means each batch would only consist of data from the same shard, which leads to low accuracy during training. However, it doesn’t apply to the data source that has already been sharded for each multi-/distributed process, since ShardingFilter is no longer required to be presented in the pipeline.

        new_dp = new_dp.map(self.open_image)

        if self.data_type == "train":
            new_dp = new_dp.map(self.apply_train_transforms)
            new_dp = new_dp.batch(batch_size=self.batch_size, drop_last=True)

        elif self.data_type == "val":
            new_dp = new_dp.map(self.apply_val_transforms)
            new_dp = new_dp.batch(batch_size=self.batch_size, drop_last=False)

        else:
            raise ValueError("Invalid transform argument.")

        new_dp = new_dp.map(torch.utils.data.default_collate)
        if self.data_type == "train":
            loader = torch.utils.data.DataLoader(dataset=new_dp, shuffle=True, num_workers=4)
        if self.data_type == "val":
            loader = torch.utils.data.DataLoader(dataset=new_dp, shuffle=False, num_workers=4)
        return loader

    def dataset_size(self):
        df = pd.read_csv(self.csv_file)
        return len(df)
    
    