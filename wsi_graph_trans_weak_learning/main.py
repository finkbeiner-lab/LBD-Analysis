#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from utils.dataset import GraphDataset
from utils.lr_scheduler import LR_Scheduler
from tensorboardX import SummaryWriter
from helper import Trainer, Evaluator, collate
from option import Options

# from utils.saliency_maps import *

from models.GraphTransformer import Classifier
from models.weight_init import weight_init
import pandas as pd
import matplotlib.pyplot as plt

def plot_curve(train_losses,val_losses,epochs, save_path, indicator="loss"):
    if indicator=="loss":
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
    else:
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(1,1+epochs), train_losses, 'bo-', label='Training Acc')
        plt.plot(np.arange(1,1+epochs), val_losses, 'ro-', label='Validation Acc')
        plt.title('Training and Validation Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        # Save the plot to a file
        plt.savefig(save_path, dpi=300)  # or .pdf, .svg, etc.
        


#args = Options().parse()
n_class = 2

torch.cuda.synchronize()
torch.backends.cudnn.deterministic = True

#data_path = args.data_path 
data_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/antibodies_data"

#model_path = args.model_path

model_path =  "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/save_models"
if not os.path.isdir(model_path): os.mkdir(model_path)

#log_path = args.log_path
log_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/save_logs"

if not os.path.isdir(log_path): os.mkdir(log_path)


#task_name = args.task_name

task_name = "LBD_classification_oxford_epoch_50_lr_1e-4_wc_4e-3_resnet_simclr1_jun5_train0"

print(task_name)
###################################
#train = args.train
train = False
#test = args.test
test = True
#graphcam = args.graphcam
graphcam = True
save_metrics = False

log_interval_local=1

print("train:", train, "test:", test, "graphcam:", graphcam)

##### Load datasets
print("preparing datasets and dataloaders......")
batch_size = 1

#train_set = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/train_oxford_new.txt"

#val_set = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/test_oxford.txt"

#val_set = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/val_oxford_new.txt"

train_set = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/antibodies_data/train3.txt"

val_set = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/antibodies_data/all_train.txt"


num_epochs  = 1
lr = 0.0001
n_features = 2048 #512


#resume = False
#resume = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/save_models/LBD_classification_2.pth"
#resume = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/save_models/LBD_classification_oxford_epoch_15_lr_1e-4.pth"
#resume =  "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/save_models/LBD_classification_oxford_epoch_75_lr_1e-4_resnet50.pth"

#resume="/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/save_models/LBD_classification_oxford_epoch_75_lr_1e-4_resnet50_new_model.pth"
#resume =  "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/save_models/LBD_classification_oxford_epoch_50_lr_1e-4_wc_4e-3_resnet50_june2_old_train_new_val.pth"
resume = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/save_models/LBD_classification_oxford_epoch_50_lr_1e-4_wc_4e-3_resnet_simclr1_jun5_train0.pth"

if train:
    ids_train = open(train_set).readlines()
    dataset_train = GraphDataset(os.path.join(data_path, ""), ids_train)
    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=10, collate_fn=collate, shuffle=True, pin_memory=True, drop_last=True)
    total_train_num = len(dataloader_train) * batch_size

ids_val = open(val_set).readlines()
dataset_val = GraphDataset(os.path.join(data_path, ""), ids_val)
dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, num_workers=10, collate_fn=collate, shuffle=False, pin_memory=True)
total_val_num = len(dataloader_val) * batch_size
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device",device)
##### creating models #############
print("creating models......")

num_epochs = num_epochs
learning_rate = lr

model = Classifier(n_class, n_features= n_features)
model = nn.DataParallel(model)
if resume:
    print('load model{}'.format(resume))
    model.load_state_dict(torch.load(resume))

if torch.cuda.is_available():
    model = model.cuda()
#model.apply(weight_init)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 4e-3)       # best:5e-4, 4e-3
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,100], gamma=0.1) # gamma=0.3  # 30,90,130 # 20,90,130 -> 150

##################################

criterion = nn.CrossEntropyLoss()

if not test:
    writer = SummaryWriter(log_dir=os.path.join(log_path,task_name))
    f_log = open(log_path + task_name + ".log", 'w')

trainer = Trainer(n_class)
evaluator = Evaluator(n_class)

best_pred = None
train_losses = []
val_losses = []
train_accs = []
val_accs = []
for epoch in range(num_epochs):
    # optimizer.zero_grad()
    model.train()
    train_loss = 0.
    total = 0.
    val_loss = 0.
    current_lr = optimizer.param_groups[0]['lr']
    print('\n=>Epoches %i, learning rate = %.7f' % (epoch+1, current_lr) + (', previous best = %.4f' % best_pred if (best_pred is not None) else ''))

    if train:
        for i_batch, sample_batched in enumerate(dataloader_train):
            #scheduler(optimizer, i_batch, epoch, best_pred)
            scheduler.step(epoch)

            preds,labels,loss = trainer.train(sample_batched,ids_train[i_batch].split("\t")[0], model, n_features=n_features)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            total += len(labels)
            
            trainer.metrics.update(labels, preds)
            #trainer.plot_cm()
             
            
            if (i_batch + 1) % log_interval_local == 0:
                print("[%d/%d] train loss: %.3f; agg acc: %.3f" % (total, total_train_num, train_loss / total, trainer.get_scores()))
                trainer.plot_cm()
            
            torch.save(model.state_dict(), os.path.join(model_path, task_name + ".pth"))
        train_loss_save = train_loss.detach().cpu().numpy()
        train_losses.append(train_loss_save / total)
        train_accs.append(trainer.get_scores())
        
    if not test: 
        print("[%d/%d] train loss: %.3f; agg acc: %.3f" % (total_train_num, total_train_num, train_loss / total, trainer.get_scores()))
        trainer.plot_cm()


    if (epoch) % 1 == 0:
        with torch.no_grad():
            model.eval()
            print("evaluating...")

            total = 0.
            batch_idx = 0
            pred_labels = []
            act_labels = []
            filename = []
            
            for i_batch, sample_batched in enumerate(dataloader_val):
                #pred, label, _ = evaluator.eval_test(sample_batched, model)
                preds, labels,loss = evaluator.eval_test(sample_batched, ids_val[i_batch].split("\t")[0], model, graphcam, n_features= n_features)
                #print(preds)
                pred_labels.append(preds.cpu().numpy()[0])
                act_labels.append(labels.cpu().numpy()[0])
                filename.append(ids_val[i_batch].split("\t")[0])
                val_loss += loss
                total += len(labels)

                evaluator.metrics.update(labels, preds)

                if (i_batch + 1) % log_interval_local == 0:
                    print('[%d/%d] val agg acc: %.3f' % (total, total_val_num, evaluator.get_scores()))
                    evaluator.plot_cm()
            
            val_losses.append(val_loss.detach().cpu().numpy() / total)
            
            print('[%d/%d] val agg acc: %.3f' % (total_val_num, total_val_num, evaluator.get_scores()))
            evaluator.plot_cm()
            print(len(pred_labels))
            print(len(act_labels))
            #print(len(filename))
            pd.DataFrame({"input":filename,"act_labels":act_labels,"pred_labels":pred_labels}).to_csv("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/figures/"+resume.split("/")[-1].replace(".pth","pred_all.csv"))

            # torch.cuda.empty_cache()
            val_acc = evaluator.get_scores()
            val_accs.append(val_acc)
            if (best_pred is None) or (val_acc > best_pred): 
                best_pred = val_acc
                if not test:
                    print("saving model...")
                    torch.save(model.state_dict(), os.path.join(model_path, task_name + ".pth"))

            log = ""
            log = log + 'epoch [{}/{}] ------ acc: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, trainer.get_scores(), evaluator.get_scores()) + "\n"

            log += "================================\n"
            print(log)
            if test: break

            f_log.write(log)
            f_log.flush()

            writer.add_scalars('accuracy', {'train acc': trainer.get_scores(), 'val acc': evaluator.get_scores()}, epoch+1)

    trainer.reset_metrics()
    evaluator.reset_metrics()
    torch.cuda.empty_cache()

if not test: f_log.close()
if save_metrics:
    plot_curve(train_losses,val_losses,num_epochs, "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/figures/"+task_name+"_loss_curve.png")
    plot_curve(train_accs,val_accs,num_epochs, "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/figures/"+task_name+"_acc_curve.png",indicator="acc")
