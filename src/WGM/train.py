from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from glob import glob
import pandas as pd
from PIL import Image
import torchdata.datapipes as dp
import random
from torch.utils.data.backward_compatibility import worker_init_fn
from sklearn.metrics import classification_report
import wandb
import plotly.graph_objects as go
from pytorch_grad_cam import GradCAM

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
import plotly.graph_objects as go
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from datapipe import WGM_dataset
cudnn.benchmark = True
plt.ion()   # interactive mode

def plot_roc_auc_curve(run, actual_labels, y_scores, classes, wsi_name,eval_path):
    y_onehot = pd.get_dummies(actual_labels, columns=classes)
    y_scores = pd.DataFrame(y_scores)
    for i, c in enumerate(classes):
        y_scores[c]=y_scores["scores"].apply(lambda l: l[i])
    y_scores.drop(["scores"], axis=1,inplace=True)
    
    table = wandb.Table(columns = [wsi_name])
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    for i in range(len(classes)):
        y_true = y_onehot.iloc[:, i]
        y_score = y_scores.iloc[:, i]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)

        name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=500
    )
    fig.write_html(os.path.join(eval_path, "ROC_Curve" + wsi_name +".html"), auto_play = False)
    table.add_data(wandb.Html(os.path.join(eval_path, "ROC_Curve" + wsi_name +".html")))
    run.log({"ROC Curve": table})



def plot_pr_curve(run, actual_labels, y_scores, classes, wsi_name,eval_path):
    table = wandb.Table(columns = [wsi_name])
    y_onehot = pd.get_dummies(actual_labels, columns=classes)
    y_scores = pd.DataFrame(y_scores)
    for i, c in enumerate(classes):
        y_scores[c]=y_scores["scores"].apply(lambda l: l[i])
    y_scores.drop(["scores"], axis=1,inplace=True)
    
    fig = go.Figure()
    for i in range(len(classes)):
        y_true = y_onehot.iloc[:, i]
        y_score = y_scores.iloc[:, i]
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        fig.add_trace(go.Scatter(x=recall, y=precision, name=classes[i], mode='lines'))

    fig.update_layout(
        xaxis_title='Recall',
        yaxis_title='Precision',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=500
    )

    fig.write_html(os.path.join(eval_path, "PR_Curve" + wsi_name +".html"), auto_play = False)
    table.add_data(wandb.Html(os.path.join(eval_path, "PR_Curve" + wsi_name +".html")))
    run.log({"PR Curve": table})



def plot_training_curve(log_metrics,eval_path):
    # Create traces
    log_metrics = pd.DataFrame(log_metrics)
    for c in ["loss", "metrics"]:
        fig = go.Figure()
        if c=="loss":
            fig.add_trace(go.Scatter(x=log_metrics[log_metrics["phase"]=="train"]["epoch"], y=log_metrics[log_metrics["phase"]=="train"][c],
                                mode='lines+markers',
                                name='Training loss'))
            #fig.add_trace(go.Scatter(x=log_metrics[log_metrics["phase"]=="val"]["epoch"], y=log_metrics[log_metrics["phase"]=="val"][c],
            #                    mode='lines+markers',
            #                    name='Validation loss'))

            fig.update_layout(title='Training/Validation loss vs Epoch',
                            xaxis_title='Epoch No',
                            yaxis_title='Loss')

            fig.write_html(eval_path + "Training_loss_curve.html")
            table = wandb.Html(eval_path + "Training_loss_curve.html")
            wandb.log({"Training Loss":table})
        else:
            fig.add_trace(go.Scatter(x=log_metrics[log_metrics["phase"]=="train"]["epoch"], y=log_metrics[log_metrics["phase"]=="train"][c],
                                mode='lines+markers',
                                name='Training Accuracy'))
            #fig.add_trace(go.Scatter(x=log_metrics[log_metrics["phase"]=="val"]["epoch"], y=log_metrics[log_metrics["phase"]=="val"][c],
            #                    mode='lines+markers',
            #                    name='Validation Accuracy'))           

            fig.update_layout(title='Training/Validation Acccuracy vs Epoch',
                            xaxis_title='Epoch No',
                            yaxis_title='Accuracy')

            fig.write_html(eval_path + "Training_Accuracy_curve.html")
            table = wandb.Html(eval_path + "Training_Accuracy_curve.html")
            wandb.log({"Training Accuracy":table})


def val_model(model,criterion):
    model.eval()
    running_corrects = 0
    running_loss = 0.0
    with torch.no_grad():
        for i, (wsi, inputs, labels) in enumerate(dataloaders['val']):
            #print("-------------------",i,"--------------------")
            inputs = inputs.squeeze(0)
            labels = labels.squeeze(0)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
    epoch_acc = running_corrects/dataset_sizes['val']
    epoch_loss = running_loss/dataset_sizes['val']
    return epoch_acc, epoch_loss
            


def train_model(run, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    log_metrics = list()
    phase = 'train'
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        # Each epoch has a training and validation phase
        phase = 'train'
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        for i, (inputs, labels) in enumerate(dataloaders[phase]):
            inputs=inputs.squeeze(0)
            labels = labels.squeeze(0)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            #print(f'Epoch {epoch} {phase} Loss: {loss.item():.4f} Batch No: {i}') 
        scheduler.step()
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        #try:
        val_acc, val_loss = val_model(model,criterion)
        print(f'Epoch {epoch} {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Val_Acc: {val_acc:.4f} Val_Loss: {val_loss:.4f}')
        log_metrics.append(dict(epoch=epoch, loss=epoch_loss, metrics=epoch_acc,val_loss=val_loss, val_metrics=val_acc))
        if epoch==5:
            torch.save({"model":model, "state": model.state_dict()}, '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/WM_models/'+run_id+'5.pth')
        if epoch==10:
            torch.save({"model":model, "state": model.state_dict()}, '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/WM_models/'+run_id+'10.pth')
        if epoch==15:
            torch.save({"model":model, "state": model.state_dict()}, '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/WM_models/'+run_id+'15.pth')
        
        #except Exception as e: 
        #    print(e)
        #    print(f'Epoch {epoch} {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        #    log_metrics.append(dict(epoch=epoch, loss=epoch_loss, metrics=epoch_acc))
            
        #print(f'Epoch {epoch} {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        #log_metrics.append(dict(epoch=epoch, loss=epoch_loss, metrics=epoch_acc,val_loss=val_loss, val_metrics=val_acc))
        #log_metrics.append(dict(epoch=epoch, loss=epoch_loss, metrics=epoch_acc))
        
    run.log({"log":log_metrics})
    artifact = wandb.Artifact(artifact_name, type='files')
    with artifact.new_file(f'ckpt/{epoch}.pt', 'wb') as f:
        torch.save(model.state_dict(), f)
    torch.save({"model":model, "state": model.state_dict()}, '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/WM_models/'+artifact_name+'.pth')
    run.log_artifact(artifact)
    #run.finish()
    return model, log_metrics

def load_saved_model(path):
    checkpoint = torch.load(path)
    model_ft = checkpoint["model"]
    model_ft.load_state_dict(checkpoint['state'])
    return model_ft

def test_model(model, run, eval_path):
    was_training = model.training
    model.eval()
    actual_labels = []
    pred_labels = []
    wsi_names = []
    scores=[]
    with torch.no_grad():
        for i, (wsi, inputs, labels) in enumerate(dataloaders['val']):
            #print("-------------------",i,"--------------------")
            inputs=inputs.squeeze()
            labels = labels.squeeze()
            inputs = inputs.to(device)
            labels = labels.to(device)
            actual_labels.extend(labels.tolist())
            outputs = model(inputs)
            scores.extend(outputs.cpu().tolist())
            _, preds = torch.max(outputs, 1)
            #print(preds)
            pred_labels.extend(preds.tolist())
            #print(wsi)
            wsi=[x[0] for x in wsi]
            wsi_names.extend(wsi)
            if i%500==0:
                print(i, "Done")
            #if (i!=0) and (i%1000==0):
            #    break
        
    output_df = pd.DataFrame({"wsi_name":wsi_names,"actual_labels":actual_labels,"pred_labels":pred_labels,"scores":scores})

    classes = [0,1,2]
    try:
        plot_roc_auc_curve(run, actual_labels, scores, classes,eval_path)
    except:
        pass

    eval_metrics = pd.DataFrame(columns=["WSI","accu_score","precision-white","recall-white","f1-score-white","support-white", "precision-grey","recall-grey","f1-score-grey","support-grey", "precision-bg","recall-bg","f1-score-bg","support-bg"])
    ind =0 
    for wsi in output_df['wsi_name'].unique():
        tmp = output_df[output_df["wsi_name"]==wsi]  
        acc_score = accuracy_score(tmp["actual_labels"], tmp["pred_labels"])
        prec_rec = precision_recall_fscore_support(tmp["actual_labels"], tmp["pred_labels"], labels=[0,1,2])
        #avg_acc.append(acc_score)
        #print("-----Evaluation Metric for WSI-------",wsi )
        #print(acc_score)
        #print(prec_rec)
        eval_metrics.loc[ind] = (wsi, acc_score, prec_rec[0][0],prec_rec[1][0],prec_rec[2][0],prec_rec[3][0], prec_rec[0][1],prec_rec[1][1],prec_rec[2][1],prec_rec[3][1], prec_rec[0][2],prec_rec[1][2],prec_rec[2][2],prec_rec[3][2] )
        ind=ind+1
        wandb.summary["Evaluation Metric for WSI :" + wsi]=acc_score
    tbl = wandb.Table(data=eval_metrics)
    run.log({"Test Evaluation Metric": tbl})
    eval_metrics.to_csv(eval_path + "eval_metric.csv")
    print(eval_metrics)
    conf_matrix = confusion_matrix(output_df["actual_labels"],output_df["pred_labels"])
    print(conf_matrix)
    try:
        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=output_df["actual_labels"].values, preds=output_df["pred_labels"].values,
                        class_names=["White","gray","background"])})
    except:
        pass

    return output_df

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(1024),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transforms.FiveCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.8965, 0.8875, 0.9023],[0.0807, 0.0911, 0.0849])
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(1024),
        #transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.8965, 0.8875, 0.9023],[0.0807, 0.0911, 0.0849])
       # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.8965, 0.8875, 0.9023],[0.0807, 0.0911, 0.0849])
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

if __name__=="main":
    TRAIN_CSV_PATH = "/home/mahirwar/Desktop/Monika/npsad_data/monika/LBD/Intermediate_data/train_redmarked.csv"
    VAL_CSV_PATH = "/home/mahirwar/Desktop/Monika/npsad_data/monika/LBD/Intermediate_data/val_redmarked.csv"
    batch_size = 8
    
    train_wgm = WGM_dataset(TRAIN_CSV_PATH, data_transforms, "train", batch_size)
    val_wgm = WGM_dataset(VAL_CSV_PATH, data_transforms, "val", batch_size)
    
    train_loader = train_wgm.build_data_pipe()
    val_loader = val_wgm.build_data_pipe()
    dataloaders = {"train":train_loader, "val": val_loader}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_datasize = train_wgm.dataset_size()
    val_datasize = val_wgm.dataset_size()
    dataset_sizes = {'train':train_datasize, 'val':val_datasize}
    
    train_config = dict(
    epochs = 20,
    batch_size = batch_size,
    num_classes = 3,
    device_id = 0,
    eval_freq = 1,
)
    test_config = dict(
    batch_size = 32
)

    model_config = dict(lr=0.001, momentum=0.9)

    optim_config = dict(step_size=7, gamma=0.1)


    wandb_config = dict(
    project='LBD',
    entity='monika-ahirwar',
    config=dict(
        train_config=train_config,
        model_config=model_config,
        optim_config=optim_config,
    ),
    save_code=False,
    group='runs',
    job_type='train',
)
    run = wandb.init(**wandb_config)
    assert run is wandb.run # run was successfully initialized, is not None
    run_id, run_dir = run.id, run.dir
    exp_name = run.name
    artifact_name = f'{run_id}-logs'
    
    
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, train_config["num_classes"])

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    #optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.00001, weight_decay=1e-5)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    model_ft, log_metrics = train_model(run, model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=train_config["epochs"])
    
    eval_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/WM_Evaluation_Metrics/"+run_id

    if not os.path.exists(eval_path):
        os.makedirs(eval_path)
    
    plot_training_curve(log_metrics,eval_path)
    output_df = test_model(model_ft, run, eval_path)
    
    run.finish()