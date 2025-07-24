import os
import sys
sys.path.append(os.path.join(os.getcwd(), *tuple(['..'])))
import argparse

from typing import Callable, Dict, List, Optional, Set
from collections import OrderedDict
import pdb
import torch
from torch import nn, Tensor
import torch.optim
import wandb
from model_mrcnn import _default_mrcnn_config, build_default
from features import build_features
#from features import transforms as T
from utils.engine import evaluate
import torchvision
import matplotlib.pyplot as plt
from visualization.explain import ExplainPredictions
import pandas as pd
import plotly.graph_objects as go
import pdb
from sklearn.metrics import precision_recall_curve, auc
import numpy as np
from torchvision import transforms



# Sets the behavior of calls such as
#   - model.to(device=torch.device(type='cpu', index=None))
#   - model.cpu()
#   - model.cuda(), model.cuda(0)
#   - model.apply(fn)
# where isinstance(model, torch.nn.Module);
# True implies model.param = model.param.data.to(device),
# False implies model.param.data = model.param.data.to(device).
#
# Selecting True emulates upcoming change in semantics of torch.nn.Parameter;
# if torch.nn.Parameter inherits directly from torch.nn.Tensor,
# then as per torch docs for these methods, their invocation yields a new Tensor (equivalently Parameter) object.
# Alternatively, the recursive calls at present function by setting the
# torch.nn.Parameter.data field for a particular parameter object,
# as well as its torch.nn.Parameter.data.grad field if applicable.
# The Parameter class here is simply a wrapper for its data Tensor,
# and any previous reference to the Parameter instance will yield its
# now-altered data Tensor. However, this may not always be possible;
# if the semantics of Parameter change, an in-place modification of the
# underlying Tensor object itself would be required (i.e. a _to()), but this is not universally supported.


torch.__future__.set_overwrite_module_params_on_conversion(True)

def visualize_augmentations(images, targets):

    plt.figure(figsize=(10,10)) # specifying the overall grid size
    plt.suptitle('Data Augmentations')
    plt.subplot(1,2, 1)
    

    for i in range(len(images)):
        display_list = []
        img = images[i].detach().cpu().numpy()
        img = img.transpose(1, 2, 0)
        display_list.append(img)
        mask = targets[i]['masks'].detach().cpu().numpy()
        mask = mask.transpose(1, 2, 0)
        display_list.append(mask)

        for j in range(2):
            plt.subplot(1,2,j+1)
            plt.imshow(display_list[j])
        
        save_name = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/reports/view_images/augmentation_{img_no}.png"

        plt.savefig(save_name.format(img_no=i))

def plotPRcurve(eval2, epoch, run, patient_id):

    df = pd.DataFrame(columns=["class","tpr","fpr","recall","precision"])

    #colors={"True":'royalblue', "Pre":'firebrick','False':"green"}
    #classes = ['True', 'Pre']
    colors={"True":'royalblue'}
    classes = ['True']
    

    len_classes = len(eval2['bbox'])

    random_rp_dict=dict()
    for c in range(len_classes): # running for all 3 classes
        ## parameters
        area_index = 0 # area - all (areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]  -> areaRngLbl = ['all', 'small', 'medium', 'large'])
        threshold_index= 0 # threshold 0.5  (iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)) -- can vary threshold from here 
        iou_type="bbox"
        maxDet = 100 
        # Selecting from 
        eval_table = eval2[iou_type][c][area_index]

        dt_score_list = np.concatenate([eval_table[i]['dtScores'][0:maxDet] for i in range(len(eval_table)) if eval_table[i]!=None])
        inds = np.argsort(-dt_score_list, kind='mergesort') 
        dtScoresSorted = dt_score_list[inds]
        dtm  = np.concatenate([eval_table[i]['dtMatches'][threshold_index][0:maxDet]  for i in range(len(eval_table)) if eval_table[i]!=None]) [inds]
        dtIg  = np.concatenate([eval_table[i]['dtIgnore'][threshold_index][0:maxDet]  for i in range(len(eval_table)) if eval_table[i]!=None]) [inds]
        gtIg = np.concatenate([eval_table[i]['gtIgnore'] for i in range(len(eval_table)) if eval_table[i]!=None])
        npig = np.count_nonzero(gtIg==0)
        tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
        fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )
        tp_sum = np.cumsum(tps, axis=0, dtype=float)
        

        if len(tp_sum) == 0:
            continue
        
        # rp =tp_sum[-1]/len(tp_sum)
        # random_rp_dict[c]=rp
        # tp_sum=tp_sum/tp_sum[-1]
        fp_sum = np.cumsum(fps, axis=0, dtype=float)
        # fp_sum=fp_sum/fp_sum[-1]
        rc_list =[]
        pr_list =[]
        for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
            rc = tp / npig
            pr = tp / (fp+tp+np.spacing(1))
            rc_list.append(rc)
            pr_list.append(pr)
        tmp = pd.DataFrame({"tpr":tp_sum,"fpr":fp_sum,"recall":rc_list,"precision":pr_list})
        tmp["class"] = c
        df = pd.concat([df, tmp])
    
    # plotly plot
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    #classes = ['True', 'Pre']
    class_auc_pr = {}

    for i in range(len(classes)):
        fig.add_trace(go.Scatter(x=df[df["class"]==i]["recall"], y=df[df["class"]==i]["precision"], name=classes[i], mode='lines', line=dict(color=colors[classes[i]])))

        #pdb.set_trace()
        precision = df[df["class"]==i]["precision"]
        recall = df[df["class"]==i]["recall"]
        try:
            class_auc_pr[i] = auc(recall, precision)
        except:
            continue
    

        # fig.add_trace(go.Scatter(x=[0,1], y=[random_rp_dict[i],random_rp_dict[i]], name=classes[i] +"_random", mode='lines',line=dict(color=colors[classes[i]], width=1,dash='dash')))

    fig.update_layout(
        plot_bgcolor='white',
        xaxis_title='Recall',
        yaxis_title='Precision',
        width=1000, height=500,
        title='Precision-Recall Curve'
    )
    fig.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black',gridcolor='lightgrey')
    fig.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black',gridcolor='lightgrey')

    
    save_name = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/PR_Curves/prcurve_"+patient_id+"{epoch}.html"
    fig_name = save_name.format(epoch=epoch)
    
    fig.write_html(fig_name)
    run.log({"Precision-Recall for "+patient_id: wandb.Html(open(fig_name))})




def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: Callable[[Dict[str, Tensor]], Tensor],
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int = 1,
    log_freq: int = 10,) -> None:

    assert model.training
    model_params = set(model.parameters())
    model_devices = set([p.device for p in model_params])
    assert model_devices == set([device]) # validate model params device
    for g in optimizer.param_groups: # validate optimizer params
        assert set(g['params']).issubset(model_params)

    log_metrics = list()

    for i, (images, targets) in enumerate(train_data_loader):
        images = [image.to(device) for image in images]
        #print(images[0].shape)
        targets = [dict([(k, v.to(device)) for k, v in target.items()]) for target in targets]
        #print(targets[0]["masks"].shape)
        """ 
        targets1=[]
        for target in targets:
            d = dict()
            for k, v in target.items():
                if k=="masks":
                    print(v.shape)
                    v=v.cpu().numpy()[:1022,:1022]
                    v = torch.as_tensor(v, dtype=torch.uint8)
                    print(v.shape)
                d[k]=v.to(device)
            targets1.append(d)
        print(targets1[0]["masks"].shape)
        targets = targets1
        """
        #visualize_augmentations(images , targets)
        # pdb.set_trace()
        optimizer.zero_grad()
        loss, metrics = loss_fn(model.forward(images, targets))
        loss.backward()
        optimizer.step()

        log_metrics.append(dict(epoch=epoch, loss=loss.item(), metrics=metrics))
        # print(dict(epoch=epoch, loss=loss.item(), metrics=metrics))
        print_logs = "epoch no : {epoch}, batch no : {batch_no}, total loss : {loss},  classifier :{classifier}, mask: {mask} ==================="
        print(print_logs.format(epoch=epoch, batch_no=i, loss=loss.item(),  classifier=metrics['loss_classifier'], mask=metrics['loss_mask']))
        if (i % log_freq) == 0:
            yield log_metrics
            log_metrics = list()

    yield log_metrics


def get_loss_fn(weights, default=0.):
    def compute_loss_fn(losses):
        item = lambda k: (k, losses[k].item())
        metrics = OrderedDict(list(map(item, [k for k in weights.keys() if k in losses.keys()] + [k for k in losses.keys() if k not in weights.keys()])))
        loss = sum(map(lambda k: losses[k] * (weights[k] if weights is not None and k in weights.keys() else default), losses.keys()))
        return loss, metrics
    return compute_loss_fn


def get_resp(prompt, prompt_fn=None, resps='n y'.split()):
    resp = input(prompt)
    while resp not in resps:
        resp = input(prompt if prompt_fn is None else propt_fn(resp))
    return resps.index(resp)


if __name__ == '__main__':
    # TODO:
    #   - add functionality for calling backward with create_graph, i.e. for higher-order derivatives
    #   - switch to support for standard torchvision-bundled transforms (i.e. instead of `features.transforms as T` try `torchvision.transforms.transforms` or `torchvision.transforms.functional`)
    #   - complete feature: add grad_optimizer support transparently (so that usage is the same for users and train_one_epoch interface whether torch.optim or grad_optim is selected, i.e. log grads automatically)
    #   - do ^^ via closures
    #   - experimental: add an API to collect params and bufs by on module and/or name; generate on-the-fly state_dicts, gradient_dicts, higher-order gradient_dicts, etc.

    parser = argparse.ArgumentParser(description='Maskrcnn training')

    parser.add_argument('base_dir', help="Enter the base dir (NAS)")
    parser.add_argument('dataset_train_location',
                        help='Enter the path train dataset resides')
    parser.add_argument('dataset_test_location',
                        help='Enter the path where test dataset resides')
    
    args = parser.parse_args()

    ## CONFIGS ##
    collate_fn = lambda _: tuple(zip(*_)) # one-liner, no need to import

    dataset_base_dir = args.base_dir
    dataset_train_location = args.dataset_train_location
    dataset_test_location = args.dataset_test_location

    train_config = dict(
        epochs = 75,
        batch_size = 6,
        num_classes = 3,
        device_id = 0,
        ckpt_freq =500,
        eval_freq = 25,
    )

    test_config = dict(
        batch_size = 1
    )

    model_config = _default_mrcnn_config(num_classes=1 + train_config['num_classes']).config
    optim_config = dict(
        #cls=grad_optim.GradSGD,
        #cls=torch.optim.SGD,
        cls=torch.optim.Adam,
        defaults=dict(lr=0.0001,weight_decay=1e-5)  #-4 is too slow,
        #weight_decay=1e-5
    )
   
    
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

    data_transforms = transforms.Compose([
        #transforms.Resize((1022,1022)),
        transforms.ToTensor(),
        
        #transforms.Normalize([0.8753, 0.8724, 0.8949],[0.0439, 0.0443, 0.0403])
    ])


    ## Dataset loading
    #train_dataset = build_features.LBD_Dataset(dataset_train_location, data_transforms,["augmented_images","augmented_labels"])
    #train_dataset = build_features.LBD_Dataset(dataset_train_location, data_transforms,["augmented_images1","augmented_labels1"])
    train_dataset = build_features.LBD_Dataset(dataset_train_location, data_transforms,["images","labels"])
    #train_dataset = build_features.LBD_Dataset(dataset_train_location, T.Compose([T.ToTensor(),T.Normalize([0.8753, 0.8724, 0.8949],[0.0439, 0.0443, 0.0403])]),["images","labels"])
    #train_dataset = build_features.LBD_Dataset(dataset_train_location, data_transforms,["images","labels"])
    #test_dataset = build_features.LBD_Dataset(dataset_test_location, T.Compose([T.ToTensor()]))

    train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=4,
            collate_fn=collate_fn)
    
    #test_data_loader = torch.utils.data.DataLoader(
    #        test_dataset, batch_size=test_config['batch_size'], shuffle=False, num_workers=4,
    #        collate_fn=collate_fn)

    
    # Model Building
    model = build_default(model_config, im_size=1024)
    #model_name = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/models/mrcnn_models/dazzling-plant-239_mrcnn_model_50.pth"
    #model.load_state_dict(torch.load(model_name))
    device = torch.device('cpu')
    if torch.cuda.is_available():
        assert train_config['device_id'] >= 0 and train_config['device_id'] < torch.cuda.device_count()
        device = torch.device('cuda', train_config['device_id'])
    model = model.to(device)
    model.train(True)

    loss_names = 'objectness rpn_box_reg classifier box_reg mask'.split()
    loss_weights = [1., 4., 1., 4., 1.,]
    loss_weights = OrderedDict([(f'loss_{name}', weight) for name, weight in zip(loss_names, loss_weights)])

    loss_fn = get_loss_fn(loss_weights)

    optimizer = optim_config['cls']([dict(params=list(model.parameters()))], **optim_config['defaults'])

    run = wandb.init(**wandb_config)
    assert run is wandb.run # run was successfully initialized, is not None
    run_id, run_dir = run.id, run.dir
    exp_name = run.name

    artifact_name = f'{run_id}-logs'
    
    trained_model_names = []
    
    # Train Data
    for epoch in range(train_config['epochs']):
        # print(f'Epoch {epoch}=======================================>.')

        for logs in train_one_epoch(model, loss_fn, optimizer, train_data_loader, device, epoch=epoch, log_freq=1):
            for log in logs: 
                print(log)
                run.log(log)

        #if epoch + 1 == train_config['epochs'] or epoch % train_config['ckpt_freq'] == 0:

        #    artifact = wandb.Artifact(artifact_name, type='files')
        #    with artifact.new_file(f'ckpt/{epoch}.pt', 'wb') as f:
        #        torch.save(model.state_dict(), f)
        #    run.log_artifact(artifact)

        #if epoch % train_config['eval_freq'] == 0:
        #    eval_res = evaluate(run, model, test_data_loader, device=device)
        
        model.train(True)
        
        if ((epoch+1)%25==0) & (epoch>0):
            model_save_name = dataset_base_dir + "models/mrcnn_models/{name}_mrcnn_model_{epoch}.pth"
            torch.save(model.state_dict(), model_save_name.format(name=exp_name, epoch=epoch))
            trained_model_names.append(model_save_name.format(name=exp_name, epoch=epoch))
            print("******************saving model*********************",model_save_name)

    
    model_save_name = dataset_base_dir + "models/mrcnn_models/{name}_mrcnn_model_{epoch}.pth"
    torch.save(model.state_dict(), model_save_name.format(name=exp_name, epoch=train_config['epochs']))
    
    
    # print("\n =================The Model is Trained!====================")
    # print("-----------------Visualizing Model predictions----------------")

    # # TODO Testing is done on Individual WSI Folders
    # input_path = '/mnt/new-nas/work/data/npsad_data/vivek/Datasets/amyb_wsi/test'

    # model = build_default(model_config, im_size=1024)
   
    # explain = ExplainPredictions(model, model_input_path = model_save_name.format(name=exp_name, epoch=train_config['epochs']), test_input_path=input_path, 
    #                             detection_threshold=0.75, wandb=run, save_result=True, ablation_cam=True, save_thresholds=False)
    # explain.generate_results()
    

    print("*****************************************************************************************************************************************************")
    print("                       Testing the model                                       ")
    epoch = 0

    ## CONFIGS ##
    collate_fn = lambda _: tuple(zip(*_)) # one-liner, no need to import

    test_config = dict(
        epochs = 1,
        batch_size = 8,
        num_classes = 3,
        device_id = 0,
        ckpt_freq =500,
        eval_freq = 30,
    )

    device = torch.device('cuda', test_config['device_id'])
    
    
    '''
    isKfold_eval = True
    test_ds = build_features.LBD_Dataset(dataset_test_location, T.Compose([T.ToTensor()]),["images","labels"])
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=test_config['batch_size'], shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_eval_res, test_eval_res_df, full_table = evaluate(run, model, test_loader, device, isKfold_eval)
    plotPRcurve(full_table, epoch, run,"all")
    tbl = wandb.Table(data=test_eval_res_df)
    run.log({"Average Evaluation Metric": tbl})
    '''
    
    
    test_patient_ids = os.listdir(dataset_test_location)
    
    if '.DS_Store' in test_patient_ids:
        test_patient_ids.remove('.DS_Store')
       
    isKfold_eval = True
    #trained_model_names = ["exalted-planet-240_mrcnn_model_24.pth", "exalted-planet-240_mrcnn_model_49.pth", "exalted-planet-240_mrcnn_model_50.pth"]
    #trained_model_names = ["beaming-flower-261_mrcnn_model_24.pth", "beaming-flower-261_mrcnn_model_50.pth"]
    #trained_model_names = ["vivid-disco-265_mrcnn_model_24.pth","vivid-disco-265_mrcnn_model_49.pth","vivid-disco-265_mrcnn_model_74.pth","vivid-disco-265_mrcnn_model_99.pth" ]
    #trained_model_names = ["toasty-dragon-301_mrcnn_model_24.pth","toasty-dragon-301_mrcnn_model_50.pth"]
    for model_name in trained_model_names:
        model_name = os.path.join("/gladstone/finkbeiner/steve//work/data/npsad_data/monika/LBD/models/mrcnn_models", model_name)
        model_config = _default_mrcnn_config(num_classes=1 + test_config['num_classes']).config
        model = build_default(model_config, im_size=1024)
        model.load_state_dict(torch.load(model_name))
        model = model.to(device) 
        eval_metric_full_training = pd.DataFrame()
        print("******Running for model:,", model_name)
        
        for t in range(len(test_patient_ids)):
            if len(os.listdir(os.path.join(dataset_test_location,test_patient_ids[t],"images")))==0:
                continue
            test_ds = build_features.LBD_Dataset(os.path.join(dataset_test_location,test_patient_ids[t]), data_transforms,["images","labels"])
            test_loader = torch.utils.data.DataLoader(test_ds, batch_size=test_config['batch_size'], shuffle=False, num_workers=4, collate_fn=collate_fn)
            test_eval_res, test_eval_res_df, full_table = evaluate(run, model, test_loader, device, isKfold_eval)
            test_eval_res_df["patient_id"] = test_patient_ids[t]
            eval_metric_full_training = pd.concat([eval_metric_full_training,test_eval_res_df])
            plotPRcurve(full_table, epoch, run,test_patient_ids[t])
        eval_metric_full_training.to_csv("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/Reports/"+model_name.split("/")[-1]+"_validation_result.csv")
        avg_eval_metric =  eval_metric_full_training.groupby(["iou_type","metric_name"])["metric_value"].mean().reset_index()
        #print(eval_metric_full_training)
        tbl = wandb.Table(data=eval_metric_full_training)
        run.log({"Val Evaluation Metric Patient-wise " + model_name.split("/")[-1]: tbl})
            
        tbl = wandb.Table(data=avg_eval_metric)
        run.log({"Average Evaluation Metric "+ model_name.split("/")[-1]: tbl})

run.finish()

