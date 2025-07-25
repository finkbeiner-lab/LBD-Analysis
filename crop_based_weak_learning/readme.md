# Training a crop based classifier

## Pretrained crops from alpha syn stained slides of PDD and DLB brains 

The pretraining is done using SimCLR model.

<p align="center">
    <img src="models/pretraining_model.png" width="50%"> <br>

  *Pretraining*

</p>


## Fine-tuned crops with weak labels as PDD or DLB 

Fine-tune the crops using pretrained model as backbone to classify crop as PDD or DLB 

<p align="center">
    <img src="models/finetuning.png" width="50%"> <br>

  *Finetuning*

</p>

<p align="center">
    <img src="models/model_performance.png" width="50%"> <br>

  *Model Performance*

</p>


## Apply gradcam to see features responsible for model's prediction


<p align="center">
    <img src="explainability/dlb_explain.png" width="50%"> <br>

  *DLB crop*

</p>

<p align="center">
    <img src="explainability/pdd_explain.png" width="50%"> <br>

  *PDD crop*

</p>
