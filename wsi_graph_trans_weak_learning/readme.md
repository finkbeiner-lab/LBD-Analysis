# Using Cancer study referenced below as basis

https://ieeexplore.ieee.org/document/9779215

<img width="405" height="152" alt="image" src="https://github.com/user-attachments/assets/db097f02-2bfc-46be-a390-4eec0d2aa668" />

<img width="263" height="162" alt="image" src="https://github.com/user-attachments/assets/b99b38c4-9310-478f-8738-64ac42483023" />



<img width="249" height="145" alt="image" src="https://github.com/user-attachments/assets/2f769718-260e-4b66-ab1c-3a9f0bc9b767" />


## Finetune our whole slide images of PDD and DLB brains

We pretrained a simCLR model on crops and use it to extract features for each crop of whole slide image, which is further represented as graph.

We pass these graphed image as input to graph transformer model.

The model then predicts if the whole slide image is of PDD or DLB and generates a highly activated region for the output

<img width="543" height="163" alt="image" src="https://github.com/user-attachments/assets/f87c81b0-56a7-4a39-b476-aa9a6ddca7c2" />

## Model performance
<img width="370" height="83" alt="image" src="https://github.com/user-attachments/assets/a1b9724f-e4bd-4bc2-8a9e-558cb20a460a" />

## Graph Cam
We generated heatmaps for correct predictions of PDD/DLB slides across different brain regions and across different antibodies 

wsi_graph_trans_weak_learning/graphcam_vis/visualization.png
