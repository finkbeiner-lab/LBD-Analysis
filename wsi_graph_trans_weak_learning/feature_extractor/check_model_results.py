from sklearn.manifold import TSNE
import seaborn as sns
import torch
from models.resnet_simclr import ResNetSimCLR
from PIL import Image

model_path  = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/tmi2022/feature_extractor/runs/May23_19-38-08_kif-gh200-02.gladstone.internal/checkpoints/model.pth"

model = ResNetSimCLR()# .to(self.device)
#model = _load_pre_trained_weights(model)
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)

model = model.to(device)



test_img = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/data/antibodies_data/tiles/output/2008-134-C34-45-Amygdala.svs/02042x_25512y.png"






embeddings = model.encode(X_test)  # get embeddings from encoder
tsne = TSNE(n_components=2)
emb_2d = tsne.fit_transform(embeddings)

sns.scatterplot(x=emb_2d[:,0], y=emb_2d[:,1], hue=labels)
