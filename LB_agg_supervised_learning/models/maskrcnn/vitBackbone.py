import timm
#from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torch import nn
import torch
from torchvision.ops.feature_pyramid_network import (
    LastLevelMaxPool,
    FeaturePyramidNetwork
)
import collections
import torch.nn.functional as F


HIDDEN_DIM =768
IMG_SIZE =1024
PATCH_SIZE=16
NUM_LAYER=12
device = torch.device('cuda', 0)


def _process_input(x: torch.Tensor) -> torch.Tensor:
    n, c, h, w = x.shape
    p = PATCH_SIZE
    torch._assert(h == IMG_SIZE, f"Wrong image height! Expected {IMG_SIZE} but got {h}!")
    torch._assert(w == IMG_SIZE, f"Wrong image width! Expected {IMG_SIZE} but got {w}!")
    n_h = h // p
    n_w = w // p

    # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
    #x = self.conv_proj(x)
    conv_proj =  nn.Conv2d(
                in_channels=3, out_channels=HIDDEN_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE, device=device
            )
    x = conv_proj(x)

    # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
    x = x.reshape(n, HIDDEN_DIM, n_h * n_w)

    # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
    # The self attention layer expects inputs in the format (N, S, E)
    # where S is the source sequence length, N is the batch size, E is the
    # embedding dimension
    x = x.permute(0, 2, 1)

    return x


class IntermediateLayerGetter(nn.ModuleDict):
    _version = 2
    __annotations__ = {
        "return_layers",
    }

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset(
            [name for name, _ in model.named_children()]
        ):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = collections.OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

        self.C = HIDDEN_DIM
        self.H = self.W = IMG_SIZE // PATCH_SIZE

    def forward(self, x):
        out = collections.OrderedDict()
        idx = 0
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                #print( out_name)
                N = x.shape[0]
                out[out_name] = F.interpolate(
                    F.instance_norm(
                        x.permute(0, 2, 1).reshape(N, self.C, self.H, self.W)
                    ),
                    scale_factor=4 / (2**idx),
                    mode="bilinear",
                )
                idx += 1
        return out
    
    
class BackboneWithFPN(nn.Module):
    def __init__(
        self,
        backbone,
        return_layers,
        in_channels_list,
        out_channels,
        extra_blocks=None,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.backbone = backbone

        self.body = IntermediateLayerGetter(
            self.backbone.blocks,
            return_layers=return_layers,
        )
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        ).to(device)
        self.out_channels = out_channels

    def forward(self, x):
        x = _process_input(x)
        x = x + self.backbone.pos_embed
        #x = self.backbone.encoder.dropout(x)
        x = self.body(x)
        x = self.fpn(x)
        return x
    
    
    
class ViTFPNWrapper(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', image_size=1024, out_channels=256):
        super().__init__()
        self.vit = timm.create_model(
            model_name, 
            pretrained=True, 
            img_size=image_size,
        )
        self.vit.pos_embed = torch.nn.Parameter(self.vit.pos_embed[:, 1:, :])
          # Removes FC layer by setting num_classes=0
        # Get channels from ViT (usually all the same)
        del self.vit.cls_token, self.vit.norm
        self.vit.to(device)
        self.out_channels = out_channels
        #self.feature_info = self.vit.feature_info
        self.fpn1 = BackboneWithFPN(
            self.vit,
            #return_layers={str(i): str(i) for i in range(4)},
            return_layers={str((NUM_LAYER - 1) - l):str(((NUM_LAYER - 1) - l - 2) // 3) for l in range(9, -1, -3)},
            #in_channels_list=[f["num_chs"] for f in self.feature_info],
            in_channels_list=[768] * 4,
            out_channels=self.out_channels,
        )

    def forward(self, x):
        return self.fpn1(x)



# Instantiate backbone
#backbone = ViTFPNWrapper('vit_base_patch16_224', image_size=1024)


#x = torch.randn(1, 3, 1024, 1024).to(device)
#out = backbone(x)

#print(out)