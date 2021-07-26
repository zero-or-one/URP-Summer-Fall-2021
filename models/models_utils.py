import numpy as np
import torch
from torch.nn import functional as F

_MODELS = {}

def _add_model(model_fn):
    _MODELS[model_fn.__name__] = model_fn
    return model_fn

@_add_model
def mlp(**kwargs):
    return MLP(**kwargs)

@_add_model
def ntk_linear(**kwargs):
    return NTK_Linear(**kwargs)

@_add_model
def ntk_mlp(**kwargs):
    return NTK_MLP(**kwargs)

@_add_model
def allcnn(**kwargs):
    return AllCNN(**kwargs)

@_add_model
def ntk_allcnn(**kwargs):
    return ntk_AllCNN(**kwargs)

@_add_model
def allcnn_no_bn(**kwargs):
    return AllCNN(batch_norm=False, **kwargs)

@_add_model
def resnet(**kwargs):
    return ResNet18(**kwargs)

@_add_model
def resnet_small(**kwargs):
    return ResNet18_small(**kwargs)

@_add_model
def wide_resnet(**kwargs):
    return Wide_ResNet(**kwargs)

@_add_model
def is_wide_resnet(**kwargs):
    return Wide_ResNetIS(**kwargs)

@_add_model
def ntk_wide_resnet(**kwargs):
    return Wide_ResNetNTK(**kwargs)

def get_model(name, **kwargs):
    return _MODELS[name](**kwargs)