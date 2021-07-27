import numpy as np
import torch
from torch.nn import functional as F
from models import *

# add models
_MODELS = {}

def _add_model(model_fn):
    _MODELS[model_fn.__name__] = model_fn
    return model_fn

@_add_model
def lr(**kwargs):
    return LinearRegression(**kwargs)

@_add_model
def pl(**kwargs):
    return PolynomialRegression(**kwargs)

@_add_model
def logr(**kwargs):
    return LogisticRegression(**kwargs)

@_add_model
def dnn(**kwargs):
    return DNN(**kwargs)

@_add_model
def cnn(**kwargs):
    return CNN(**kwargs)

@_add_model
def resnet(**kwargs):
    return ResNet18(**kwargs)

# get the models
def get_model(name, **kwargs):
    return _MODELS[name](**kwargs)

'''
@_add_model
def resnet_small(**kwargs):
    return ResNet18_small(**kwargs)

@_add_model
def allcnn_no_bn(**kwargs):
    return AllCNN(batch_norm=False, **kwargs)

@_add_model
def wide_resnet(**kwargs):
    return Wide_ResNet(**kwargs)

@_add_model
def is_wide_resnet(**kwargs):
    return Wide_ResNetIS(**kwargs)

@_add_model
def ntk_wide_resnet(**kwargs):
    return Wide_ResNetNTK(**kwargs)
'''
