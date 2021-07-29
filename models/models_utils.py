import numpy as np
import torch
from torch.nn import functional as F
from models import *


# no need to automatically add models for now

def get_model(name, **kwargs):
    if name == "linear_regression":
        model = LinearRegression(input_size=kwargs.get("input_size"), output_size=kwargs.get("output_size"))
    elif name == "polynomial_regression":
        model = PolynomialRegression(degree=kwargs.get("degree"))
    elif name == "logistic_regression":
        model = LogisticRegression(input_size=kwargs.get("input_size"))
    elif name == "dnn":
        model = DNN(input_size=kwargs.get("input_size"), num_classes=kwargs.get("num_classes"),
                    hidden_size=kwargs.get("hidden_size"), num_layer=kwargs.get("num_layer"), activation=kwargs.get("activation"))
    elif name == "cnn":
        model = CNN(filters_percentage=kwargs.get("filters_percentage"), n_channels=kwargs.get("n_channels"),
                    num_classes=kwargs.get("num_classes"), dropout=kwargs.get("dropout"), batch_norm=kwargs.get("batch_norm"))
    elif name == "resnet18":
        model = ResNet18(filters_percentage=kwargs.get("filters_percentage"), n_channels=kwargs.get("n_channels"),
                    num_classes=kwargs.get("num_classes"), block=kwargs.get("block"), num_blocks=kwargs.get("num_blocks"))
    return model

def get_param(name):
    if name == "linear_regression":
        args = {}
    elif name == "polynomial_regression":
        args = {}
    elif name == "logistic_regression":
        args = {}
    elif name == "dnn":
        args = {}
    elif name == "cnn":
        args = {}
    elif name == "resnet":
        args = {}
    return args

'''
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
