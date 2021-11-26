import os
import sys
sys.path.append("../URP")
path = ".."
from learning.learn_utils import *
from learning.learn import *
from utils import *
import numpy as np
import time
import torch
sys.path.append(path + "/data")
from data_utils import *


class Encoder(nn.Module):
    '''Encode the data to carry destructive information about its features'''
    def __init__(self, input_size=512, hidden_size=16, num_layer=2, activation=nn.ReLU()):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.activation = activation
        self.layers = self._make_layers()
        #self.model = model

    def _make_layers(self):
        layer = []
        layer += [
        #Layer(self.input_size, self.hidden_size),
        nn.Linear(self.input_size, self.hidden_size),
        self.activation]
        #layer += [self.activation]
        for i in range(self.num_layer - 2):
            layer += [nn.Linear(self.hidden_size, self.hidden_size)]
            layer += [self.activation]
        layer += [nn.Linear(self.hidden_size, self.input_size)]
        return nn.Sequential(*layer)

    def forward(self, x, dataset):
        if dataset == 'mnist':
          #print(x.shape)
          x = x.float()
          l = x.size(1)
          x = x.reshape(x.size(0), -1, self.input_size)
          x = self.layers(x)
          #x = self.model(x)
          x = x.reshape(x.size(0), -1)
          return x          
        x = x.float()
        l = x.size(2)
        x = x.reshape(x.size(0), x.size(1), self.input_size)
        x = self.layers(x)
        x = x.reshape(x.size(0), x.size(1), l, -1)
        return x

    def predict(self, dataloader, device, dataset):
      out = []
      tar = []
      for batch_idx, (data, target) in enumerate(dataloader):
          data, target = data.to(device), target.to(device)
          encoded = self.forward(data,dataset)
          if dataset == 'mnist':
            encoded = encoded.reshape(encoded.size(0), 32, 32)
          for i in range(len(encoded)):
            uni = encoded[i]
            out.append(uni.unsqueeze(0))
            tar.append(target[i])
      dataset = ForgetDataset(out, tar)
      loader = DataLoader(dataset, dataloader.batch_size)
      return loader


    '''
    def freeze(self):
        for param in self.layers.parameters():
            param.requires_grad = False
    '''

class AscentFunction(torch.autograd.Function):
    '''Gradient ascent'''
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_input):
        return -grad_input

def make_ascent(loss):
    return AscentFunction.apply(loss)

def model_learning(model, unfreeze):
    for param in model.parameters():
        param.requires_grad = unfreeze
    return model

def nip(v1,v2):
    '''

    :param v1: prediction
    :param v2: actual scrubbing update
    :return: Normalized Inner Product between v1 and v2
    '''
    nip = (np.inner(v1/np.linalg.norm(v1), v2/np.linalg.norm(v2)))
    return nip

def distance(model, model0):
    distance=0
    normalization=0
    for (k, p), (k0, p0) in zip(model.named_parameters(), model0.named_parameters()):
        current_dist=(p-p0).pow(2).sum().item()
        current_norm=p.pow(2).sum().item()
        distance+=current_dist
        normalization+=current_norm
    print(f'Distance: {np.sqrt(distance)}')
    print(f'Normalized Distance: {1.0*np.sqrt(distance/normalization)}')
    return 1.0*np.sqrt(distance/normalization)

def weight_reset(model):
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.Linear) or isinstance(model, nn.Conv1d):
        model.reset_parameters()


#------------------------------------------------------------------------------------------
import numpy as np
import keras.backend as K
from keras.regularizers import Regularizer


def computer_fisher(model, imgset, num_sample=30):
    f_accum = []
    for i in range(len(model.weights)):
        f_accum.append(np.zeros(K.int_shape(model.weights[i])))
    f_accum = np.array(f_accum)
    for j in range(num_sample):
        img_index = np.random.randint(imgset.shape[0])
        for m in range(len(model.weights)):
            grads = K.gradients(K.log(model.output), model.weights)[m]
            result = K.function([model.input], [grads])
            f_accum[m] += np.square(result([np.expand_dims(imgset[img_index], 0)])[0])
    f_accum /= num_sample
    return f_accum


class ewc_reg(Regularizer):
    def __init__(self, fisher, prior_weights, Lambda=0.1):
        self.fisher = fisher
        self.prior_weights = prior_weights
        self.Lambda = Lambda

    def __call__(self, x):
        regularization = 0.
        regularization += self.Lambda * K.sum(self.fisher * K.square(x - self.prior_weights))
        return regularization

    def get_config(self):
        return {'Lambda': float(self.Lambda)}