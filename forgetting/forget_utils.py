import os
import sys
sys.path.append("../URP")
from learning.learn_utils import *
from learning.learn import *
from utils import *
import numpy as np
import time
import torch


class Encoder(nn.Module):
    '''Encode the data to carry destructive information about its features'''
    def __init__(self, input_size=512, hidden_size=16, num_layer=2, activation=nn.ReLU()):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.activation = activation
        self.layers = self._make_layers()

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

    def forward(self, x):
        x = x.float()
        x = x.reshape(x.size(0), self.input_size)
        return self.layers(x)

def NIP(v1,v2):
    '''

    :param v1: prediction
    :param v2: actual scrubbing update
    :return: Normalized Inner Product between v1 and v2
    '''
    nip = (np.inner(v1/np.linalg.norm(v1),v2/np.linalg.norm(v2)))
    return nip

def parameter_count(model):
    count=0
    for p in model.parameters():
        count+=np.prod(np.array(list(p.shape)))
    print(f'Total Number of Parameters: {count}')

def vectorize_params(model):
    param = []
    for p in model.parameters():
        param.append(p.data.view(-1).cpu().numpy())
    return np.concatenate(param)

def print_param_shape(model):
    for k,p in model.named_parameters():
        print(k,p.shape)

def distance(model,model0):
    distance=0
    normalization=0
    for (k, p), (k0, p0) in zip(model.named_parameters(), model0.named_parameters()):
        space='  ' if 'bias' in k else ''
        current_dist=(p.data0-p0.data0).pow(2).sum().item()
        current_norm=p.data0.pow(2).sum().item()
        distance+=current_dist
        normalization+=current_norm
    print(f'Distance: {np.sqrt(distance)}')
    print(f'Normalized Distance: {1.0*np.sqrt(distance/normalization)}')
    return 1.0*np.sqrt(distance/normalization)


def delta_w_utils(model_init,dataloader, lossfn, dataset, num_classes, model, name='complete'):
    model_init.eval()
    dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=1, shuffle=False)
    G_list = []
    f0_minus_y = []
    for idx, batch in enumerate(dataloader):#(tqdm(dataloader,leave=False)):
        batch = [tensor.to(next(model_init.parameters()).device) for tensor in batch]
        input, target = batch
        if 'mnist' in dataset:
            input = input.view(input.shape[0],-1)
        target = target.cpu().detach().numpy()
        output = model_init(input)
        G_sample=[]
        for cls in range(num_classes):
            grads = torch.autograd.grad(output[0,cls],model_init.parameters(),retain_graph=True)
            grads = np.concatenate([g.view(-1).cpu().numpy() for g in grads])
            G_sample.append(grads)
            G_list.append(grads)
        if lossfn=='mse':
            p = output.cpu().detach().numpy().transpose()
            #loss_hess = np.eye(len(p))
            target = 2*target-1
            f0_y_update = p-target
        elif lossfn=='ce':
            p = torch.nn.functional.softmax(output,dim=1).cpu().detach().numpy().transpose()
            p[target]-=1
            f0_y_update = model.deepcopy(p)
        f0_minus_y.append(f0_y_update)
    return np.stack(G_list).transpose(), np.vstack(f0_minus_y)
