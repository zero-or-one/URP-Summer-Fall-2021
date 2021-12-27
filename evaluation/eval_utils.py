import copy
import sys
sys.path.append("../URP")
from utils import *
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from sklearn.svm import SVC


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

def copy_params(model, model0):
    for p in model.parameters():
        p.data0 = p.data.clone()
    for p in model0.parameters():
        p.data0 = p.data.clone()

def get_pdf(p, num_classes, is_base_dist=False, alpha=3e-6):
    var = copy.deepcopy(1./(p.grad2_acc+1e-8))
    var = var.clamp(max=1e3)
    if p.size(0) == num_classes:
        var = var.clamp(max=1e2)
    var = alpha * var
    
    if p.ndim > 1:
        var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
    if not is_base_dist:
        mu = copy.deepcopy(p.data0.clone())
    else:
        mu = copy.deepcopy(p.data0.clone())
    if p.size(0) == num_classes and num_to_forget is None:
        mu[class_to_forget] = 0
        var[class_to_forget] = 0.0001
    if p.size(0) == num_classes:
        # Last layer
        var *= 10
    elif p.ndim == 1:
        # BatchNorm
        var *= 10
#         var*=1
    return mu, var


def l2_distance(weights, weights_retrain):
    l2 = np.sum(weights**2 - weights_retrain**2)
    l2 = np.sqrt(l2)
    return l2

def kl_divergence(mu0, var0, mu1, var1):
    return ((mu1 - mu0).pow(2)/var0 + var1/var0 - torch.log(var1/var0) - 1).sum()
'''
def kl_divergence(p, q):
    return np.sum(p[i] * np.log2(p[i]/q[i]) for i in range(len(p)))
'''


def get_variance(model1, model2, alpha):
    delta_w_s = []
    delta_w_m0 = []

    for i, (k, p) in enumerate(model1.named_parameters()):
        mu, var = get_pdf(p, False, alpha=alpha)
        delta_w_s.append(var.view(-1))

    for i, (k, p) in enumerate(model2.named_parameters()):
        mu, var = get_pdf(p, False, alpha=alpha)
        delta_w_m0.append(var.view(-1))
    return torch.cat(delta_w_s), torch.cat(delta_w_m0)


def get_metrics(model,dataloader,criterion, lossfn='ce', dataset='cifar10', samples_correctness=False,use_bn=False,delta_w=None,scrub_act=False, device='cuda'):
    activations=[]
    predictions=[]
    if use_bn:
        model.train()
        dataloader = torch.utils.data.DataLoader(retain_loader.dataset, batch_size=128, shuffle=True)
        for i in range(10):
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                output = model(data)
    dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=1, shuffle=False)
    model.eval()
    metrics = AverageMeter()
    mult = 0.5 if lossfn=='mse' else 1
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        if lossfn=='mse':
            target=(2*target-1)
            target = target.type(torch.cuda.FloatTensor).unsqueeze(1)
        if 'mnist' in dataset:
            data=data.view(data.shape[0],-1)
        output = model(data)
        loss = mult*criterion(output, target)
        if samples_correctness:
            activations.append(torch.nn.functional.softmax(output,dim=1).cpu().detach().numpy().squeeze())
            predictions.append(get_error(output,target))
        metrics.update(n=data.size(0), loss=loss.item(), error=get_error(output, target))
    if samples_correctness:
        return metrics.avg, np.stack(activations), np.array(predictions)
    else:
        return metrics.avg

def get_error(output, target):
    if output.shape[1]>1:
        pred = output.argmax(dim=1, keepdim=True)
        return 1. - pred.eq(target.view_as(pred)).float().mean().item()
    else:
        pred = output.clone()
        pred[pred>0]=1
        pred[pred<=0]=-1
        return 1 - pred.eq(target.view_as(pred)).float().mean().item()
        
        
def save_dict(m0_name, log_dict):
    np.save(f"logs/{m0_name.split('/')[1].split('.')[0]}.npy", log_dict)


def hessian(dataset, model, device='cuda'):
    model.eval()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()

    for p in model.parameters():
        p.grad_acc = 0
        p.grad2_acc = 0
    
    for data, orig_target in tqdm(train_loader):
        data, orig_target = data.to(device), orig_target.to(device)
        output = model(data)
        prob = F.softmax(output, dim=-1).data

        for y in range(output.shape[1]):
            target = torch.empty_like(orig_target).fill_(y)
            loss = loss_fn(output, target)
            model.zero_grad()
            loss.backward(retain_graph=True)
            for p in model.parameters():
                if p.requires_grad:
                    p.grad_acc += (orig_target == target).float() * p.grad.data
                    p.grad2_acc += prob[:, y] * p.grad.data.pow(2)
    for p in model.parameters():
        p.grad_acc /= len(train_loader)
        p.grad2_acc /= len(train_loader)

'''
def delta_w_utils(model_init, dataloader, lossfn, dataset, num_classes, model, name='complete'):
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
'''