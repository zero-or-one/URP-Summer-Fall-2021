import numpy as np
import torch 
import torch.nn.functional as F
import torch.optim as optim

def class_accuracy(pred_labels, true_labels):
    return sum(y_pred == y_act for y_pred, y_act in zip(pred_labels, true_labels))

# L1 regularization
def l1_loss(model,model_init,weight_decay):
    l2_loss = 0
    for (k,p),(k_init,p_init) in zip(model.named_parameters(),model_init.named_parameters()):
        if p.requires_grad:
            l2_loss +=  (p-p_init).pow(2).sum()
    l2_loss *= (weight_decay/2.)
    return l2_loss

# L2 regularization
def l2_loss(model,model_init,weight_decay):
    l2_loss = 0
    for (k,p),(k_init,p_init) in zip(model.named_parameters(), model_init.named_parameters()):
        if p.requires_grad:
            l2_loss +=  (p-p_init).pow(2).sum()
    l2_loss *= (weight_decay/2.)
    return l2_loss

def configure_learning_rate(optimizer, epoch):
    pass