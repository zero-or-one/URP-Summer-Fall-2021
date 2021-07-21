import numpy as np
import torch 
import torch.nn.functional as F
import torch.optim as optim

def class_accuracy(pred_labels, true_labels):
    return sum(y_pred == y_act for y_pred, y_act in zip(pred_labels, true_labels))

# L1 regularization
def l1_term(model, weight_decay):
    loss = sum([l.norm(1) for l in model.parameters()])
    return weight_decay * loss

# L2 regularization
def l2_term(model,model_init,weight_decay):
    loss = 0
    for (_, x), (_, x0) in zip(model.named_parameters(), model_init.named_parameters()):
        if x0.requires_grad:
            loss +=  ((x-x0) *  (x-x0)).sum()
    return loss * (weight_decay / 2.)


def configure_learning_rate(optimizer, epoch):
    pass


def fit():
    pass