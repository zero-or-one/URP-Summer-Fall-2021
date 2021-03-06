
import os
import sys
import numpy as np
from eval_utils import *
from sklearn.svm import SVC
import seaborn as sns
import torch.nn as nn
#sys.path.append("../URP")


def distance_old(model, model0):
    distance=0
    normalization=0
    for (k, p), (k0, p0) in zip(model.named_parameters(), model0.named_parameters()):
        current_dist=(p-p0).pow(2).sum().item()
        current_norm=p.pow(2).sum().item()
        distance+=current_dist
        normalization+=current_norm
    print(f'Distance: {np.sqrt(distance)}')
    print(f'Normalized Distance: {1.0*np.sqrt(distance/normalization)}')
    return np.sqrt(distance)

def distance(model, model0):
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
    #return distance
    
def KL(modelf, model,  num_classes=10, alpha=1e-6):
    # Computes the amount of information not forgotten at all layers using the given alpha
    total_kl = 0
    for (k, p), (k0, p0) in zip(modelf.named_parameters(), model.named_parameters()):
        mu0, var0 = get_pdf(p, num_classes, False, alpha=alpha)
        mu1, var1 = get_pdf(p0, True, alpha=alpha)
        kl = kl_divergence(mu0, var0, mu1, var1).item()
        total_kl += kl
        print(k, f'{kl:.1f}')
    print("Total:", total_kl)
    return total_kl

def activations_predictions(model,dataloader,dataset):
    criterion = torch.nn.CrossEntropyLoss()
    metrics,activations,predictions=get_metrics(model,dataloader,criterion,True)
    print(f"{name} -> Loss:{np.round(metrics['loss'],3)}, Error:{metrics['error']}")
    log_dict[f"{name}_loss"]=metrics['loss']
    log_dict[f"{name}_error"]=metrics['error']
    return activations,predictions
    
def predictions_distance(l1,l2,name):
    dist = np.sum(np.abs(l1-l2))
    print(f"Predictions Distance {name} -> {dist}")
    log_dict[f"{name}_predictions"]=dist
    
def activations_distance(a1,a2,name):
    dist = np.linalg.norm(a1-a2,ord=1,axis=1).mean()
    print(f"Activations Distance {name} -> {dist}")
    log_dict[f"{name}_activations"]=dist
    


def test_activations(model_scrubf, modelf0, delta_w_s, delta_w_m0, data_loader, \
                     loss_fn=nn.CrossEntropyLoss(), \
                     optimizer=torch.optim.SGD, \
                     seed=1, quiet=False):
    model_scrubf.eval()
    modelf0.eval()

    data_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=1, shuffle=False)

    metrics = AverageMeter()
    num_classes = data_loader.dataset.targets.max().item() + 1

    for idx, batch in enumerate(tqdm(data_loader, leave=False)):
        batch = [tensor.to(next(model_scrubf.parameters()).device) for tensor in batch]
        input, target = batch

        output_sf = model_scrubf(input)
        G_sf = []

        for cls in range(num_classes):
            grads = torch.autograd.grad(output_sf[0, cls], model_scrubf.parameters(), retain_graph=True)
            grads = torch.cat([g.view(-1) for g in grads])
            G_sf.append(grads)

        grads = torch.autograd.grad(output_sf[0, cls], model_scrubf.parameters(), retain_graph=False)

        G_sf = torch.stack(G_sf)  # .pow(2)
        delta_f_sf_update = torch.matmul(G_sf, delta_w_s.sqrt() * torch.empty_like(delta_w_s).normal_())
        G_sf = G_sf.pow(2)
        delta_f_sf = torch.matmul(G_sf, delta_w_s)

        output_m0 = modelf0(input)
        G_m0 = []

        for cls in range(num_classes):
            grads = torch.autograd.grad(output_m0[0, cls], modelf0.parameters(), retain_graph=True)
            grad_m0 = torch.cat([g.view(-1) for g in grads])
            G_m0.append(grad_m0)

        grads = torch.autograd.grad(output_m0[0, cls], modelf0.parameters(), retain_graph=False)

        G_m0 = torch.stack(G_m0).pow(2)
        delta_f_m0 = torch.matmul(G_m0, delta_w_m0)

        kl = ((output_m0 - output_sf).pow(2) / delta_f_m0 + delta_f_sf / delta_f_m0 - torch.log(
            delta_f_sf / delta_f_m0) - 1).sum()

        torch.manual_seed(seed)
        output_sf += delta_f_sf_update  # delta_f_sf.sqrt()*torch.empty_like(delta_f_sf).normal_()

        loss = loss_fn(output_sf, target)
        metrics.update(n=input.size(0), loss=loss.item(), error=get_error(output_sf, target), kl=kl.item())

    return metrics.avg

def get_variance(model1,model2,alpha,seed=1):
    
    delta_w_s = []
    delta_w_m0 = []
    
    for i, (k,p) in enumerate(model1.named_parameters()):
        mu, var = get_mean_var(p, False, alpha=alpha)
        delta_w_s.append(var.view(-1))

    for i, (k,p) in enumerate(model2.named_parameters()):
        mu, var = get_mean_var(p, False, alpha=alpha)
        delta_w_m0.append(var.view(-1))

    return torch.cat(delta_w_s), torch.cat(delta_w_m0)
    

def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


import matplotlib
def plot_info(ax,df,information_list,title,no_barplot):
    
    if no_barplot:
        sns.lineplot(x="info", y="error",data=df,ci='sd',ax=ax)
        ax.set(xscale="log")#,yscale='log')
        ax.set_xlabel('Remaining Information (NATs)',size=16)
        ax.set_ylabel('Test Error (%)',size=16)
        ax.set_title(title,size=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.tick_params(axis="x", labelsize=16) 
    else:
        y_pos = range(len(information_list))
        ax.grid(zorder=0)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.bar(y_pos, information_list, align='center', color=matplotlib.cm.get_cmap('tab10')(0.95), width=0.5,capsize=5)
        ax.set_title('Information in Activations',size=18)
        ax.set_facecolor('whitesmoke')
        ax.tick_params(axis="y", labelsize=18)
        ax.set_xticks(y_pos)
        ax.set_xticklabels(labels=labels, size=18, rotation=45)
        ylabel='NATs'
        ax.set_ylabel(ylabel,size=18)
        ax.set_ylim(bottom=-0.001)
        
    ax.set_facecolor(np.array([231,231,240])/256)#'whitesmoke')
    ax.grid(color='white')


def collect_prob(data_loader, model):
    data_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=1, shuffle=False)
    prob = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader, leave=False)):
            batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            data, target = batch
            output = model(data)
            prob.append(F.softmax(output, dim=-1).data)
    return torch.cat(prob)


def get_membership_attack_data(retain_loader, forget_loader, test_loader, model):
    retain_prob = collect_prob(retain_loader, model)
    forget_prob = collect_prob(forget_loader, model)
    test_prob = collect_prob(test_loader, model)

    X_r = torch.cat([entropy(retain_prob), entropy(test_prob)]).cpu().numpy().reshape(-1, 1)
    Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])

    X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    Y_f = np.concatenate([np.ones(len(forget_prob))])
    return X_f, Y_f, X_r, Y_r

def get_membership_attack_prob(retain_loader, forget_loader, test_loader, model):
    # ????
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(retain_loader, forget_loader, test_loader, model)
    clf = SVC(C=3, gamma='auto', kernel='rbf')
    # clf = LogisticRegression(class_weight='balanced',solver='lbfgs',multi_class='multinomial')
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    return results.mean()

def plot_entropy_dist(model, ax, title, datasets, class_to_forget, num_to_forget):
    train_loader_full, test_loader_full = datasets.get_loaders(dataset, batch_size=100, seed=0, augment=False,
                                                               shuffle=False)
    indexes = np.flatnonzero(np.array(train_loader_full.dataset.targets) == class_to_forget)
    replaced = np.random.RandomState(0).choice(indexes, size=100 if num_to_forget == 100 else len(indexes),
                                               replace=False)
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(train_loader_full, test_loader_full, model, replaced)
    sns.distplot(np.log(X_r[Y_r == 1]).reshape(-1), kde=False, norm_hist=True, rug=False, label='retain', ax=ax)
    sns.distplot(np.log(X_r[Y_r == 0]).reshape(-1), kde=False, norm_hist=True, rug=False, label='test', ax=ax)
    sns.distplot(np.log(X_f).reshape(-1), kde=False, norm_hist=True, rug=False, label='forget', ax=ax)
    ax.legend(prop={'size': 14})
    ax.tick_params(labelsize=12)
    ax.set_title(title, size=18)
    ax.set_xlabel('Log of Entropy', size=14)
    ax.set_ylim(0, 0.4)
    ax.set_xlim(-35, 2)


def membership_attack(retain_loader, forget_loader, test_loader, model):
    prob = get_membership_attack_prob(retain_loader, forget_loader, test_loader, model)
    print("Attack prob: ", prob)
    return prob


def compute_forgetting(task_no, dataloader, dset_size, use_gpu):
    """
    Inputs
    1) task_no: The task number on which you want to compute the forgetting
    2) dataloader: The dataloader that feeds in the data to the model
    Outputs
    1) forgetting: The amount of forgetting undergone by the model
    Function: Computes the "forgetting" that the model has on the
    """

    # get the results file
    store_path = os.path.join(os.getcwd(), "models", "Task_" + str(task_no))
    model_path = os.path.join(os.getcwd(), "models")
    device = torch.device("cuda:0" if use_gpu else "cpu")

    # get the old performance
    file_object = open(os.path.join(store_path, "performance.txt"), 'r')
    old_performance = file_object.read()
    file_object.close()

    # load the model for inference
    model = model_inference(task_no, use_gpu=False)
    model.to(device)

    running_corrects = 0

    for data in dataloader:
        input_data, labels = data
        del data

        if (use_gpu):
            input_data = input_data.to(device)
            labels = labels.to(device)

        else:
            input_data = Variable(input_data)
            labels = Variable(labels)

        output = model.tmodel(input_data)
        del input_data

        _, preds = torch.max(output, 1)

        running_corrects += torch.sum(preds == labels.data)
        del preds
        del labels

    epoch_accuracy = running_corrects.double() / dset_size

    old_performance = float(old_performance)
    forgetting = epoch_accuracy.item() - old_performance

    return forgetting



'''
def feature_injection_test(X, Y, remove_sizes, num_repeats=50, reg=1e-4, outliers_to_remove=None):

    (n, d) = X.shape
    ws = {
        'xs': remove_sizes,
        'no removal': np.zeros((len(remove_sizes), num_repeats)),
        'residual': np.zeros((len(remove_sizes), num_repeats)),
        'influence': np.zeros((len(remove_sizes), num_repeats)),
        # 'retrain':      np.zeros((len(remove_sizes), num_repeats)),
    }
    X_extra_col = np.append(X, np.zeros((n, 1)), axis=1)

    for j in range(num_repeats):
        if outliers_to_remove is None:
            viable_indices = [idx for idx in range(n) if Y[idx] == 1]
        else:
            viable_indices = outliers_to_remove
        special_indices = []
        for i in range(len(ws['xs'])):
            size = ws['xs'][i]
            size_to_select = size - len(special_indices)
            new_selected_indices = np.random.choice(viable_indices, size=size_to_select, replace=False)
            viable_indices = [idx for idx in viable_indices if idx not in new_selected_indices]
            special_indices = special_indices + list(new_selected_indices)

            X_injected = np.copy(X_extra_col)
            X_injected[special_indices, -1] = 1
            indices_to_keep = [idx for idx in range(n) if idx not in special_indices]

            theta_original_extra_col = lin_exact(X_injected, Y, reg=reg)
            theta_retrain_extra_col = lin_exact(X_injected[indices_to_keep], Y[indices_to_keep], reg=reg)
            theta_inf_extra_col = lin_inf(X_injected, Y, theta_original_extra_col, special_indices, reg=reg)
            theta_res_extra_col = lin_res(X_injected, Y, theta_original_extra_col, special_indices, reg=reg)

            ws['no removal'][i, j] = theta_original_extra_col[-1] / theta_original_extra_col[-1]
            ws['influence'][i, j] = theta_inf_extra_col[-1] / theta_original_extra_col[-1]
            ws['residual'][i, j] = theta_res_extra_col[-1] / theta_original_extra_col[-1]
            # ws['retrain'][i, j] = theta_retrain_extra_col[-1]     / theta_original_extra_col[-1]

    return ws
'''
