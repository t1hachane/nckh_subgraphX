""" Training and testing of the model
"""
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from models import init_model_dict, init_optim
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter
import copy

cuda = True if torch.cuda.is_available() else False

class EarlyStopping:
    
    def __init__(self, patience=7, verbose=False, delta=0.001):
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.best_epoch = None
        self.early_stop = False
        self.metric_higher_better_max = 0.0
        self.delta = delta

    def __call__(self, metric_higher_better, epoch, model_dict):

        score = metric_higher_better

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric_higher_better, epoch, model_dict)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f'Early stop at epoch {epoch}th after {self.counter} epochs not increasing score from epoch {self.best_epoch}th with best score {self.best_score}')
        else:
            self.best_score = score
            self.save_checkpoint(metric_higher_better, epoch, model_dict)
            self.counter = 0

    def save_checkpoint(self, metric_higher_better, epoch, model_dict):
        self.best_weights = copy.deepcopy(model_dict)
        self.metric_higher_better_max = metric_higher_better
        self.best_epoch = epoch


# def prepare_trte_data(data_folder, view_list):
def prepare_trte_data(data_folder, view_list, postfix_tr='_tr', postfix_te='_val'):
    # load train, test data
    num_view = len(view_list)
    # begin: need to fix
    labels_tr = np.loadtxt(os.path.join(data_folder, f"labels{postfix_tr}.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, f"labels{postfix_te}.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)

    data_tr_list = []
    data_te_list = []
    for i in view_list:
        # data loading and processing (feature selection, normalizing)
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+f"{postfix_tr}.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+f"{postfix_te}.csv"), delimiter=','))
    # end
    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))
    data_train_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                       data_tensor_list[i][idx_dict["te"]].clone()),0))
    labels = np.concatenate((labels_tr, labels_te))
    
    return data_train_list, data_all_list, idx_dict, labels


def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine" # cosine distance
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        adj_test_list.append(gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric))
    
    return adj_train_list, adj_test_list


def train_epoch(data_list, adj_list, label, one_hot_label, sample_weight, model_dict, optim_dict, train_VCDN=True):
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for m in model_dict:
        model_dict[m].train()    
    num_view = len(data_list)
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)].zero_grad()
        ci_loss = 0
        ci = model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i]))
        ci_loss = torch.mean(torch.mul(criterion(ci, label),sample_weight))
        ci_loss.backward()
        optim_dict["C{:}".format(i+1)].step()
        loss_dict["C{:}".format(i+1)] = ci_loss.detach().cpu().numpy().item()
    if train_VCDN and num_view >= 2:
        optim_dict["C"].zero_grad()
        c_loss = 0
        ci_list = []
        for i in range(num_view):
            ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i])))
        c = model_dict["C"](ci_list)    
        c_loss = torch.mean(torch.mul(criterion(c, label),sample_weight))
        c_loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()
    
    return loss_dict
    

def test_epoch(data_list, adj_list, te_idx, model_dict):
    for m in model_dict:
        model_dict[m].eval()
    with torch.no_grad(): #added to save memory ram and increasing speed of inference/evaluation
        num_view = len(data_list)
        ci_list = []
        for i in range(num_view):
            ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i])))
        if num_view >= 2:
            c = model_dict["C"](ci_list)    
        else:
            c = ci_list[0]
        c = c[te_idx,:]
        prob = F.softmax(c, dim=1).data.cpu().numpy()
        return prob


def train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c, 
               num_epoch_pretrain, num_epoch,
               rseed,
               postfix_tr='_tr', postfix_te='_val',
               patience=20,
               verbose = True,
               dim_he_list=[400,400,200]):
    if rseed>=0:
        torch.manual_seed(rseed)
        np.random.seed(rseed)

    test_inverval = 50
    num_view = len(view_list)
    dim_hvcdn = pow(num_class,num_view)
    if num_class == 2:
        adj_parameter = 2
        dim_he_list = [200,200,100]
    if num_class > 2:
        adj_parameter = 10
        # dim_he_list = [400,400,200]

    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list, postfix_tr, postfix_te)

    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
    dim_list = [x.shape[1] for x in data_tr_list]
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()
    
    print("\nPretrain GCNs...")
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    for epoch in range(num_epoch_pretrain):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, 
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_VCDN=False)
    print("\nTraining...")
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)

    if patience is not None:
        # using early stopping: 
        early_stopping = EarlyStopping(patience = patience, verbose = verbose) #added
    for epoch in range(num_epoch+1):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, 
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)
        
        te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
        te_acc = accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
        if num_class == 2:
            te_f1 = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
            te_auc = roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:,1])
        else:
            te_f1_weighted = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')
            te_f1_macro = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')

        if verbose:
            if epoch % test_inverval == 0:
                print("\nTest: Epoch {:d}".format(epoch))
                print("Test ACC: {:.3f}".format(te_acc))
                if num_class == 2:
                    print("Test F1: {:.3f}".format(te_f1))
                    print("Test AUC: {:.3f}".format(te_auc))
                else:
                    print("Test F1 weighted: {:.3f}".format(te_f1_weighted))
                    print("Test F1 macro: {:.3f}".format(te_f1_macro))
                print()
        if patience is not None:
            # using early stopping:
            early_stopping(te_acc, epoch, model_dict)
            if early_stopping.early_stop:
                # Throw back best weight/ best model dict
                model_dict = early_stopping.best_weights
                break
    return model_dict