import os
from re import I
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
# from model import MMDynamic
from utils import gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter
import copy

def prepare_trte_data_tcga_mcts(data_folder, view_list, postfix_tr='_tr', postfix_te='_val'):
    # load train, test data
    num_view = len(view_list)
    # begin: need to fix
    print(data_folder)
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
    
    # adj_matrix_list = []
    # for i in view_list:
    #     adj_matrix_list.append(np.loadtxt(os.path.join(data_folder, f"adj{i}.csv"), delimiter=','))
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

def prepare_data_gremi_mcts(data_folder, view_list, omic = 1, dataset = '_tr'): 
    data = torch.load(data_folder, weights_only=False)
    data_tr = data['data_tr']
    data_te = data['data_te']
    tr_labels = data['tr_labels']
    te_labels = data['te_labels']
    exp_adj1 = data['exp_adj1']
    exp_adj2 = data['exp_adj2']
    exp_adj3 = data['exp_adj3']
    return data_tr, data_te, tr_labels, te_labels, exp_adj1, exp_adj2, exp_adj3