import numpy as np
import os
import numpy as np
import torch
import torch.nn as nn
import csv
import codecs
import torch_geometric
from scipy.sparse import coo_matrix 
from models.models import GCN_E, Classifier_1, VCDN
from models.model_GREMI import Fusion
import torch.nn.functional as F


cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

def init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hc, gcn_dopout=0.5):
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i+1)] = GCN_E(dim_list[i], dim_he_list, gcn_dopout)
        model_dict["C{:}".format(i+1)] = Classifier_1(dim_he_list[-1], num_class)
    if num_view >= 2:
        model_dict["C"] = VCDN(num_view, num_class, dim_hc)
    return model_dict

def load_model_dict(folder, model_dict):
    device = torch.device("cuda" if cuda else "cpu")
    for module in model_dict:
        if os.path.exists(os.path.join(folder, module+".pth")):
#            print("Module {:} loaded!".format(module)) 
            if device.type == 'cuda':
                model_dict[module].load_state_dict(torch.load(os.path.join(folder, module+".pth"), map_location="cuda:{:}".format(torch.cuda.current_device())))
            else: 
                model_dict[module].load_state_dict(
                    torch.load(os.path.join(folder, module+".pth"), 
                             map_location='cpu')
                )
        else:
            print("WARNING: Module {:} from model_dict is not loaded!".format(module))
        if cuda:
            model_dict[module].cuda()    
    return model_dict

def infer_mogonet(data_list, adj_list, model_dict):
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
        prob = F.softmax(c, dim=1).data.cpu().numpy()
        return prob
    
def load_model_GREMI(num_class, view_list, in_dim, model_folder, device="cpu"):
    model = Fusion(
        num_class=num_class,
        num_views=len(view_list),
        hidden_dim=[64],
        dropout=0.1,
        in_dim=in_dim,
        dim1=in_dim[0],
        dim2=in_dim[1],
        dim3=in_dim[2],
    ).to(device)

    checkpoint = torch.load(os.path.join(model_folder, f"best_model.pth"), map_location=device)
    model.load_state_dict(checkpoint["net"])
    model.eval()
    return model
    
def infer_gremi(data_list, adj_list, checkpoint):
    omic1, omic2, omic3 = data_list[0], data_list[1], data_list[2]
    new_omic1 = omic1.reshape(-1, omic1.shape[1], 1)
    new_omic2 = omic2.reshape(-1, omic2.shape[1], 1)
    new_omic3 = omic3.reshape(-1, omic3.shape[1], 1)
    adj1, adj2, adj3 = adj_list[0], adj_list[1], adj_list[2]
    pred = checkpoint.infer(new_omic1, new_omic2, new_omic3, adj1, adj2, adj3)
    prob = F.softmax(pred, dim=1).data.cpu().numpy()
    return prob
    
# ------GEN ADJ MAT TENSOR------
def to_sparse(x):
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def graph_from_dist_tensor(dist, parameter, self_dist=True):
    if self_dist:
        assert dist.shape[0]==dist.shape[1], "Input is not pairwise dist matrix"
    g = (dist <= parameter).float()
    if self_dist:
        diag_idx = np.diag_indices(g.shape[0])
        g[diag_idx[0], diag_idx[1]] = 0
        
    return g

def gen_adj_mat_tensor(data, parameter, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    g = graph_from_dist_tensor(dist, parameter, self_dist=True)
    if metric == "cosine":
        adj = 1-dist
    else:
        raise NotImplementedError
    adj = adj*g 
    adj_T = adj.transpose(0,1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T*(adj_T > adj).float() - adj*(adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    adj = to_sparse(adj)
    
    return adj

def gen_test_adj_mat_tensor(data, trte_idx, parameter, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    adj = torch.zeros((data.shape[0], data.shape[0]))
    if cuda:
        adj = adj.cuda()
    num_tr = len(trte_idx["tr"])
    
    dist_tr2te = cosine_distance_torch(data[trte_idx["tr"]], data[trte_idx["te"]])
    g_tr2te = graph_from_dist_tensor(dist_tr2te, parameter, self_dist=False)
    if metric == "cosine":
        adj[:num_tr,num_tr:] = 1-dist_tr2te
    else:
        raise NotImplementedError
    adj[:num_tr,num_tr:] = adj[:num_tr,num_tr:]*g_tr2te
    
    dist_te2tr = cosine_distance_torch(data[trte_idx["te"]], data[trte_idx["tr"]])
    g_te2tr = graph_from_dist_tensor(dist_te2tr, parameter, self_dist=False)
    if metric == "cosine":
        adj[num_tr:,:num_tr] = 1-dist_te2tr
    else:
        raise NotImplementedError
    adj[num_tr:,:num_tr] = adj[num_tr:,:num_tr]*g_te2tr # retain selected edges
    
    adj_T = adj.transpose(0,1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T*(adj_T > adj).float() - adj*(adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    adj = to_sparse(adj)
    
    return adj

def cal_adj_mat_parameter(edge_per_node, data, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    parameter = torch.sort(dist.reshape(-1,)).values[edge_per_node*data.shape[0]]
    return parameter.data.cpu().numpy().item()

################
# Layer Utils
################
def define_act_layer(act_type='Tanh'):
    if act_type == 'Tanh':
        act_layer = nn.Tanh()
    elif act_type == 'ReLU':
        act_layer = nn.ReLU()
    elif act_type == 'Sigmoid':
        act_layer = nn.Sigmoid()
    elif act_type == 'LSM':
        act_layer = nn.LogSoftmax(dim=1)
    elif act_type == "none":
        act_layer = None
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return act_layer

################
################
def adj_to_PyG_edge_index(adj):
    coo_A = coo_matrix(adj)
    edge_index, edge_weight = torch_geometric.utils.convert.from_scipy_sparse_matrix(coo_A)
    return edge_index

def data_to_PyG_data(x, edge_index, y):
    out_data = x
    out_edge_index = edge_index
    out_label = y
    PyG_data = torch_geometric.data.Data(x=out_data, edge_index=out_edge_index, y=out_label)
    return PyG_data

def PyG_edge_index_to_adj(edge_index):
    adj = torch_geometric.utils.to_dense_adj(edge_index=edge_index)
    return adj

def data_write_csv(file_name, datas):#
  file_csv = codecs.open(file_name,'w+','utf-8')#
  writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
  for data in datas:
    writer.writerow(data)
  print("csv saved")