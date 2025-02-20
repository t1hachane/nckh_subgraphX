import numpy as np
import networkx as nx
import rdkit.Chem as Chem
import matplotlib.pyplot as plt
from textwrap import wrap

def infer_model(model_name, model_path):
    if model_name == "MOGONET":
        data_list, adj_list,
        
    pred = 

    pass

def test_epoch_mogonet(data_list, adj_list, te_idx, model_dict):
    for m in model_dict:
        model_dict[m].eval()
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