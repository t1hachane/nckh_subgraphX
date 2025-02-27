from typing import Any, Dict, List, Optional, Set, Tuple
import torch
from torch_geometric.data import Batch, Data
from utils import infer_gremi
import numpy as np

def gnn_score(data_train_list, 
              adj_tr_list, 
              idx_dict,
              model_dict, 
              coalition: Tuple[int, ...], 
            #   data: Data, 
              target_class: torch.tensor) -> float:
    
    tr_idx = idx_dict['tr']

    coalition_tmp = [i - 1 for i in coalition]
    num_feat = data_train_list[0].shape[1]
    data_train_list[0][:, list(set(range(num_feat)) - set(coalition_tmp))] = 0

    # prob = infer_mogonet(data_train_list, adj_tr_list, model_dict)
    prob = infer_gremi(data_train_list, adj_tr_list, model_dict)
    # score = torch.sigmoid(logits).cpu().detach().numpy()
    target_class = target_class[tr_idx]
    score = prob[0][target_class]
    score_total = np.sum(score)
    return score_total