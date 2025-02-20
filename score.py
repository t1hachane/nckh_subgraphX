from typing import Any, Dict, List, Optional, Set, Tuple
import torch
from torch_geometric.data import Batch, Data
from utils import *

def gnn_score(data_all_list, adj_tr_list, te_idx, model_dict, coalition: Tuple[int, ...], data: Data, target_class: torch.tensor) -> float:
    print(f'Coalition: {coalition}')
    node_mask = torch.zeros(len(coalition), dtype=torch.bool, device='cpu')
    print(node_mask.shape)
    coalition_tmp = [i - 1 for i in coalition]
    node_mask[coalition_tmp] = 1

    row, col = data.edge_index
    edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)

    mask_edge_index = data.edge_index[:, edge_mask]

    mask_data = Data(x=data.x, edge_index=mask_edge_index)
    mask_data = Batch.from_data_list([mask_data])

    prob = infer_mogonet(data_all_list, adj_tr_list, te_idx, model_dict)
    # score = torch.sigmoid(logits).cpu().detach().numpy()
    score = prob[0][target_class]
    score_total = torch.sum(score)
    return score_total