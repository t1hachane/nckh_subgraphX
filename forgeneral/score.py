import math
from argparse import ArgumentParser, Namespace
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from typing_extensions import Literal

import ipywidgets as widgets
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import FloatTensor, LongTensor, Tensor
from torch_geometric.data import Batch, Data, InMemoryDataset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINConv, global_max_pool
from torch_geometric.explain import GNNExplainer
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models import GIN
from torch_geometric.utils import k_hop_subgraph, remove_self_loops, to_networkx
from tqdm.notebook import trange, tqdm

def gnn_score(coalition: Tuple[int, ...], data: Data, model: torch.nn.Module, target_class: torch.tensor) -> float:
    print(f'Coalition: {coalition}')
    node_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
    node_mask[list(coalition)] = 1

    row, col = data.edge_index
    edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)

    mask_edge_index = data.edge_index[:, edge_mask]

    mask_data = Data(x=data.x, edge_index=mask_edge_index)
    mask_data = Batch.from_data_list([mask_data])

    logits = model(x=mask_data.x, edge_index=mask_data.edge_index, batch=mask_data.batch)
    # score = torch.sigmoid(logits).cpu().detach().numpy()
    prob = torch.sigmoid(logits).cpu().detach().numpy()
    score = prob[0][target_class]
    score_total = torch.sum(score)
    return score_total