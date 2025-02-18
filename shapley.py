"""
shapely.py
Description: implements about different types of approximation methods for shapley values
Author: Haiyang Yu
Date: 2/20/2021
"""
import copy
import torch
import numpy as np
from typing import Union
from scipy.special import comb
from itertools import combinations
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data, Batch, Dataset, DataLoader


def value_func_decorator(value_func):
    """ input: list of the connected graph (X, A)
        return sum of the value of all connected graphs
    """
    def value_sum_func(batch, target_class):
        with torch.no_grad():
            logits, probs, _ = value_func(batch)
            score = probs[:, target_class]
        return score

    return value_sum_func


def GnnNets_GC2value_func(gnnNets, target_class):
    def value_func(batch):
        with torch.no_grad():
            logits, prob, _ = gnnNets(batch)
            score = prob[:, target_class]
        return score
    return value_func


def GnnNets_NC2value_func(gnnNets_NC, node_idx: Union[int, torch.tensor], target_class: torch.tensor):
    def value_func(data):
        with torch.no_grad():
            logits, prob, _ = gnnNets_NC(data)
            # select the corresponding node prob through the node idx on all the sampling graphs
            batch_size = data.batch.max() + 1
            prob = prob.reshape(batch_size, -1, logits.shape[-1])
            score = prob[:, node_idx, target_class]
            return score
    return value_func


def get_graph_build_func(build_method):
    if build_method.lower() == 'zero_filling':
        return graph_build_zero_filling
    elif build_method.lower() == 'split':
        return graph_build_split
    else:
        raise NotImplementedError


class MarginalSubgraphDataset(Dataset):

    def __init__(self, data, exclude_mask, include_mask, subgraph_build_func):
        self.num_nodes = data.num_nodes
        self.X = data.x
        self.edge_index = data.edge_index
        self.device = self.X.device

        self.label = data.y
        self.exclude_mask = torch.tensor(exclude_mask).type(torch.float32).to(self.device)
        self.include_mask = torch.tensor(include_mask).type(torch.float32).to(self.device)
        self.subgraph_build_func = subgraph_build_func

    def __len__(self):
        return self.exclude_mask.shape[0]

    def __getitem__(self, idx):
        exclude_graph_X, exclude_graph_edge_index = self.subgraph_build_func(self.X, self.edge_index, self.exclude_mask[idx])
        include_graph_X, include_graph_edge_index = self.subgraph_build_func(self.X, self.edge_index, self.include_mask[idx])
        exclude_data = Data(x=exclude_graph_X, edge_index=exclude_graph_edge_index)
        include_data = Data(x=include_graph_X, edge_index=include_graph_edge_index)
        return exclude_data, include_data


def marginal_contribution(data: Data, exclude_mask: np.array, include_mask: np.array,
                          value_func, subgraph_build_func):
    """ Calculate the marginal value for each pair. Here exclude_mask and include_mask are node mask. """
    marginal_subgraph_dataset = MarginalSubgraphDataset(data, exclude_mask, include_mask, subgraph_build_func)
    dataloader = DataLoader(marginal_subgraph_dataset, batch_size=256, shuffle=False, pin_memory=True, num_workers=0)

    marginal_contribution_list = []

    for exclude_data, include_data in dataloader:
        exclude_values = value_func(exclude_data)
        include_values = value_func(include_data)
        margin_values = include_values - exclude_values
        marginal_contribution_list.append(margin_values)

    marginal_contributions = torch.cat(marginal_contribution_list, dim=0)
    return marginal_contributions


def graph_build_zero_filling(X, edge_index, node_mask: np.array):
    """ subgraph building through masking the unselected nodes with zero features """
    ret_X = X * node_mask.unsqueeze(1)
    return ret_X, edge_index


def graph_build_split(X, edge_index, node_mask: np.array):
    """ subgraph building through spliting the selected nodes from the original graph """
    row, col = edge_index
    edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
    ret_edge_index = edge_index[:, edge_mask]
    return X, ret_edge_index


def l_shapley(coalition: list, data: Data, local_raduis: int,
              value_func: str, subgraph_building_method='zero_filling'):
    """ shapley value where players are local neighbor nodes """
    graph = to_networkx(data)
    num_nodes = graph.number_of_nodes()
    subgraph_build_func = get_graph_build_func(subgraph_building_method)

    local_region = copy.copy(coalition)
    for k in range(local_raduis - 1):
        k_neiborhoood = []
        for node in local_region:
            k_neiborhoood += list(graph.neighbors(node))
        local_region += k_neiborhoood
        local_region = list(set(local_region))

    set_exclude_masks = []
    set_include_masks = []
    nodes_around = [node for node in local_region if node not in coalition]
    num_nodes_around = len(nodes_around)

    for subset_len in range(0, num_nodes_around + 1):
        node_exclude_subsets = combinations(nodes_around, subset_len)
        for node_exclude_subset in node_exclude_subsets:
            set_exclude_mask = np.ones(num_nodes)
            set_exclude_mask[local_region] = 0.0
            if node_exclude_subset:
                set_exclude_mask[list(node_exclude_subset)] = 1.0
            set_include_mask = set_exclude_mask.copy()
            set_include_mask[coalition] = 1.0

            set_exclude_masks.append(set_exclude_mask)
            set_include_masks.append(set_include_mask)

    exclude_mask = np.stack(set_exclude_masks, axis=0)
    include_mask = np.stack(set_include_masks, axis=0)
    num_players = len(nodes_around) + 1
    num_player_in_set = num_players - 1 + len(coalition) - (1 - exclude_mask).sum(axis=1)
    p = num_players
    S = num_player_in_set
    coeffs = torch.tensor(1.0 / comb(p, S) / (p - S + 1e-6))

    marginal_contributions = \
        marginal_contribution(data, exclude_mask, include_mask, value_func, subgraph_build_func)

    l_shapley_value = (marginal_contributions.squeeze().cpu() * coeffs).sum().item()
    return l_shapley_value


def mc_shapley(coalition: list, data: Data,
               value_func: str, subgraph_building_method='zero_filling',
               sample_num=1000) -> float:
    """ monte carlo sampling approximation of the shapley value """
    subset_build_func = get_graph_build_func(subgraph_building_method)

    num_nodes = data.num_nodes
    node_indices = np.arange(num_nodes)
    coalition_placeholder = num_nodes
    set_exclude_masks = []
    set_include_masks = []

    for example_idx in range(sample_num):
        subset_nodes_from = [node for node in node_indices if node not in coalition]
        random_nodes_permutation = np.array(subset_nodes_from + [coalition_placeholder])
        random_nodes_permutation = np.random.permutation(random_nodes_permutation)
        split_idx = np.where(random_nodes_permutation == coalition_placeholder)[0][0]
        selected_nodes = random_nodes_permutation[:split_idx]
        set_exclude_mask = np.zeros(num_nodes)
        set_exclude_mask[selected_nodes] = 1.0
        set_include_mask = set_exclude_mask.copy()
        set_include_mask[coalition] = 1.0

        set_exclude_masks.append(set_exclude_mask)
        set_include_masks.append(set_include_mask)

    exclude_mask = np.stack(set_exclude_masks, axis=0)
    include_mask = np.stack(set_include_masks, axis=0)
    marginal_contributions = marginal_contribution(data, exclude_mask, include_mask, value_func, subset_build_func)
    mc_shapley_value = marginal_contributions.mean().item()

    return mc_shapley_value


def mc_l_shapley(coalition: list, data: Data, local_raduis: int,
                 value_func: str, subgraph_building_method='zero_filling',
                 sample_num=1000) -> float:
    """ monte carlo sampling approximation of the l_shapley value """
    graph = to_networkx(data)
    num_nodes = graph.number_of_nodes()
    subgraph_build_func = get_graph_build_func(subgraph_building_method)

    local_region = copy.copy(coalition)
    for k in range(local_raduis - 1):
        k_neiborhoood = []
        for node in local_region:
            k_neiborhoood += list(graph.neighbors(node))
        local_region += k_neiborhoood
        local_region = list(set(local_region))

    coalition_placeholder = num_nodes
    set_exclude_masks = []
    set_include_masks = []
    for example_idx in range(sample_num):
        subset_nodes_from = [node for node in local_region if node not in coalition]
        random_nodes_permutation = np.array(subset_nodes_from + [coalition_placeholder])
        random_nodes_permutation = np.random.permutation(random_nodes_permutation)
        split_idx = np.where(random_nodes_permutation == coalition_placeholder)[0][0]
        selected_nodes = random_nodes_permutation[:split_idx]
        set_exclude_mask = np.ones(num_nodes)
        set_exclude_mask[local_region] = 0.0
        set_exclude_mask[selected_nodes] = 1.0
        set_include_mask = set_exclude_mask.copy()
        set_include_mask[coalition] = 1.0

        set_exclude_masks.append(set_exclude_mask)
        set_include_masks.append(set_include_mask)

    exclude_mask = np.stack(set_exclude_masks, axis=0)
    include_mask = np.stack(set_include_masks, axis=0)
    marginal_contributions = \
        marginal_contribution(data, exclude_mask, include_mask, value_func, subgraph_build_func)

    mc_l_shapley_value = (marginal_contributions).mean().item()
    return mc_l_shapley_value


def gnn_score(coalition: list, data: Data, value_func: str,
              subgraph_building_method='zero_filling') -> torch.Tensor:
    """ the value of subgraph with selected nodes """
    num_nodes = data.num_nodes
    subgraph_build_func = get_graph_build_func(subgraph_building_method)
    mask = torch.zeros(num_nodes).type(torch.float32)
    mask[coalition] = 1.0
    ret_x, ret_edge_index = subgraph_build_func(data.x, data.edge_index, mask)
    mask_data = Data(x=ret_x, edge_index=ret_edge_index)
    mask_data = Batch.from_data_list([mask_data])
    score = value_func(mask_data)
    # get the score of predicted class for graph or specific node idx
    return score.item()


def NC_mc_l_shapley(coalition: list, data: Data, local_raduis: int,
                    value_func: str, node_idx: int=-1, subgraph_building_method='zero_filling', sample_num=1000) -> float:
    """ monte carlo approximation of l_shapley where the target node is kept in both subgraph """
    graph = to_networkx(data)
    num_nodes = graph.number_of_nodes()
    subgraph_build_func = get_graph_build_func(subgraph_building_method)

    local_region = copy.copy(coalition)
    for k in range(local_raduis - 1):
        k_neiborhoood = []
        for node in local_region:
            k_neiborhoood += list(graph.neighbors(node))
        local_region += k_neiborhoood
        local_region = list(set(local_region))

    coalition_placeholder = num_nodes
    set_exclude_masks = []
    set_include_masks = []
    for example_idx in range(sample_num):
        subset_nodes_from = [node for node in local_region if node not in coalition]
        random_nodes_permutation = np.array(subset_nodes_from + [coalition_placeholder])
        random_nodes_permutation = np.random.permutation(random_nodes_permutation)
        split_idx = np.where(random_nodes_permutation == coalition_placeholder)[0][0]
        selected_nodes = random_nodes_permutation[:split_idx]
        set_exclude_mask = np.ones(num_nodes)
        set_exclude_mask[local_region] = 0.0
        set_exclude_mask[selected_nodes] = 1.0
        if node_idx != -1:
            set_exclude_mask[node_idx] = 1.0
        set_include_mask = set_exclude_mask.copy()
        set_include_mask[coalition] = 1.0  # include the node_idx

        set_exclude_masks.append(set_exclude_mask)
        set_include_masks.append(set_include_mask)

    exclude_mask = np.stack(set_exclude_masks, axis=0)
    include_mask = np.stack(set_include_masks, axis=0)
    marginal_contributions = \
        marginal_contribution(data, exclude_mask, include_mask, value_func, subgraph_build_func)

    mc_l_shapley_value = (marginal_contributions).mean().item()
    return mc_l_shapley_value

