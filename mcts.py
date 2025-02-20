import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from score import gnn_score
from load_dataset import *

import networkx as nx
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import FloatTensor, LongTensor, Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import k_hop_subgraph, remove_self_loops, to_networkx
from tqdm.notebook import trange

class MCTSNode(object):
    def __init__(self,
                 coalition: Tuple[int, ...],
                 data: Data,
                 ori_graph: nx.Graph,
                 c_puct: float,
                 W: float = 0,
                 N: int = 0,
                 P: float = 0) -> None:
        # ten cac node (cac feature duoc chon) cua subgraph thuoc node hien tai
        self.coalition = coalition

        # input feature matrix ung voi cac feature duoc chon (cua benh nhan sample TCGA trong ma tran ban dau m sample x n gene)
        self.data = data

        # graph cua node cha thuoc node nay? hoac graph cua node goc
        self.ori_graph = ori_graph

        # hyper parameter lamda
        self.c_puct = c_puct

        # tong reward cua cac subgraph cua node con cua node hien tai
        self.W = W

        # so lan node nay duoc tham/chon
        self.N = N

        # reward cua node hien tai
        self.P = P # Immediate reward
        self.children: List[MCTSNode] = []

    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def U(self, n: int) -> float:
        return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)

    @property
    def size(self) -> int:
        return len(self.coalition)
    
def get_best_mcts_node(results: List[MCTSNode], max_nodes: int) -> MCTSNode:
    results = [result for result in results if 2 <= result.size <= max_nodes]
    if len(results) == 0:
        raise ValueError(f'All subgraphs have more than {max_nodes} nodes.')
    results = sorted(results, key=lambda result: result.size)
    best_result = max(results, key=lambda result: (result.P, -result.size))

    return best_result

class MCTS(object):
    def __init__(self,
                 x: FloatTensor,
                 edge_index: LongTensor,
                 model: torch.nn.Module,
                 num_hops: int,
                 n_rollout: int,
                 min_nodes: int,
                 c_puct: float,
                 num_expand_nodes: int,
                 high2low: bool,
                 data_folder: str,
                 view_list: list,
                 ) -> None:
        self.x = x
        self.edge_index = edge_index
        self.num_hops = num_hops # số lượng lớp trong mạng GNN

        self.model = model
        self.data = Data(x=self.x, edge_index=self.edge_index)
        self.graph = to_networkx(
            Data(x=self.x, edge_index=remove_self_loops(self.edge_index)[0]),
            to_undirected=True
        )

        self.data = Batch.from_data_list([self.data])
        self.num_nodes = self.graph.number_of_nodes()
        self.n_rollout = n_rollout
        self.min_nodes = min_nodes # min node trong 1 subgraph, khi subgraph có số node nhỏ hơn min_node thì dừng prune
        self.c_puct = c_puct
        self.num_expand_nodes = num_expand_nodes # con số k để chọn top k node with highest degrees xem xét prune 
        self.high2low = high2low 

        self.root_coalition = tuple(range(self.num_nodes))
        self.MCTSNodeClass = partial(MCTSNode, data=self.data, ori_graph=self.graph, c_puct=self.c_puct)
        self.root = self.MCTSNodeClass(coalition=self.root_coalition)
        self.state_map = {self.root.coalition: self.root}
        self.data_train_list, self.data_all_list, self.idx_dict, self.labels = prepare_trte_data_tcga_mcts(data_folder, view_list)
        self.adj_tr_list, self.adj_te_list = gen_trte_adj_mat(self.data_train_list, self.data_all_list, self.idx_dict, adj_parameter=10)

    def mcts_rollout(self, tree_node: MCTSNode) -> float:
        if len(tree_node.coalition) <= self.min_nodes:
            return tree_node.P
        if len(tree_node.children) == 0:
            tree_children_coalitions = set()
            tree_subgraph = self.graph.subgraph(tree_node.coalition)
            all_nodes = sorted(
                tree_subgraph.nodes,
                key=lambda node: tree_subgraph.degree[node],
                reverse=self.high2low
            )
            all_nodes_set = set(all_nodes)
            expand_nodes = all_nodes[:self.num_expand_nodes]

            # top k node xem xét để bỏ (bị prune)
            for expand_node in expand_nodes:

                # tạo 1 subgraph mới -> có thể có nhiều thành phần liên thông -> nên mới được gọi là subgraph_coalition
                subgraph_coalition = all_nodes_set - {expand_node}

                # số lượng connected component còn lại khi xóa bỏ each_node và cạnh liên kết với each_node -> tập các subgraph nên mới có chữ s
                subgraphs = (
                    self.graph.subgraph(connected_component)
                    for connected_component in nx.connected_components(self.graph.subgraph(subgraph_coalition))
                )

                # chọn connected component có số lượng node lớn nhất
                subgraph = max(subgraphs, key=lambda subgraph: subgraph.number_of_nodes())

                # tạo 1 subgraph mới từ MỘT connected component lớn nhất
                new_coalition = tuple(sorted(subgraph.nodes()))

                # tạo 1 node mới từ subgraph mới
                new_node = self.state_map.setdefault(new_coalition, self.MCTSNodeClass(coalition=new_coalition))
                if new_coalition not in tree_children_coalitions:
                    tree_node.children.append(new_node)
                    tree_children_coalitions.add(new_coalition)

            for child in tree_node.children:
                if child.P == 0:
                    child.P = gnn_score(self.data_all_list, self.adj_tr_list, self.idx_dict, self.labels, coalition=child.coalition, data=child.data, model=self.model)

        sum_count = sum(child.N for child in tree_node.children)
        selected_node = max(tree_node.children, key=lambda x: x.Q() + x.U(n=sum_count))
        v = self.mcts_rollout(tree_node=selected_node)
        selected_node.W += v
        selected_node.N += 1

        return v

    def run_mcts(self) -> List[MCTSNode]:
        for _ in trange(self.n_rollout):
            self.mcts_rollout(tree_node=self.root)

        explanations = [node for _, node in self.state_map.items()]
        explanations = sorted(explanations, key=lambda x: (x.P, -x.size), reverse=True)
        print(len(explanations))
        return explanations


class Explain(object):

    def __init__(self,
                 model: torch.nn.Module,
                 num_hops: Optional[int] = None,
                 n_rollout: int = 20,
                 min_nodes: int = 5,
                 c_puct: float = 10.0,
                 num_expand_nodes: int = 14,
                 high2low: bool = False) -> None:
        self.model = model
        self.model.eval()
        self.num_hops = num_hops

        if self.num_hops is None:
            self.num_hops = sum(isinstance(module, MessagePassing) for module in self.model.modules())

        # MCTS hyperparameters
        self.n_rollout = n_rollout
        self.min_nodes = min_nodes
        self.c_puct = c_puct
        self.num_expand_nodes = num_expand_nodes
        self.high2low = high2low

    def explain(self, x: Tensor, edge_index: Tensor, max_nodes: int) -> MCTSNode:
        # Create an MCTS object with the provided graph
        mcts = MCTS(
            x=x,
            edge_index=edge_index,
            model=self.model,
            num_hops=self.num_hops,
            n_rollout=self.n_rollout,
            min_nodes=self.min_nodes,
            c_puct=self.c_puct,
            num_expand_nodes=self.num_expand_nodes,
            high2low=self.high2low
        )

        # Run the MCTS search
        mcts_nodes = mcts.run_mcts()
        # such that the subgraph has at most max_nodes nodes
        best_mcts_node = get_best_mcts_node(mcts_nodes, max_nodes=max_nodes)

        print(type(best_mcts_node.coalition))
        return best_mcts_node