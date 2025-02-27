import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from score import gnn_score
from load_dataset import *

import networkx as nx
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import FloatTensor, LongTensor, Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import k_hop_subgraph, remove_self_loops, to_networkx
from tqdm.notebook import trange

class MCTSNode(object):
    def __init__(self,
                 coalition: Tuple[int, ...],
                #  omics_datas: List[FloatTensor],
                 ori_graph: nx.Graph,
                 c_puct: float = 10.0,
                 W: float = 0,
                 N: int = 0,
                 P: float = 0):
        '''
        Input:
            coalition: Tuple[int, ...] (tập index của các node trong graph hiện tại của node này index so với 2000 gene)
            omics_datas: [list tensor 3 ma trận omics]
            ori_graph: nx.Graph graph gốc của node này (trước khi bị prune thành một subgraph khác cho childnode)
            c_puct: float
            W: float
            N: int
            P: float (immediate reward)
        '''

        assert isinstance(coalition, tuple), f"coalition must be a tuple, not {type(coalition)}"

        # assert isinstance(omics_datas, list), f"omics_datas must be a list, not {type(omics_datas)}"

        assert isinstance(ori_graph, nx.Graph), f"ori_graph must be a nx.Graph, not {type(ori_graph)}"

        self.coalition = coalition
        # self.omics_data = omics_datas

        self.ori_graph = ori_graph
        self.c_puct = c_puct
        self.W = W
        self.N = N
        self.P = P
        self.children: List[MCTSNode] = []

        # # zero out the features of the nodes that are not in the coalition 
        # # dang hardcore la omic 1
        # self.omics_data[0] = self.omics_data[0].clone()
        # self.omics_data[0][list(set(range(2000)) - set(coalition))] = 0

    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def U(self, n: int) -> float:
        return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)

    @property
    def size(self) -> int:
        return len(self.coalition)

    
def get_best_mcts_node(results, max_nodes):
    '''
    Input: 
        results: List[MCTSNode]

        max_nodes: int (the subgraphs may contain the maximum number of nodes that <= min_nodes (an predefined upper bound - like top 500 biomarkers) like 490) => can continue to get the subgraph with the number of nodes <= 300 for example (tiep tuc lay top 300 tu top 500 (490) da tim duoc)
    Output: 
        best_result: MCTSNode
    '''
    results = [result for result in results if 2 <= result.size <= max_nodes]
    if len(results) == 0:
        raise ValueError(f'All subgraphs have more than {max_nodes} nodes.')
    results = sorted(results, key=lambda result: result.size)
    best_result = max(results, key=lambda result: (result.P, -result.size))

    return best_result

class MCTS(object):
    def __init__(self,
                #  omics_datas_orig: List[FloatTensor],
                 edge_index: LongTensor,
                 model_dict: Dict[str, Any],
                 adj_tr_list,
                #  num_hops: int,
                 n_rollout: int,
                 min_nodes: int,
                 c_puct: float,
                 num_expand_nodes: int,
                 high2low: bool,
                 data_folder: str,
                 view_list: list,
                 ):
        '''
        Input:
            omics_datas_orig: [list tensor 3 ma trận omics] ban đầu chưa bị zero out feature nào


        '''
        # assert isinstance(omics_datas_orig, list), f"omics_datas must be a list, not {type(omics_datas_orig)}"
        assert isinstance(edge_index, LongTensor), f"edge_index must be a 2D LongTensor, not {type(edge_index)}"
        # assert isinstance(model_dict, dict), f"model_dict must be a dict, not {type(model_dict)}"

        assert edge_index.size(0) == 2, f"edge_index must have 2 rows, got {edge_index.size(0)}"


        # self.omics_datas_orig = omics_datas_orig
        self.edge_index = edge_index
        self.model_dict = model_dict
        # self.num_hops = num_hops # số lượng lớp trong mạng GNN
        '''
            Hyperparameters
            - min_nodes: trong 1 subgraph, khi subgraph có số node nhỏ hơn min_node thì dừng prune
            - c_puct: hyperparameter lambda
            - num_expand_nodes: con số k để chọn top k node with highest degrees xem xét prune

        '''
        self.n_rollout = n_rollout
        self.min_nodes = min_nodes 
        self.c_puct = c_puct
        self.num_expand_nodes = num_expand_nodes  
        self.high2low = high2low 

        self.data_train_list, self.data_all_list, self.idx_dict, self.labels = prepare_trte_data_tcga_mcts(data_folder, view_list)


        # self.adj_tr_list, self.adj_te_list = gen_trte_adj_mat(self.data_train_list, self.data_all_list, self.idx_dict, adj_parameter=10)
        self.adj_tr_list = adj_tr_list

        self.graph = nx.from_numpy_array(self.adj_tr_list[0].to_dense().cpu().numpy())
        self.num_nodes = self.graph.number_of_nodes()

        self.root_coalition = tuple(range(self.num_nodes))
        self.MCTSNodeClass = partial(MCTSNode,
                                    #  omics_datas=self.omics_datas_orig, 
                                     ori_graph=self.graph, 
                                     c_puct=self.c_puct)

        
        self.root = self.MCTSNodeClass(coalition=self.root_coalition)
        self.state_map = {self.root.coalition: self.root}


    def mcts_rollout(self, tree_node: MCTSNode) -> float:
        assert isinstance(tree_node, MCTSNode), f"tree_node must be a MCTSNode, not {type(tree_node)}"

        if len(tree_node.coalition) <= self.min_nodes:
            print(f"len(tree_node.coalition) = {len(tree_node.coalition)}")
            print(tree_node.coalition)
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
                print("EXPAND_NODE", expand_node)
                # tạo 1 subgraph mới -> có thể có nhiều thành phần liên thông -> nên mới được gọi là subgraph_coalition
                subgraph_coalition = all_nodes_set - {expand_node}

                print("ALL_NODES_SET", len(all_nodes_set))
                print(f"SUBGRAPH_COALITION, {len(subgraph_coalition)}")

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

                # xoa bo di 2 not nhung cung co the dem toi 1 connected component giong nhau => vi chon component lon nhat trong cac thanh phan lien thong sau khi xoa di 2 not do => khong phai node nao cung co k con => co toi da k con
                if new_coalition not in tree_children_coalitions:
                    tree_node.children.append(new_node)
                    # print(f"new_node.coalition, {new_node.coalition}")
                    tree_children_coalitions.add(new_coalition)

            for child in tree_node.children:
                if child.P == 0:
                    child.P = gnn_score(self.data_train_list, self.adj_tr_list, 
                    self.idx_dict, 
                    model_dict=self.model_dict, coalition=child.coalition, 
                    # data=child.omics_data, 
                    target_class=self.labels)

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
        return explanations


# SỬA truyền vào những gì init của Explain
class Explain(object):
    def __init__(self,
                 model_dict,
                 adj_tr_list,
                #  num_hops: int,
                 n_rollout: int = 3,
                 min_nodes: int = 500,
                 c_puct: float = 10.0,
                 num_expand_nodes: int = 14,
                 high2low: bool = False):
        self.model_dict = model_dict
        self.adj_tr_list = adj_tr_list
        # self.model_dict.eval()
        # self.num_hops = num_hops

        # if self.num_hops is None:
        #     self.num_hops = sum(isinstance(module, MessagePassing) for module in self.model.modules())

        # MCTS hyperparameters
        self.n_rollout = n_rollout
        self.min_nodes = min_nodes
        self.c_puct = c_puct
        self.num_expand_nodes = num_expand_nodes
        self.high2low = high2low

    def explain(self, edge_index, max_nodes, data_folder, view_list):
        '''
        Input: 
            x: tensor m x n matrix (m patients, n genes)

            edge_index: tensor 2 x m matrix (m edges)

            max_nodes: int for def get_best_mcts_node(results, max_nodes)

            data_folder: str

            view_list: list
        Output: 
            MCTSNode
        '''
        mcts = MCTS(
            edge_index=edge_index,
            model_dict=self.model_dict,
            adj_tr_list=self.adj_tr_list,
            # num_hops=self.num_hops,
            n_rollout=self.n_rollout,
            min_nodes=self.min_nodes,
            c_puct=self.c_puct,
            num_expand_nodes=self.num_expand_nodes,
            high2low=self.high2low,

            data_folder=data_folder,
            view_list=view_list
        )

        # Run the MCTS search
        mcts_nodes = mcts.run_mcts()

        # select the subgraph that has at most max_nodes nodes
        best_mcts_node = get_best_mcts_node(mcts_nodes, max_nodes=max_nodes)

        return best_mcts_node