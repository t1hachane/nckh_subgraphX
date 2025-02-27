import numpy as np
# Env
import torch
from utils import load_model_GREMI
from mcts import Explain
from load_dataset import prepare_trte_data_tcga_mcts, gen_trte_adj_mat
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def pipeline(model_folder, data_folder, view_list,
             n_rollout, min_nodes, c_puct, num_expand_nodes, high2low, max_node_afterMCTS, omic = 1):
        
    data_tr_list, data_trte_list, trte_idx, labels = prepare_trte_data_tcga_mcts(data_folder, view_list)
    # adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter=10)

    exp_adj1 = torch.tensor(pd.read_csv(f"{data_folder}/adj1.csv", header=0, index_col=0).values, dtype=torch.float32)
    
    exp_adj2 = torch.tensor(pd.read_csv(f"{data_folder}/adj2.csv", header=0, index_col=0).values, dtype=torch.float32)

    exp_adj3 = torch.tensor(pd.read_csv(f"{data_folder}/adj3.csv", header=0, index_col=0).values, dtype=torch.float32)

    adj_tr_list = torch.stack([exp_adj1, exp_adj2, exp_adj3])
    # model_dict = init_model_dict(num_view=len(view_list), num_class=len(np.unique(labels)), dim_list=[data_tr_list[i].shape[1] for i in range(len(data_tr_list))], dim_he_list=[250,300,150], dim_hc=64)
    # model_dict = load_model_dict(model_folder, model_dict)

    model_dict = load_model_GREMI(num_class=len(np.unique(labels)), view_list=view_list, in_dim=[data_tr_list[i].shape[1] for i in range(len(data_tr_list))], model_folder=model_folder, device=device)
    # hardcore tạm :)) omic = 0
    adj_dense = adj_tr_list[0].to_dense()
    edge_index_omic = torch.tensor(np.where(adj_dense == 1), dtype=torch.long)
    edge_index_omic = torch.cat([edge_index_omic, torch.flip(edge_index_omic, [0])], dim=1)
    
    model_explain = Explain(model_dict, adj_tr_list=adj_tr_list,
                            n_rollout=n_rollout, min_nodes=min_nodes, c_puct=c_puct, num_expand_nodes=num_expand_nodes,high2low=high2low)

    best_MCTSNode = model_explain.explain(
    edge_index=edge_index_omic, 
    max_nodes=max_node_afterMCTS, 
    data_folder=data_folder,
    view_list=view_list)

    biomarkers_idx = best_MCTSNode.coalition

    return biomarkers_idx

if __name__ == '__main__':
    model_folder = 'checkpoint/GBM_GREMI'
    data_folder = 'datasets/GBM_700/1/'
    view_list = [1,2,3]
    n_rollout = 3
    min_nodes = 400
    c_puct = 10.0
    num_expand_nodes = 5
    high2low = True
    max_node_afterMCTS = 500
    omic = 1

    biomarker_idx = pipeline(model_folder, data_folder, view_list, n_rollout, min_nodes, c_puct, num_expand_nodes, high2low, max_node_afterMCTS, omic)

    print(biomarker_idx)

