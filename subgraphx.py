import numpy as np
# Env
from utils import *
from mcts import Explain
from load_dataset import prepare_trte_data_tcga_mcts, gen_trte_adj_mat

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def pipeline(model_folder, data_folder, view_list,
             n_rollout, min_nodes, c_puct, num_expand_nodes, high2low, max_node_afterMCTS, omic = 1):
        
    data_tr_list, data_trte_list, trte_idx, labels = prepare_trte_data_tcga_mcts(data_folder, view_list)
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter=10)

    model_dict = init_model_dict(num_view=len(view_list), num_class=len(np.unique(labels)), dim_list=[data_tr_list[i].shape[1] for i in range(len(data_tr_list))], dim_he_list=[250,300,150], dim_hc=64)
    model_dict = load_model_dict(model_folder, model_dict)
        
    # hardcore tạm :)) omic = 0
    adj_dense = adj_tr_list[0].to_dense()
    edge_index_omic = torch.tensor(np.where(adj_dense == 1), dtype=torch.long)
    edge_index_omic = torch.cat([edge_index_omic, torch.flip(edge_index_omic, [0])], dim=1)
    
    model_explain = Explain(model_dict, 
                            n_rollout=n_rollout, min_nodes=min_nodes, c_puct=c_puct, num_expand_nodes=num_expand_nodes, high2low=high2low)

    best_MCTSNode = model_explain.explain(x=data_tr_list, 
    edge_index=edge_index_omic, 
    max_nodes=max_node_afterMCTS, data_folder=data_folder,
    view_list=view_list)

    biomarkers_idx = best_MCTSNode.coalition

    return biomarkers_idx

if __name__ == '__main__':
    model_folder = 'checkpoint/GBM_MOGONET'
    data_folder = 'datasets/GBM/1/'
    view_list = [1,2,3]
    n_rollout = 3
    min_nodes = 100
    c_puct = 10.0
    num_expand_nodes = 10
    high2low = True
    max_node_afterMCTS = 500
    omic = 1

    biomarker_idx = pipeline(model_folder, data_folder, view_list, n_rollout, min_nodes, c_puct, num_expand_nodes, high2low, max_node_afterMCTS, omic)

    print(biomarker_idx)

