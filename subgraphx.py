import numpy as np
# Env
from utils import *
from mcts import Explain
from load_dataset import prepare_trte_data_tcga_mcts, gen_trte_adj_mat
def pipeline(dataset_name, model_name):
    if dataset_name == 'GBM':
        data_folder = 'TCGA_GBM_GExCNVxMETH_2000_MinMaxScaler\1'

    if model_name == "MOGONET":
        model_folder = 'checkpoint/GBM_MOGONET'
        view_list = [1,2,3]
        data_tr_list, data_trte_list, trte_idx, labels = prepare_trte_data_tcga_mcts(data_folder, view_list)
        adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter=10)

        model_dict = init_model_dict(num_view=len(view_list), num_class=len(np.unique(labels)), dim_list=[data_tr_list[i].shape[1] for i in range(len(data_tr_list))], dim_he_list=[250,300,150], dim_hc=20)
        model_dict = load_model_dict(model_folder, model_dict)
        
        num_biomarkers = 200
        model_explain = Explain(model=model_dict, min_nodes=num_biomarkers)

        biomerkers_idx = model_explain.coalition

    return biomerkers_idx

if __name__ == '__main__':
    dataset_name = 'GBM'
    model_name = 'MOGONET'
    pipeline(dataset_name, model_name)
