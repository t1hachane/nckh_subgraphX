# NCKH2025_SubgraphX

This is a variant of the implementation of Paper 
[On Explainability of Graph Neural Networks via Subgraph Explorations](https://arxiv.org/abs/2102.05152)

# Các option thí nghiệm

- Model: GREMI, MOGONET
- Data: 
  - TCGA format: GBM, gồm
    - `1_tr`: feature matrix => trong quá trình chạy thuật toán thì một số cột sẽ được zero ứng với là bỏ feature nào (feature nào được chọn/giữ) => dựa vào `MCTS.coalition`.
    - `labels`: không đổi trong suốt quá trình chạy thuật toán. 
    - `adj1`: hiện tại chưa dùng
  - GREMI format: ROSMAP, BRCA, gồm các `data_tr`, `tr_omic`, `tr_labels`, `data_te`, `te_omic`, `te_labels`, `exp_adj1`, `exp_adj2` lưu dưới dạng dict trong file `.pt`
- Matrix: ma trận kề có thể sinh ra từ `gen_adj_mat_tensor` hoặc `WCGNA`.

# Docs

1. Dataset
0
2. Checkpoint
3. Các file chính
3.1. Utils: cần chú ý tới các hàm
- Load model để infer: `init_model_dict`, `load_model_dict`, `infer_mogonet`.
- 3 hàm để sinh ra matrix cho load data: `gen_adj_mat_tensor`, `gen_test_adj_mat_tensor`, `cal_adj_mat_parameter`.
3.2. Load data: chuẩn bị data
- Hàm prepare feature matrix và label: `prepare_trte_data_tcga_mcts`
- Hàm sinh ra ma trận kề: `gen_trte_adj_mat`
3.3. Cài đặt thuật toán: `mcts.py`
- `MCTS` gồm có 2 phương thức chính: `mcts_rollout` <--gọi-- `run_mcts`, khởi tạo thì có các thuộc tính data được sinh ra từ: `prepare_trte_data_tcga_mcts`, `gen_trte_adj_mat`.
- `get_best_mcts_node`: lấy lá tốt nhất từ các lá của một cây MCTS
- `Explain` có phương thức `explain` --gọi--> `run_mcts` và `get_best_mcts_node`
3.4. Chạy: `subgraphx.py`
- Gọi `MCTS` => chiều sửa code, không cần dùng hàm `prepare_trte_data_tcga_mcts` -`get_best_mcts_node` và nữa vì bản thân `MCTS` đã gọi 2 hàm đó rồi

Import: utils --> load data --> mcts.py
                              |--> subgraphx.py  