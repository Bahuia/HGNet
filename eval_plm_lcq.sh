#!/bin/bash

devices=3

CUDA_LAUNCH_BLOCKING=1 python -u eval_plm.py \
--dataset lcq \
--seed 2021 \
--test_path ./data/LC-QuAD/annotated_test.pkl \
--plm_mode bert-base-uncased \
--not_use_subgraph \
--not_use_segment_embedding \
--context_mode attention \
--d_h 256 \
--d_emb 300 \
--gpu $devices \
--n_lstm_layers 1 \
--n_gnn_blocks 3 \
--heads 4 \
--beam_size 7 \
--max_n_step 20 \
--cpt_path ./runs/lcq/1653361300/checkpoints/best_snapshot_epoch_62_val_aqg_acc_83.6_val_acc_43.2_model.pt \
--result_path result_plm.pkl \
--kb_endpoint http://10.201.89.70:8890//sparql