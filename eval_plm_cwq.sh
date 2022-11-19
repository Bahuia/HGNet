#!/bin/bash

devices=3

CUDA_LAUNCH_BLOCKING=1 python -u eval_plm.py \
--dataset cwq \
--seed 2021 \
--test_path ./data/ComplexWebQuestions/annotated_test.pkl \
--plm_mode bert-base-uncased \
--context_mode attention \
--not_use_mention_feature \
--d_h 256 \
--d_emb 300 \
--gpu $devices \
--n_lstm_layers 1 \
--n_gnn_blocks 3 \
--heads 4 \
--beam_size 5 \
--max_n_step 45 \
--cpt_path ./runs/cwq/1668586785/checkpoints/best_snapshot_epoch_17_val_aqg_acc_82.8030954428203_val_acc_54.22757237030668_model.pt \
--result_path result_plm.pkl \
--kb_endpoint http://10.201.89.70:8890//sparql
