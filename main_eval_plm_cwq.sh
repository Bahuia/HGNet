#!/bin/bash

devices=1

CUDA_LAUNCH_BLOCKING=1 python -u main_eval_plm.py \
--dataset cwq \
--seed 2021 \
--test_data ./data/ComplexWebQuestions/annotated_test.pkl \
--plm_mode bert-base-uncased \
--not_copy_v \
--not_copy_e \
--context_mode attention \
--not_matching_score \
--not_mention_feature \
--not_matching_feature \
--d_h 256 \
--d_emb 300 \
--d_f 32 \
--gpu $devices \
--n_lstm_layers 1 \
--n_gnn_blocks 3 \
--heads 4 \
--beam_size 5 \
--max_num_op 45 \
--sparql_cache_path ./vocab/sparql_cache_cwq.pkl \
--cpt ./runs/cwq/1653210400/checkpoints/best_snapshot_epoch_20_val_aqg_acc_84.72341645170536_val_acc_54.77214101461737_model.pt \
--result_name result_plm.pkl \
--kb_endpoint http://10.201.102.90:8890//sparql
