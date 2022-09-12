#!/bin/bash

devices=2

CUDA_LAUNCH_BLOCKING=1 python -u main_eval_plm.py \
--dataset wsp \
--seed 2021 \
--test_data ./data/WebQSP/annotated_test.pkl \
--plm_mode bert-base-uncased \
--not_segment_embedding \
--not_matching_feature \
--context_mode attention \
--d_h 256 \
--d_emb 300 \
--d_f 32 \
--gpu $devices \
--n_lstm_layers 1 \
--n_gnn_blocks 3 \
--heads 4 \
--beam_size 7 \
--max_num_op 45 \
--cpt ./runs/wsp/1653310474/checkpoints/best_snapshot_epoch_29_val_aqg_acc_86.46288209606988_val_acc_63.630692451653154_model.pt \
--result_name result_plm.pkl \
--kb_endpoint http://10.201.102.90:8890//sparql
