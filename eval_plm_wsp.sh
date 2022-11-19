#!/bin/bash

devices=3

CUDA_LAUNCH_BLOCKING=1 python -u eval_plm.py \
--dataset wsp \
--seed 2021 \
--test_path ./data/WebQSP/annotated_test.pkl \
--not_use_eg \
--plm_mode bert-base-uncased \
--not_use_segment_embedding \
--context_mode attention \
--d_h 256 \
--d_emb 300 \
--gpu $devices \
--n_lstm_layers 1 \
--n_gnn_blocks 3 \
--heads 4 \
--beam_size 7 \
--max_n_step 45 \
--cpt_path ./runs/wsp/1653310474/checkpoints/best_snapshot_epoch_29_val_aqg_acc_86.46288209606988_val_acc_63.630692451653154_model.pt \
--result_path result_plm.pkl \
--kb_endpoint http://10.201.89.70:8890//sparql
