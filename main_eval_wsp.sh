#!/bin/bash

devices=3

CUDA_LAUNCH_BLOCKING=1 python -u main_eval.py \
--dataset wsp \
--seed 2021 \
--test_data ./data/WebQSP/annotated_test.pkl \
--wo_vocab ./vocab/word_vocab_wsp.pkl \
--not_kb_constraint \
--not_save_result \
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
--cpt ./runs/wsp/1652622095/checkpoints/best_snapshot_epoch_67_val_aqg_acc_79.47598253275109_val_acc_57.01809107922645_model.pt \
--result_name result.pkl \
--kb_endpoint http://10.201.102.90:8890//sparql
