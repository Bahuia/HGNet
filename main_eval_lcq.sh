#!/bin/bash

devices=2

CUDA_LAUNCH_BLOCKING=1 python -u main_eval.py \
--dataset lcq \
--seed 2021 \
--test_data ./data/LC-QuAD/annotated_test.pkl \
--wo_vocab ./vocab/word_vocab_lcq.pkl \
--not_kb_constraint \
--not_save_result \
--not_matching_score \
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
--max_num_op 20 \
--cpt ./runs/lcq/1653312288/checkpoints/best_snapshot_epoch_80_val_aqg_acc_78.8_val_acc_32.2_model.pt \
--result_name result.pkl \
--kb_endpoint http://10.201.102.90:8890//sparql