#!/bin/bash

devices=2

CUDA_LAUNCH_BLOCKING=1 python -u eval.py \
--dataset lcq \
--seed 2021 \
--test_path ./data/LC-QuAD/annotated_test.pkl \
--vocab_path ./vocab/word_vocab_lcq.pkl \
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
--cpt_path ./runs/lcq/1653312288/checkpoints/best_snapshot_epoch_80_val_aqg_acc_78.8_val_acc_32.2_model.pt \
--result_path result.pkl \
--kb_endpoint http://10.201.89.70:8890//sparql