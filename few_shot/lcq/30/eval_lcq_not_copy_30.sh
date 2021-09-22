#!/bin/bash

devices=2

CUDA_LAUNCH_BLOCKING=1 python -u ../../../eval.py \
--dataset lcq \
--seed 2021 \
--test_data ../../../data/LC-QuAD/annotated_test.pkl \
--wo_vocab ../../../vocab/word_vocab_lcq.pkl \
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
--alpha 0.2 \
--beta 0 \
--sparql_cache_path ../../../vocab/sparql_cache_lcq.pkl \
--cpt ./runs/lcq/1628742187/checkpoints/best_snapshot_epoch_97_val_aqg_acc_69.0_val_acc_9.4_model.pt \
--kb_endpoint http://10.201.61.163:8890//sparql \
--not_copy_e \
--not_copy_v
