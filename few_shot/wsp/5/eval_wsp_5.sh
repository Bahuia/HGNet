#!/bin/bash

devices=3

CUDA_LAUNCH_BLOCKING=1 python -u ../../../eval.py \
--dataset wsp \
--seed 2021 \
--test_data ../../../data/WebQSP/annotated_test.pkl \
--wo_vocab ../../../vocab/word_vocab_wsp.pkl \
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
--sparql_cache_path ../../../vocab/sparql_cache_wsp.pkl \
--cpt ./runs/wsp/1631093025/checkpoints/best_snapshot_epoch_49_val_aqg_acc_65.3150343106675_val_acc_20.524017467248907_model.pt \
--kb_endpoint http://10.201.69.194:8890//sparql

# http://10.201.188.200:8890//sparql freebase