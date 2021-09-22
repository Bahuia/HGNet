#!/bin/bash

devices=1

CUDA_LAUNCH_BLOCKING=1 python -u ../eval_non_hierarchical.py \
--dataset wsp \
--seed 2021 \
--test_data ../../data/WebQSP/annotated_test.pkl \
--wo_vocab ../../vocab/word_vocab_wsp.pkl \
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
--sparql_cache_path ../../vocab/sparql_cache_wsp.pkl \
--cpt ./runs/wsp/1626661262/checkpoints/best_snapshot_epoch_34_val_aqg_acc_39.86275733000624_val_acc_28.44666250779788_model.pt \
--kb_endpoint http://10.201.69.194:8890//sparql