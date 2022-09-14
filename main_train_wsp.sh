#!/bin/bash

devices=2

CUDA_LAUNCH_BLOCKING=1 python -u main_train.py \
--dataset wsp \
--seed 2021 \
--train_data ./data/WebQSP/annotated_train.pkl \
--valid_data ./data/WebQSP/annotated_test.pkl \
--glove_path /home/cyr/resources/GloVe/glove.42B.300d.txt \
--wo_vocab ./vocab/word_vocab_wsp.pkl \
--emb_cache ./vocab/word_embeddings_cache_wsp.pt \
--sparql_cache_path ./vocab/sparql_cache_wsp.pkl \
--context_mode attention \
--not_segment_embedding \
--not_matching_feature \
--not_matching_score \
--not_kb_constraint \
--readout identity \
--att_type affine \
--d_h 256 \
--d_emb 300 \
--d_f 32 \
--gpu $devices \
--n_lstm_layers 1 \
--n_gnn_blocks 3 \
--heads 4 \
--start_valid_epoch 20 \
--n_epochs 150 \
--bs 16 \
--ag 1 \
--lr 2e-4 \
--max_num_op 45 \
--beam_size 7