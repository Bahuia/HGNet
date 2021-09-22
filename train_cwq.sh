#!/bin/bash

devices=3

CUDA_LAUNCH_BLOCKING=1 python -u train.py \
--dataset cwq \
--seed 2021 \
--train_data ./data/ComplexWebQuestions/annotated_train.pkl \
--valid_data ./data/ComplexWebQuestions/annotated_dev.pkl \
--glove_path /home/cyr/resources/GloVe/glove.42B.300d.txt \
--wo_vocab ./vocab/word_vocab_cwq.pkl \
--emb_cache ./vocab/word_embeddings_cache_cwq.pt \
--sparql_cache_path ./vocab/sparql_cache_cwq.pkl \
--save_all_cpt \
--context_mode attention \
--not_matching_score \
--not_mention_feature \
--not_matching_feature \
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
--n_epochs 45 \
--bs 16 \
--ag 1 \
--lr 2e-4 \
--max_num_op 45 \
--beam_size 7