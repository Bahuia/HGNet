#!/bin/bash

devices=3

CUDA_LAUNCH_BLOCKING=1 python -u train.py \
--dataset lcq \
--seed 2021 \
--train_path ./data/LC-QuAD/annotated_train.pkl \
--valid_path ./data/LC-QuAD/annotated_dev.pkl \
--glove_path /home/cyr/resources/GloVe/glove.42B.300d.txt \
--vocab_path ./vocab/word_vocab_lcq.pkl \
--embed_cache_path ./vocab/word_embeddings_cache_lcq.pt \
--context_mode attention \
--not_use_segment_embedding \
--not_use_eg \
--readout identity \
--att_type affine \
--d_h 256 \
--d_emb 300 \
--gpu $devices \
--n_lstm_layers 1 \
--n_gnn_blocks 3 \
--heads 4 \
--n_valid_epochs 10 \
--n_epochs 80 \
--bs 16 \
--ag 1 \
--lr 2e-4 \
--max_n_step 20 \
--beam_size 7 \
--not_copy_v \
--not_copy_e \
--training_proportion 0.05