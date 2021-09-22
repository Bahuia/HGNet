#!/bin/bash

devices=1

CUDA_LAUNCH_BLOCKING=1 python -u train.py \
--dataset lcq \
--seed 2021 \
--train_data ../../data/LC-QuAD/parsed_train.json \
--valid_data ../../data/LC-QuAD/parsed_dev.json \
--rel_pool ../../data/LC-QuAD/relation_pool.json \
--glove_path /home/cyr/resources/GloVe/glove.42B.300d.txt \
--wo_vocab ../../vocab/word_vocab_lcq.pkl \
--emb_cache ../../vocab/word_embeddings_cache_lcq.pt \
--d_h_wo 512 \
--d_emb_wo 300 \
--gpu $devices \
--n_lstm_layers 1 \
--n_epochs 30 \
--ns 50 \
--bs 16 \
--lr 1e-3 \
--rel_topk 10 \
--margin 0.1