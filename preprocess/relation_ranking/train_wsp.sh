#!/bin/bash

devices=3

CUDA_LAUNCH_BLOCKING=1 python -u train.py \
--dataset wsp \
--seed 2021 \
--train_data ../../data/WebQSP/parsed_train.json \
--valid_data ../../data/WebQSP/parsed_test.json \
--rel_pool ../../data/WebQSP/relation_pool.json \
--glove_path /home/cyr/resources/GloVe/glove.42B.300d.txt \
--wo_vocab ../../vocab/word_vocab_wsp.pkl \
--emb_cache ../../vocab/word_embeddings_cache_wsp.pt \
--d_h_wo 512 \
--d_emb_wo 300 \
--gpu $devices \
--n_lstm_layers 1 \
--n_epochs 30 \
--ns 50 \
--bs 16 \
--lr 1e-3 \
--rel_topk 50 \
--margin 0.1