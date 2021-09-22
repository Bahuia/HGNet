#!/bin/bash

devices=3

CUDA_LAUNCH_BLOCKING=1 python -u predict.py \
--dataset wsp \
--test_data ../../data/WebQSP/parsed_train.json \
--output ../../data/WebQSP/candidate_relations_train \
--rel_pool ../../data/WebQSP/relation_pool.json \
--wo_vocab ../../vocab/word_vocab_wsp.pkl \
--emb_cache ../../vocab/word_embeddings_cache_wsp.pt \
--d_h_wo 512 \
--d_emb_wo 300 \
--rel_topk 50 \
--gpu $devices \
--st_pos 0 \
--ed_pos 3000 \
--cpt ./runs/wsp/1624800424/checkpoints/best_snapshot_epoch_13_best_val_recall_93.9026402640264_model.pt

CUDA_LAUNCH_BLOCKING=1 python -u predict.py \
--dataset wsp \
--test_data ../../data/WebQSP/parsed_train.json \
--output ../../data/WebQSP/candidate_relations_train \
--rel_pool ../../data/WebQSP/relation_pool.json \
--wo_vocab ../../vocab/word_vocab_wsp.pkl \
--emb_cache ../../vocab/word_embeddings_cache_wsp.pt \
--d_h_wo 512 \
--d_emb_wo 300 \
--rel_topk 50 \
--gpu $devices \
--st_pos 3000 \
--ed_pos 6000 \
--cpt ./runs/wsp/1624800424/checkpoints/best_snapshot_epoch_13_best_val_recall_93.9026402640264_model.pt

CUDA_LAUNCH_BLOCKING=1 python -u predict.py \
--dataset wsp \
--test_data ../../data/WebQSP/parsed_test.json \
--output ../../data/WebQSP/candidate_relations_test \
--rel_pool ../../data/WebQSP/relation_pool.json \
--wo_vocab ../../vocab/word_vocab_wsp.pkl \
--emb_cache ../../vocab/word_embeddings_cache_wsp.pt \
--d_h_wo 512 \
--d_emb_wo 300 \
--rel_topk 50 \
--gpu $devices \
--st_pos 0 \
--ed_pos 3000 \
--cpt ./runs/wsp/1624800424/checkpoints/best_snapshot_epoch_13_best_val_recall_93.9026402640264_model.pt
