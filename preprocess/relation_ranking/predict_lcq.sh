#!/bin/bash

devices=1

CUDA_LAUNCH_BLOCKING=1 python -u predict.py \
--dataset lcq \
--test_data ../../data/LC-QuAD/parsed_train.json \
--output ../../data/LC-QuAD/candidate_relations_train \
--rel_pool ../../data/LC-QuAD/relation_pool.json \
--wo_vocab ../../vocab/word_vocab_lcq.pkl \
--emb_cache ../../vocab/word_embeddings_cache_lcq.pt \
--d_h_wo 512 \
--d_emb_wo 300 \
--rel_topk 50 \
--gpu $devices \
--st_pos 0 \
--ed_pos 3000 \
--cpt ./runs/lcq/1624799696/checkpoints/best_snapshot_epoch_10_best_val_recall_87.80000000000003_model.pt

CUDA_LAUNCH_BLOCKING=1 python -u predict.py \
--dataset lcq \
--test_data ../../data/LC-QuAD/parsed_train.json \
--output ../../data/LC-QuAD/candidate_relations_train \
--rel_pool ../../data/LC-QuAD/relation_pool.json \
--wo_vocab ../../vocab/word_vocab_lcq.pkl \
--emb_cache ../../vocab/word_embeddings_cache_lcq.pt \
--d_h_wo 512 \
--d_emb_wo 300 \
--rel_topk 50 \
--gpu $devices \
--st_pos 3000 \
--ed_pos 6000 \
--cpt ./runs/lcq/1624799696/checkpoints/best_snapshot_epoch_10_best_val_recall_87.80000000000003_model.pt

CUDA_LAUNCH_BLOCKING=1 python -u predict.py \
--dataset lcq \
--test_data ../../data/LC-QuAD/parsed_dev.json \
--output ../../data/LC-QuAD/candidate_relations_dev \
--rel_pool ../../data/LC-QuAD/relation_pool.json \
--wo_vocab ../../vocab/word_vocab_lcq.pkl \
--emb_cache ../../vocab/word_embeddings_cache_lcq.pt \
--d_h_wo 512 \
--d_emb_wo 300 \
--rel_topk 50 \
--gpu $devices \
--st_pos 0 \
--ed_pos 3000 \
--cpt ./runs/lcq/1624799696/checkpoints/best_snapshot_epoch_10_best_val_recall_87.80000000000003_model.pt

CUDA_LAUNCH_BLOCKING=1 python -u predict.py \
--dataset lcq \
--test_data ../../data/LC-QuAD/parsed_test.json \
--output ../../data/LC-QuAD/candidate_relations_test \
--rel_pool ../../data/LC-QuAD/relation_pool.json \
--wo_vocab ../../vocab/word_vocab_lcq.pkl \
--emb_cache ../../vocab/word_embeddings_cache_lcq.pt \
--d_h_wo 512 \
--d_emb_wo 300 \
--rel_topk 50 \
--gpu $devices \
--st_pos 0 \
--ed_pos 3000 \
--cpt ./runs/lcq/1624799696/checkpoints/best_snapshot_epoch_10_best_val_recall_87.80000000000003_model.pt