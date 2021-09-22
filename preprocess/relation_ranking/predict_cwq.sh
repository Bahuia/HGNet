#!/bin/bash

devices=3

CUDA_LAUNCH_BLOCKING=1 python -u predict.py \
--dataset cwq \
--test_data ../../data/ComplexWebQuestions/parsed_train.json \
--output ../../data/ComplexWebQuestions/candidate_relations_train \
--rel_pool ../../data/ComplexWebQuestions/relation_pool.json \
--wo_vocab ../../vocab/word_vocab_cwq.pkl \
--emb_cache ../../vocab/word_embeddings_cache_cwq.pt \
--d_h_wo 512 \
--d_emb_wo 300 \
--rel_topk 50 \
--gpu $devices \
--st_pos 0 \
--ed_pos 3000 \
--cpt ./runs/cwq/1624801913/checkpoints/best_snapshot_epoch_6_best_val_recall_95.52808563038737_model.pt

CUDA_LAUNCH_BLOCKING=1 python -u predict.py \
--dataset cwq \
--test_data ../../data/ComplexWebQuestions/parsed_train.json \
--output ../../data/ComplexWebQuestions/candidate_relations_train \
--rel_pool ../../data/ComplexWebQuestions/relation_pool.json \
--wo_vocab ../../vocab/word_vocab_cwq.pkl \
--emb_cache ../../vocab/word_embeddings_cache_cwq.pt \
--d_h_wo 512 \
--d_emb_wo 300 \
--rel_topk 50 \
--gpu $devices \
--st_pos 3000 \
--ed_pos 6000 \
--cpt ./runs/cwq/1624801913/checkpoints/best_snapshot_epoch_6_best_val_recall_95.52808563038737_model.pt

CUDA_LAUNCH_BLOCKING=1 python -u predict.py \
--dataset cwq \
--test_data ../../data/ComplexWebQuestions/parsed_train.json \
--output ../../data/ComplexWebQuestions/candidate_relations_train \
--rel_pool ../../data/ComplexWebQuestions/relation_pool.json \
--wo_vocab ../../vocab/word_vocab_cwq.pkl \
--emb_cache ../../vocab/word_embeddings_cache_cwq.pt \
--d_h_wo 512 \
--d_emb_wo 300 \
--rel_topk 50 \
--gpu $devices \
--st_pos 6000 \
--ed_pos 9000 \
--cpt ./runs/cwq/1624801913/checkpoints/best_snapshot_epoch_6_best_val_recall_95.52808563038737_model.pt

CUDA_LAUNCH_BLOCKING=1 python -u predict.py \
--dataset cwq \
--test_data ../../data/ComplexWebQuestions/parsed_train.json \
--output ../../data/ComplexWebQuestions/candidate_relations_train \
--rel_pool ../../data/ComplexWebQuestions/relation_pool.json \
--wo_vocab ../../vocab/word_vocab_cwq.pkl \
--emb_cache ../../vocab/word_embeddings_cache_cwq.pt \
--d_h_wo 512 \
--d_emb_wo 300 \
--rel_topk 50 \
--gpu $devices \
--st_pos 9000 \
--ed_pos 12000 \
--cpt ./runs/cwq/1624801913/checkpoints/best_snapshot_epoch_6_best_val_recall_95.52808563038737_model.pt

CUDA_LAUNCH_BLOCKING=1 python -u predict.py \
--dataset cwq \
--test_data ../../data/ComplexWebQuestions/parsed_train.json \
--output ../../data/ComplexWebQuestions/candidate_relations_train \
--rel_pool ../../data/ComplexWebQuestions/relation_pool.json \
--wo_vocab ../../vocab/word_vocab_cwq.pkl \
--emb_cache ../../vocab/word_embeddings_cache_cwq.pt \
--d_h_wo 512 \
--d_emb_wo 300 \
--rel_topk 50 \
--gpu $devices \
--st_pos 12000 \
--ed_pos 15000 \
--cpt ./runs/cwq/1624801913/checkpoints/best_snapshot_epoch_6_best_val_recall_95.52808563038737_model.pt

CUDA_LAUNCH_BLOCKING=1 python -u predict.py \
--dataset cwq \
--test_data ../../data/ComplexWebQuestions/parsed_train.json \
--output ../../data/ComplexWebQuestions/candidate_relations_train \
--rel_pool ../../data/ComplexWebQuestions/relation_pool.json \
--wo_vocab ../../vocab/word_vocab_cwq.pkl \
--emb_cache ../../vocab/word_embeddings_cache_cwq.pt \
--d_h_wo 512 \
--d_emb_wo 300 \
--rel_topk 50 \
--gpu $devices \
--st_pos 15000 \
--ed_pos 18000 \
--cpt ./runs/cwq/1624801913/checkpoints/best_snapshot_epoch_6_best_val_recall_95.52808563038737_model.pt

CUDA_LAUNCH_BLOCKING=1 python -u predict.py \
--dataset cwq \
--test_data ../../data/ComplexWebQuestions/parsed_train.json \
--output ../../data/ComplexWebQuestions/candidate_relations_train \
--rel_pool ../../data/ComplexWebQuestions/relation_pool.json \
--wo_vocab ../../vocab/word_vocab_cwq.pkl \
--emb_cache ../../vocab/word_embeddings_cache_cwq.pt \
--d_h_wo 512 \
--d_emb_wo 300 \
--rel_topk 50 \
--gpu $devices \
--st_pos 18000 \
--ed_pos 21000 \
--cpt ./runs/cwq/1624801913/checkpoints/best_snapshot_epoch_6_best_val_recall_95.52808563038737_model.pt

CUDA_LAUNCH_BLOCKING=1 python -u predict.py \
--dataset cwq \
--test_data ../../data/ComplexWebQuestions/parsed_train.json \
--output ../../data/ComplexWebQuestions/candidate_relations_train \
--rel_pool ../../data/ComplexWebQuestions/relation_pool.json \
--wo_vocab ../../vocab/word_vocab_cwq.pkl \
--emb_cache ../../vocab/word_embeddings_cache_cwq.pt \
--d_h_wo 512 \
--d_emb_wo 300 \
--rel_topk 50 \
--gpu $devices \
--st_pos 21000 \
--ed_pos 24000 \
--cpt ./runs/cwq/1624801913/checkpoints/best_snapshot_epoch_6_best_val_recall_95.52808563038737_model.pt

CUDA_LAUNCH_BLOCKING=1 python -u predict.py \
--dataset cwq \
--test_data ../../data/ComplexWebQuestions/parsed_train.json \
--output ../../data/ComplexWebQuestions/candidate_relations_train \
--rel_pool ../../data/ComplexWebQuestions/relation_pool.json \
--wo_vocab ../../vocab/word_vocab_cwq.pkl \
--emb_cache ../../vocab/word_embeddings_cache_cwq.pt \
--d_h_wo 512 \
--d_emb_wo 300 \
--rel_topk 50 \
--gpu $devices \
--st_pos 24000 \
--ed_pos 27000 \
--cpt ./runs/cwq/1624801913/checkpoints/best_snapshot_epoch_6_best_val_recall_95.52808563038737_model.pt

CUDA_LAUNCH_BLOCKING=1 python -u predict.py \
--dataset cwq \
--test_data ../../data/ComplexWebQuestions/parsed_train.json \
--output ../../data/ComplexWebQuestions/candidate_relations_train \
--rel_pool ../../data/ComplexWebQuestions/relation_pool.json \
--wo_vocab ../../vocab/word_vocab_cwq.pkl \
--emb_cache ../../vocab/word_embeddings_cache_cwq.pt \
--d_h_wo 512 \
--d_emb_wo 300 \
--rel_topk 50 \
--gpu $devices \
--st_pos 27000 \
--ed_pos 30000 \
--cpt ./runs/cwq/1624801913/checkpoints/best_snapshot_epoch_6_best_val_recall_95.52808563038737_model.pt

CUDA_LAUNCH_BLOCKING=1 python -u predict.py \
--dataset cwq \
--test_data ../../data/ComplexWebQuestions/parsed_dev.json \
--output ../../data/ComplexWebQuestions/candidate_relations_dev \
--rel_pool ../../data/ComplexWebQuestions/relation_pool.json \
--wo_vocab ../../vocab/word_vocab_cwq.pkl \
--emb_cache ../../vocab/word_embeddings_cache_cwq.pt \
--d_h_wo 512 \
--d_emb_wo 300 \
--rel_topk 50 \
--gpu $devices \
--st_pos 0 \
--ed_pos 4000 \
--cpt ./runs/cwq/1624801913/checkpoints/best_snapshot_epoch_6_best_val_recall_95.52808563038737_model.pt

CUDA_LAUNCH_BLOCKING=1 python -u predict.py \
--dataset cwq \
--test_data ../../data/ComplexWebQuestions/parsed_test.json \
--output ../../data/ComplexWebQuestions/candidate_relations_test \
--rel_pool ../../data/ComplexWebQuestions/relation_pool.json \
--wo_vocab ../../vocab/word_vocab_cwq.pkl \
--emb_cache ../../vocab/word_embeddings_cache_cwq.pt \
--d_h_wo 512 \
--d_emb_wo 300 \
--rel_topk 50 \
--gpu $devices \
--st_pos 0 \
--ed_pos 4000 \
--cpt ./runs/cwq/1624801913/checkpoints/best_snapshot_epoch_6_best_val_recall_95.52808563038737_model.pt