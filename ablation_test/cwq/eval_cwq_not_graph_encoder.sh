#!/bin/bash

devices=0

CUDA_LAUNCH_BLOCKING=1 python -u ../../eval.py \
--dataset cwq \
--seed 2021 \
--test_data ../../data/ComplexWebQuestions/annotated_test.pkl \
--wo_vocab ../../vocab/word_vocab_cwq.pkl \
--context_mode attention \
--not_matching_score \
--not_mention_feature \
--not_matching_feature \
--d_h 256 \
--d_emb 300 \
--d_f 32 \
--gpu $devices \
--n_lstm_layers 1 \
--n_gnn_blocks 3 \
--heads 4 \
--beam_size 5 \
--max_num_op 45 \
--sparql_cache_path ../../vocab/sparql_cache_cwq.pkl \
--cpt ./runs/cwq/1625587824/checkpoints/best_snapshot_epoch_37_val_aqg_acc_67.23989681857266_val_acc_43.250214961306966_model.pt \
--kb_endpoint http://10.201.7.66:8890//sparql \
--not_graph_encoder