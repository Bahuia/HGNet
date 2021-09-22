#!/bin/bash

devices=3

CUDA_LAUNCH_BLOCKING=1 python -u ../eval_non_hierarchical.py \
--dataset cwq \
--seed 2021 \
--test_data ../../data/ComplexWebQuestions/annotated_dev.pkl \
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
--cpt ./runs/cwq/1626354925/checkpoints/best_snapshot_epoch_26_val_aqg_acc_65.83548294640298_val_acc_40.985955861278306_model.pt \
--result_name result_dev.pkl \
--kb_endpoint http://10.201.69.194:8890//sparql