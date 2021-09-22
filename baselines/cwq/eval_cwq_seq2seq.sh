#!/bin/bash

CUDA_LAUNCH_BLOCKING=1 python -u ../eval_seq2seq.py \
--dataset cwq \
--test_data ../../data/ComplexWebQuestions/parsed_test.json \
--rel_pool ../../data/ComplexWebQuestions/relation_pool.json \
--sparql_path ./runs/bart_cpt/cpt0/generated_predictions.txt \
--results_path ./runs/bart_cpt/cpt0/results.pkl \
--kb_endpoint http://10.201.69.194:8890//sparql