#!/bin/bash

CUDA_LAUNCH_BLOCKING=1 python -u ../eval_seq2seq.py \
--dataset lcq \
--test_data ../../data/LC-QuAD/parsed_test.json \
--sparql_path ./runs/bart_cpt/cpt0/generated_predictions.txt \
--results_path ./runs/bart_cpt/cpt0/results.pkl \
--kb_endpoint http://10.201.61.163:8890//sparql