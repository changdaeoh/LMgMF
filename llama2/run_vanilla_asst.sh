#!bin/bash

CUDA_VISIBLE_DEVICES=5,6 python inference.py \
--model_name 'meta-llama/Llama-2-7b-hf' \
--p_type 1 \
--prompt_file '/data2/changdae/lmgmf/data/assist2009/q_prompt_1.pkl'

CUDA_VISIBLE_DEVICES=5,6 python inference.py \
--model_name 'meta-llama/Llama-2-7b-hf' \
--p_type 2 \
--prompt_file '/data2/changdae/lmgmf/data/assist2009/q_prompt_2.pkl'