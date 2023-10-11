#!/bin/bash

cd ..

DP='full'
METHOD='ncf'
GRADE='entire'

for SEED in 1 2 3
do
for DIM in 10
do
for ncf_l in 2
do
for LR in 0.001
do

python3 train.py \
--gpu 1 \
--seed $SEED \
--lr $LR \
--ld $DIM \
--grade "entire" \
--method $METHOD \
--data_prop $DP \
--dataset 'benchmark' \
--data_root '/data2/changdae/lmgmf/data/assist2009/rating_dataset/' \
--epochs 100 \
--ncf_layer $ncf_l \
--wb_run_name 'ncf' \
--wb_name 'lmgmf-assist-final'

done
done
done
done