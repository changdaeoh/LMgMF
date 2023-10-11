#!/bin/bash

cd ..

DP='full'
METHOD='vfm'
GRADE='entire'

for SEED in 1 2 3
do
for klc in 0.5
do
for DIM in 20
do
for LR in 0.01
do

python3 train.py \
--gpu 7 \
--seed $SEED \
--lr $LR \
--ld $DIM \
--grade "entire" \
--method $METHOD \
--data_prop $DP \
--dataset 'benchmark' \
--data_root '/data2/changdae/lmgmf/data/assist2009/rating_dataset/' \
--epochs 100 \
--kl_coef $klc \
--wb_run_name 'vfm' \
--wb_name 'lmgmf-assist-final'
done
done
done
done
