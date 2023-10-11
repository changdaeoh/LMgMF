#!/bin/bash

cd ..

DP='full'
METHOD='dins-ngcf'
GRADE='entire'

for SEED in 1 2 3
do
for pool in 'concat'
do
for DIM in 64
do
for alpha in 1.0 
do
for l2 in 1e-4
do
for negs in 64
do
for LR in 0.0001
do

python3 main_dins.py \
--gpu 4 \
--seed $SEED \
--lr $LR \
--l2 $l2 \
--ld $DIM \
--grade "entire" \
--method $METHOD \
--data_prop $DP \
--dataset 'benchmark' \
--data_root '/data2/changdae/lmgmf/data/assist2009/rating_dataset/full/' \
--epochs 100 \
--bs 1024 \
--pool $pool \
--ns 'dins' \
--alpha $alpha \
--n_negs $negs \
--wb_run_name $METHOD \
--wb_name 'lmgmf-assist-final'

done
done
done
done
done
done
done