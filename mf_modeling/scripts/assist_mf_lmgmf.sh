cd ..

DP='full'
METHOD='lmdistill_i'
prompt_type=1

for SEED in 1 2 3
do
for GRADE in entire
do
for DIM in 10
do
for LR in 0.005
do
for REG_B in 1e-5
do
for REG_F in 1e-3
do
for DU in 0.0
do
for DI in 0.00005
do

python3 train.py \
--gpu 4 \
--seed $SEED \
--lr $LR \
--ld $DIM \
--reg_b $REG_B \
--reg_f $REG_F \
--grade "entire" \
--method $METHOD \
--data_prop $DP \
--dataset 'benchmark' \
--data_root '/data2/changdae/lmgmf/data/assist2009/rating_dataset/' \
--iemb_path /data2/changdae/lmgmf/llama2/assist2009_q_prompt_${prompt_type}.pkl_vanilla.npy \
--prompt_type $prompt_type \
--dist_u $DU \
--dist_i $DI \
--epochs 100 \
--wb_run_name 'assist-lmgmf-best' \
--wb_name 'lmgmf-ablation'
done
done
done
done
done
done
done
done



DP='full'
METHOD='vanilla'

for SEED in 1 2 3
do
for GRADE in entire
do
for DIM in 10
do
for LR in 0.005
do
for REG_B in 1e-5
do
for REG_F in 1e-4
do
for DU in 0.0
do
for DI in 0.0
do

python3 train.py \
--gpu 4 \
--seed $SEED \
--lr $LR \
--ld $DIM \
--reg_b $REG_B \
--reg_f $REG_F \
--grade "entire" \
--method $METHOD \
--data_prop $DP \
--dataset 'benchmark' \
--data_root '/data2/changdae/lmgmf/data/assist2009/rating_dataset/' \
--iemb_path '/data2/changdae/lmgmf/llama2/assist2009_q_prompt_1.pkl_vanilla.npy' \
--dist_u $DU \
--dist_i $DI \
--epochs 100 \
--wb_run_name 'assist-mf-best' \
--wb_name 'lmgmf-ablation'
done
done
done
done
done
done
done
done