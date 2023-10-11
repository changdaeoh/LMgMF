GRADE='entire'
DP='full'
METHOD='lmdistill_ui'

for SEED in 1 2 3
do
for DIM in 10
do
for LR in 0.05
do
for REG_B in 1e-5
do
for REG_F in 1e-5
do
for DU in 1e-4
do
for DI in 1e-3
do

python3 train.py \
--gpu 0 \
--seed $SEED \
--lr $LR \
--ld $DIM \
--reg_b $REG_B \
--reg_f $REG_F \
--grade $GRADE \
--method $METHOD \
--data_prop $DP \
--dist_u $DU \
--dist_i $DI \
--epochs 100 \
--wb_name 'lmguided'

done
done
done
done
done
done
done