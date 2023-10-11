GRADE='entire'
DP='full'
METHOD='lminit_i'

for SEED in 1 2 3
do
for DIM in 4096
do
for LR in 0.05
do
for REG_B in 1e-6
do
for REG_F in 1e-5
do
for DU in 0.0
do
for DI in 0.0
do

python3 train.py \
--gpu 1 \
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