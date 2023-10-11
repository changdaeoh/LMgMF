cd ..

DP='full'
prompt_type=1

for METHOD in 'lmlr_i'
do
for SEED in 1 2 3
do
for GRADE in entire
do
for LR in 0.001
do

python3 train.py \
--gpu 7 \
--seed $SEED \
--lr $LR \
--grade "entire" \
--method $METHOD \
--data_prop $DP \
--dataset 'benchmark' \
--data_root '/data2/changdae/lmgmf/data/assist2009/rating_dataset/' \
--iemb_path /data2/changdae/lmgmf/llama2/assist2009_q_prompt_${prompt_type}.pkl_vanilla.npy \
--prompt_type $prompt_type \
--epochs 100 \
--wb_run_name 'lmlr_i' \
--wb_name 'lmgmf-assist-final'

done
done
done
done