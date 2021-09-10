
CHECKPOINT=""
LR=1e-5
EPOCH=15
BS=16
YEAR=2014
WARM=100
name="cmed_roberta-entity-cmlm-luke2"
mkdir outputs
mkdir outputs/${name}

checkpoints="roberta-entity-cmlm-luke_8_460000.ckpt"
iterations="460000"
array_checkpoints=($checkpoints)
array_iterations=($iterations)
seed=4
count=${#array_checkpoints[@]}
for i in `seq 1 $count`
do
     ckpt=${array_checkpoints[$i-1]} 
     iter=${array_iterations[$i-1]}
     mkdir outputs/${name}/${iter}
     mkdir outputs/${name}/${iter}/${YEAR}_30_${seed}
     python -m cli.ed_finetune \
          --weight_path ${name}/${ckpt} \
          --config_name ${name}/config \
          --epochs ${EPOCH} \
          --train_bs ${BS} \
          --year ${YEAR}  \
          --seed ${seed} \
          --warmup_step ${WARM} \
          --name ${LR}_${WARM}_${YEAR}_${BS}-${EPOCH}-${seed} \
          --lr ${LR} \
          --output_path outputs/${name}/${iter}/${YEAR}_30_${seed}

done


seed=3
count=${#array_checkpoints[@]}
for i in `seq 1 $count`
do
     ckpt=${array_checkpoints[$i-1]} 
     iter=${array_iterations[$i-1]}
     mkdir outputs/${name}/${iter}
     mkdir outputs/${name}/${iter}/${YEAR}_30_${seed}
     python -m cli.ed_finetune \
          --weight_path ${name}/${ckpt} \
          --config_name ${name}/config \
          --epochs ${EPOCH} \
          --train_bs ${BS} \
          --year ${YEAR}  \
          --seed ${seed} \
          --warmup_step ${WARM} \
          --name ${LR}_${WARM}_${YEAR}_${BS}-${EPOCH}-${seed} \
          --lr ${LR} \
          --output_path outputs/${name}/${iter}/${YEAR}_30_${seed}
done



seed=2
count=${#array_checkpoints[@]}
for i in `seq 1 $count`
do
     ckpt=${array_checkpoints[$i-1]} 
     iter=${array_iterations[$i-1]}
     mkdir outputs/${name}/${iter}
     mkdir outputs/${name}/${iter}/${YEAR}_30_${seed}
     python -m cli.ed_finetune \
          --weight_path ${name}/${ckpt} \
          --config_name ${name}/config \
          --epochs ${EPOCH} \
          --train_bs ${BS} \
          --year ${YEAR}  \
          --seed ${seed} \
          --warmup_step ${WARM} \
          --name ${LR}_${WARM}_${YEAR}_${BS}-${EPOCH}-${seed} \
          --lr ${LR} \
          --output_path outputs/${name}/${iter}/${YEAR}_30_${seed}

done



seed=1
count=${#array_checkpoints[@]}
for i in `seq 1 $count`
do
     ckpt=${array_checkpoints[$i-1]} 
     iter=${array_iterations[$i-1]}
     mkdir outputs/${name}/${iter}
     mkdir outputs/${name}/${iter}/${YEAR}_30_${seed}
     python -m cli.ed_finetune \
          --weight_path ${name}/${ckpt} \
          --config_name ${name}/config \
          --epochs ${EPOCH} \
          --train_bs ${BS} \
          --year ${YEAR}  \
          --seed ${seed} \
          --warmup_step ${WARM} \
          --name ${LR}_${WARM}_${YEAR}_${BS}-${EPOCH}-${seed} \
          --lr ${LR} \
          --output_path outputs/${name}/${iter}/${YEAR}_30_${seed}

done


