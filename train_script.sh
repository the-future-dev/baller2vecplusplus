#!/usr/bin/env bash

#JOB=$(date +%Y%m%d%H%M%S)
JOB="20230927175152"

echo "train:" >> ${JOB}.yaml
task=basketball  # "basketball" or "toy".
echo "  task: ${task}" >> ${JOB}.yaml
if [[ "$task" = "basketball" ]]
then

    echo "  train_samples_per_epoch: 20000" >> ${JOB}.yaml
    echo "  valid_samples: 1000" >> ${JOB}.yaml
    echo "  workers: 8" >> ${JOB}.yaml
    echo "  learning_rate: 1.0e-5" >> ${JOB}.yaml
    echo "  patience: 20" >> ${JOB}.yaml

    echo "dataset:" >> ${JOB}.yaml
    echo "  hz: 5" >> ${JOB}.yaml
    echo "  secs: 4.2" >> ${JOB}.yaml
    echo "  player_traj_n: 11" >> ${JOB}.yaml
    echo "  max_player_move: 4.5" >> ${JOB}.yaml

    echo "model:" >> ${JOB}.yaml
    echo "  embedding_dim: 20" >> ${JOB}.yaml
    echo "  sigmoid: none" >> ${JOB}.yaml
    echo "  mlp_layers: [128, 256, 512]" >> ${JOB}.yaml
    echo "  nhead: 8" >> ${JOB}.yaml
    echo "  dim_feedforward: 2048" >> ${JOB}.yaml
    echo "  num_layers: 6" >> ${JOB}.yaml
    echo "  dropout: 0.0" >> ${JOB}.yaml
    echo "  b2v: False" >> ${JOB}.yaml

else

    echo "  workers: 8" >> ${JOB}.yaml
    echo "  learning_rate: 1.0e-4" >> ${JOB}.yaml

    echo "model:" >> ${JOB}.yaml
    echo "  embedding_dim: 20" >> ${JOB}.yaml
    echo "  sigmoid: none" >> ${JOB}.yaml
    echo "  mlp_layers: [64, 128]" >> ${JOB}.yaml
    echo "  nhead: 4" >> ${JOB}.yaml
    echo "  dim_feedforward: 512" >> ${JOB}.yaml
    echo "  num_layers: 2" >> ${JOB}.yaml
    echo "  dropout: 0.0" >> ${JOB}.yaml
    echo "  b2v: True" >> ${JOB}.yaml

fi

# Save experiment settings.
mkdir -p ${EXPERIMENTS_DIR}/${JOB}
mv ${JOB}.yaml ${EXPERIMENTS_DIR}/${JOB}/

gpu=0
cd ${PROJECT_DIR}
nohup python3 train_baller2vecplusplus.py ${JOB} ${gpu} > ${EXPERIMENTS_DIR}/${JOB}/train.log &