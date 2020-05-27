#!/bin/bash
# bash ./scripts-search/GDAS-search-NASNet-space.sh cifar10 1 -1
echo script name: $0
echo $# arguments
if [ "$#" -ne 4 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for dataset, track_running_stats, and seed"
  exit 1
fi
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

dataset=$1
track_running_stats=$2
seed=$3
superweight=$4
space=darts

if [ "$dataset" == "cifar10" ] || [ "$dataset" == "cifar100" ]; then
  data_path="$TORCH_HOME/cifar.python"
else
  data_path="$TORCH_HOME/cifar.python/ImageNet16"
fi

save_dir=./output/search-cell-${space}/SPOS-${dataset}-BN${track_running_stats}


if [ "$superweight" == "None" ] || [ "$superweight" == "none" ]; then
  OMP_NUM_THREADS=4 python ./exps/algos/SPOS.py \
    --save_dir ${save_dir} \
    --dataset ${dataset} --data_path ${data_path} \
    --search_space_name ${space} \
    --config_path  configs/search-opts/SPOS-NASNet-CIFAR.config \
    --model_config configs/search-archs/SPOS-NASNet-CIFAR.config \
    --track_running_stats ${track_running_stats} \
    --workers 4 --print_freq 200 --rand_seed ${seed}
else
  OMP_NUM_THREADS=4 python ./exps/algos/SPOS.py \
    --save_dir ${save_dir} \
    --dataset ${dataset} --data_path ${data_path} \
    --search_space_name ${space} \
    --config_path  configs/search-opts/SPOS-NASNet-CIFAR.config \
    --model_config configs/search-archs/SPOS-NASNet-CIFAR.config \
    --track_running_stats ${track_running_stats} \
    --workers 4 --print_freq 200 --rand_seed ${seed} \
    --supernet_path ${superweight}
fi

