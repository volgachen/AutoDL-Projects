#!/bin/bash
# bash ./scripts-search/GDAS-search-NASNet-space.sh cifar10 1 -1
echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
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
space=darts

if [ "$dataset" == "cifar10" ] || [ "$dataset" == "cifar100" ]; then
  data_path="$TORCH_HOME/cifar.python"
else
  data_path="$TORCH_HOME/cifar.python/ImageNet16"
fi

save_dir=./output/search-cell-${space}/ENAS-${dataset}-BN${track_running_stats}

OMP_NUM_THREADS=4 python ./exps/algos/ENAS.py \
	--save_dir ${save_dir} \
	--dataset ${dataset} --data_path ${data_path} \
	--search_space_name ${space} \
	--config_path  configs/search-opts/ENAS-NASNet-CIFAR.config \
	--model_config configs/search-archs/ENAS-NASNet-CIFAR.config \
	--controller_entropy_weight 0.0001 \
	--controller_bl_dec 0.99 \
	--controller_train_steps 300 \
	--controller_num_aggregate 1 \
	--controller_num_samples 100 \
	--workers 4 --print_freq 200 --rand_seed ${seed}
