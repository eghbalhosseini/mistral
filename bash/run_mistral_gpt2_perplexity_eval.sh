#!/bin/bash
#SBATCH --job-name=mini
#SBATCH --array=0
#SBATCH --time=2-12:00:00
#SBATCH --ntasks=1
#SBATCH --mem=120G
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu
#SBATCH --partition=evlab

. ~/.bash_profile
conda activate mistral
echo $(which python)
cd /om/user/ehoseini/mistral/


#CUDA_VISIBLE_DEVICES=0 python train.py --config conf/gpt2-mistral-small-config.yaml --nnodes 1 --nproc_per_node 2 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 2 --model.initial_weights /om/user/ehoseini/MyData/mistral/caprica-gpt2-small-x81/ckpt_400000/pytorch_model.bin --run_training False

#python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 train.py --config conf/gpt2-mistral-small-config.yaml --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 2 --model.initial_weights /om/user/ehoseini/MyData/mistral/caprica-gpt2-small-x81/ckpt_400000/pytorch_model.bin --run_training False

CUDA_VISIBLE_DEVICES=0 python train.py --config conf/gpt2-mistral-small-config.yaml --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 2 --model.initial_weights /om/user/ehoseini/MyData/mistral/caprica-gpt2-small-x81/ckpt_400000/pytorch_model.bin --run_training False