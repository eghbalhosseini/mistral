#!/bin/bash
#SBATCH --job-name=mini
#SBATCH --array=0
#SBATCH --time=2-12:00:00
#SBATCH --gres=gpu:RTXA6000:4
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

python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 train.py --config conf/tutorial-gpt2-micro.yaml --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 64 --run_id minberta_1m