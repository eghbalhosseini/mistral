#!/bin/bash
#SBATCH --job-name=mini
#SBATCH --array=0
#SBATCH --time=8:00:00
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

deepspeed --num_gpus 4 --num_nodes 1 --master_addr node094 train.py --config conf/tutorial-gpt2-micro.yaml --nnodes 1 --nproc_per_node 4 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 4 --training_arguments.deepspeed conf/deepspeed/z3-conf.json \
 --run_id tutorial-gpt2-micro-multi-node > tutorial-gpt2-micro-multi-node.out 2> tutorial-gpt2-micro-multi-node.err --hostfile conf/deepspeed/hostfile_node094