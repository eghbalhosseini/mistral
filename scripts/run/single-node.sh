# Single-Node, Single GPU, No GC, FP16, Device BSZ = 8
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/gpt2-benchmark-config.yaml --nnodes 1 --nproc_per_node 1 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 8

python train.py --config conf/miniBERTa-1M-gpt2-micro.yaml --nnodes 1 --nproc_per_node 1 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 8


CUDA_VISIBLE_DEVICES=0 python train.py --config conf/OM_miniBERTa-1M-gpt2-micro.yaml --nnodes 1 --nproc_per_node 1 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 8