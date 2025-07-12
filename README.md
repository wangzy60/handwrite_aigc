### 单机多卡训练

torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=12345 train.py
