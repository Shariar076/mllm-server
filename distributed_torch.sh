<<<<<<< HEAD
# export MASTER_ADDR=169.235.27.224  # e.g. 169.235.26.162
# export NCCL_SOCKET_IFNAME=ens10f0
# export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=INFO
# export MASTER_PORT=29500
# export WORLD_SIZE=2
# export RANK=0
# export CUDA_VISIBLE_DEVICES=0,1
# python distributed_torch.py

torchrun \
  --nproc_per_node=1 \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr=169.235.27.224 \
  --master_port=29500 \
  distributed_torch.py
=======
export NCCL_SOCKET_IFNAME=eno1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export MASTER_ADDR=169.235.26.162  # e.g. 169.235.26.162
export MASTER_PORT=29500
export WORLD_SIZE=2
export RANK=0
export CUDA_VISIBLE_DEVICES=1,2
python distributed_torch.py
>>>>>>> 9c02206 (update-mlnlp)
