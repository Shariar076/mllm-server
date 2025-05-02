import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
<<<<<<< HEAD
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['NCCL_SOCKET_IFNAME'] = 'ens10f0'

    os.environ['MASTER_ADDR'] = '169.235.27.224'
=======
    os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'

    os.environ['MASTER_ADDR'] = '127.0.0.1'
>>>>>>> 9c02206 (update-mlnlp)
    os.environ['MASTER_PORT'] = '29500'
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def demo_basic(rank, world_size):
    setup(rank, world_size)
    
    # Create a tensor on each GPU
    tensor = torch.ones(1).cuda(rank) * (rank + 1)
    print(f"[Before all_reduce] Rank {rank}, tensor: {tensor}")

    # Sum all tensors across GPUs
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"[After all_reduce] Rank {rank}, tensor: {tensor}")
    
    cleanup()

def run_demo(world_size):
    mp.spawn(demo_basic,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Need at least 2 GPUs to test torch.distributed with NCCL")
    else:
        run_demo(world_size)

