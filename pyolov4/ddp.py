import torch.cuda
from torch import distributed as dist


def init_ddp(local_rank: int, total_batch_size: int):
    """Initializing distributed learning

    Args:
        local_rank: local rank of process
        total_batch_size: total batch size for training

    Returns:

    """
    assert torch.cuda.device_count() > local_rank, "Number of physical GPUs less than the local rank"
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    assert total_batch_size % world_size == 0, 'total_batch_size must be multiple of CUDA device count'
    batch_size = total_batch_size // world_size
    return device, world_size, global_rank, batch_size
