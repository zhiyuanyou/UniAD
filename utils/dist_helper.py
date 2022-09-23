import os
import subprocess

import torch
import torch.distributed as dist


def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()
    rank = get_rank()
    world_size = get_world_size()
    addr = "localhost"
    if "SLURM_JOB_ID" in os.environ:
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
    # specify master port
    if port is not None:
        os.environ["MASTER_PORT"] = str(port)
    elif "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = addr

    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank % num_gpus)
    os.environ["RANK"] = str(rank)

    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        init_method="env://",
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return rank, world_size


def get_rank():
    """Replace linklink.get_rank"""
    rank_cands = ["SLURM_PROCID", "MV2_COMM_WORLD_RANK", "PMI_RANK"]
    for rank_name in rank_cands:
        if rank_name in os.environ:
            return int(os.environ[rank_name])
    return int(os.environ.get("SLURM_PROCID", 0))


def get_world_size():
    """Replace linklink.get_world_size"""
    ws_cands = ["SLURM_NTASKS", "MV2_COMM_WORLD_SIZE", "PMI_SIZE"]
    for ws_name in ws_cands:
        if ws_name in os.environ:
            return int(os.environ[ws_name])
    return int(os.environ.get("SLURM_NTASKS", 1))
