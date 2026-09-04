import os
import torch.distributed as dist
import torch
import random
import numpy as np
import torch.distributed as dist
import transformers

def ddp_setup(**pgkwargs):
   """
   Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
   """
   local_rank = int(os.environ["LOCAL_RANK"])
   global_rank = int(os.environ["RANK"])
   world_size = int(os.environ['WORLD_SIZE'])
   device_type = get_device_type()
   seed_offset = global_rank
   torch.cuda.set_device(local_rank) # set default GPU to local rank
   
   #In theory, rank and world_size can be inferred from env variables, but better to be explicit.
   dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size, init_method = "env://", **pgkwargs)
   return device_type, local_rank, global_rank, world_size, seed_offset

def single_gpu_setup():
    local_rank = 0
    global_rank = 0
    world_size = 1
    device_type = get_device_type()
    seed_offset = 0
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
    return device_type, local_rank, global_rank, world_size, seed_offset

def environment_setup(seed:int, use_seed_offset:bool = False,**pgkwargs):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        device_type, local_rank, global_rank, world_size, seed_offset = ddp_setup(**pgkwargs)
    else:
        device_type, local_rank, global_rank, world_size, seed_offset = single_gpu_setup()
    seed_everything(seed + seed_offset * int(use_seed_offset))
    return device_type, local_rank, global_rank, world_size

def cleanup():
    if dist.is_initialized():
        print("Cleaning up distributed environment")
        dist.destroy_process_group()
    
def get_world_size():
    # 
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def get_model(model):
    if isinstance(model, torch.nn.DataParallel) \
      or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model

def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def is_master():
    return get_rank() == 0

def get_device_type():
    return "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    transformers.set_seed(seed)



def get_remaining_dataloader(data_loader):
    """
    Given a DataLoader with a DistributedSampler that has drop_last=True,
    this function returns a new DataLoader that contains the remaining samples
    that were not included in the original DataLoader due to dropping the last
    incomplete batch.

    This is useful when it's critical to ensure all samples in the dataset are processed.

    Args:
        data_loader (torch.utils.data.DataLoader): The original DataLoader.

    Returns:
        torch.utils.data.DataLoader: A new DataLoader with the remaining samples.
    """
    # Check if distributed environment is running
    if (dist.is_initialized() and data_loader.sampler is not None):
        assert hasattr(data_loader.sampler, "total_size") and hasattr(data_loader.sampler, "dataset"), "Sampler must have 'total_size' and 'dataset' attributes."

        # Only create a new DataLoader if there are remaining samples that original dataloader didn't cover
        if data_loader.sampler.total_size < len(data_loader.sampler.dataset):
            
            # remaining_indices = set(range(len(data_loader.sampler.dataset))) - set(list(iter(data_loader.sampler.indices)))
            
            remaining_dataset = torch.utils.data.Subset(data_loader.dataset, range(data_loader.sampler.total_size, len(data_loader.sampler.dataset)))
            remaining_dataloader = torch.utils.data.DataLoader(
                remaining_dataset,
                shuffle=False,
                drop_last=False,
                sampler=None,
                batch_sampler=None,
                batch_size=data_loader.batch_size,
                num_workers=data_loader.num_workers,
                pin_memory=data_loader.pin_memory,
                collate_fn=data_loader.collate_fn,
                timeout =data_loader.timeout,
                worker_init_fn=data_loader.worker_init_fn,
                prefetch_factor=data_loader.prefetch_factor,
                persistent_workers=data_loader.persistent_workers
            )
        
            return remaining_dataloader
 