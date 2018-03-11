#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process


def run(rank, size):
    """Blocking point to point communication."""
    tensor = torch.zeros(1)
    if rank == 0:
        # The master node is going to send the tensor with data `1` to node 1.
        tensor += 1
        dist.send(tensor=tensor, dst=1)
    else:
        # The other nodes will receive the tensor with data `1`.
        dist.recv(tensor=tensor, src=0)
    print("RANK ", rank, " has data ", tensor[0])


def init_processes(rank, size, fn, backend="tcp"):
    """Initialize the distributed environment.

    Pytorch supports several styles of initialization. The options are
    described here:
    http://pytorch.org/docs/master/distributed.html#initialization
    We're opting for the default environment variable initialization.
    MASTER_ADDR specifies the address of the rank 0 (master) node.
    MASTER_PORT specifies a free port on the master node.
    We could also specify WORLD_SIZE via environment variable, but we opt
    to specify it via the call to the int function.
    Similarly, we specify RANK in the call even though we could use an
    environment variable.
    
    NOTE: We use tcp as the default backend because we want send and recv
    functionality. This low level control comes at the cost of not having
    GPU compatibility. When we use DistributedDataParallel with a real model, 
    we don't need such precise control and can use the gloo backend. You
    can find a table listing the capabilities of each backend at
    http://pytorch.org/docs/master/distributed.html#module-torch.distributed
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    # We're going to emulate having multiple machines by using processes
    # and having them communicate in a distributed fashion.
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
