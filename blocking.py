#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process


# This is the IP of the master node - my laptop in this case.
NODE_0_IP = "192.168.1.30"


def run(rank, size):
    """Non-Blocking point to point communication."""
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        # The master node is going to send the tensor with data `1` to node 1.
        tensor += 1
        req = dist.isend(tensor=tensor, dst=1)
        print("Rank 0 started sending")
    else:
        # The other nodes will receive the tensor with data `1`.
        req = dist.irecv(tensor=tensor, src=0)
        print("Rank 1 started receiving")
    req.wait()  # Guarantee that the communication took place
    print("RANK ", rank, " has data ", tensor[0])


def init_processes(size, fn, backend="tcp"):
    """Initialize the distributed environment.

    Pytorch supports several styles of initialization. The options are
    described here:
    http://pytorch.org/docs/master/distributed.html#initialization
    We're opting for the default environment variable initialization.
    MASTER_ADDR specifies the address of the rank 0 (master) node.
    MASTER_PORT specifies a free port on the master node.
    We could also specify WORLD_SIZE via environment variable, but we opt
    to specify it via the call to the int function.
    We receive RANK from an environment variable.

    NOTE: We use tcp as the default backend because we want send and recv
    functionality. This low level control comes at the cost of not having
    GPU compatibility. When we use DistributedDataParallel with a real model, 
    we don't need such precise control and can use the gloo backend. You
    can find a table listing the capabilities of each backend at
    http://pytorch.org/docs/master/distributed.html#module-torch.distributed
    """
    os.environ["MASTER_ADDR"] = NODE_0_IP
    os.environ["MASTER_PORT"] = "29500"  # Arbitrary.
    dist.init_process_group(backend, world_size=size)
    fn(int(os.environ["RANK"]), size)


if __name__ == "__main__":
    size = 2  # my laptop and my desktop
    processes = []
    init_processes(size, run)
