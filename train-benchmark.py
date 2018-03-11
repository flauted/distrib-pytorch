#!/usr/bin/env python
import os
import argparse
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing
from torch.autograd import Variable
from torchvision import datasets, transforms


def get_data(batch_size=32, distributed_training=True):
    """Get the data."""
    train_dataset = datasets.MNIST(
        "./data", train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    if distributed_training:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(not distributed_training),
        num_workers=5,
        pin_memory=True,
        sampler=train_sampler)

    valid_dataset = datasets.MNIST(
        "./data", train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False)
    return train_loader, valid_loader


class Net(nn.Module):
    """ Network architecture. """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def format_time(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return "%d h, %02d m, %02d s" % (h, m, s)



def train(model, train_loader, cpu=False, num_epochs=10):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    start_time = time.time()
    total_batches = 0
    epoch_times = []
    for epoch in range(num_epochs):
        epoch_time = time.time()
        epoch_loss = 0.0
        batches = 0
        for data, target in train_loader:
            batches += 1
            total_batches += 1
            if not cpu:
                data, target = Variable(data.cuda()), Variable(target.cuda())
            else:
                data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.data[0]
            loss.backward()
            optimizer.step()
        total_sec = time.time() - start_time
        time_elapsed = format_time(time.time() - start_time)
        epoch_elapsed = format_time(time.time() - epoch_time)
        epoch_times.append(time.time() - epoch_time)
        print("Epoch {epoch}, time elapsed {time} [{e_time}], num batches {batches}: "
              "total loss={epoch_loss}".format(
            epoch=epoch,
            time=time_elapsed,
            e_time=epoch_elapsed,
            epoch_loss=epoch_loss,
            batches=batches))
    print("\nTotal {num_epochs} epochs "
          "({total_batches} batches, {batches} batches per epoch). \n    "
          "total time: {time},\n    avg epoch time: {avg_e_time}".format(
        num_epochs=num_epochs,
        total_batches=total_batches,
        batches=total_batches/num_epochs,
        time=format_time(time.time() - start_time),
        avg_e_time=format_time((sum(epoch_times)/len(epoch_times)))))


if __name__ == "__main__":
    # We'll be running the train dataset distributed, but the valid locally
    parser = argparse.ArgumentParser(
        description="Train a model on MNIST across a network.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # No matter what, we use batch size.
    parser.add_argument("-b", "--batch-size", type=int, default=100)
    subparsers = parser.add_subparsers(help="Subparser.", dest="distributed")

    local = subparsers.add_parser(
        "local", help="Use local training.")
    local.add_argument("-c", "--cpu", action="store_true",
                       help="Train on CPU.")

    dister = subparsers.add_parser(
        "distributed", help="Use distributed training.")
    dister.add_argument("-w", "--world-size", type=int, required=True,
                        help="The number of nodes that will run the training.")
    d_subparsers = dister.add_subparsers(help="d_subparser.", dest="method")

    filemd = d_subparsers.add_parser(
        "shared-file", help="Use shared file to coordinate nodes.")
    filemd.add_argument("-r", "--rank", type=int, default=None,
                        help="The rank 0 node (master) has some extra "
                             "responsibilities.")
    filemd.add_argument("-f", "--file", required=True, type=str,
                        help="Location of nonexistant file that nodes will"
                             " use to coordinate. Specify a standard"
                             " path. We handle the `file://` internally")
    filemd.add_argument("-g", "--group", type=str, default="MNIST",
                        help="A group allows multiple jobs to run on the same"
                             " cluster without collision.")

    envmd = d_subparsers.add_parser(
        "env-vars", help="Use environment variables to negotiate a port.")
    envmd.add_argument("-r", "--rank", type=int, required=True,
                       help="The rank 0 node (master) has some extra "
                            "responsibilities.")
    envmd.add_argument("-a", "--master-addr", type=str, required=True,
                       help="IP of the node 0 machine.")
    envmd.add_argument("-p", "--master-port", type=int, default=29500,
                       help="An open port on the master node.")
    FLAGS = parser.parse_args()

    FLAGS.distributed = True if FLAGS.distributed == "distributed" else False
    FLAGS.cpu = False if FLAGS.distributed else FLAGS.cpu

    assert FLAGS.batch_size > 0, "Specify a positive batch size."
    if FLAGS.distributed:
        assert FLAGS.batch_size // FLAGS.world_size == FLAGS.batch_size / FLAGS.world_size, (
            "--batch-size must be an integer multiple of --world-size.")
        assert FLAGS.batch_size >= FLAGS.world_size, (
            "--batch-size must be larger than --world-size")
        if FLAGS.method == "shared-file":
            assert not FLAGS.file.startswith("file://"), (
                "Do not start --file with `file://`. This is handled internally.")
            assert not os.path.isdir(FLAGS.file)
    if FLAGS.cpu is False:
        assert torch.cuda.is_available()

    if FLAGS.distributed:
        print("Training on a network environment with method `{}`".format(
            FLAGS.method))
        # initialize first. Everything else is locked until you do.
        # using DistributedDataParallel REQUIRES backends `nccl` or `gloo`
        if FLAGS.method == "shared-file":
            # Rank is optional for filemethod inits.
            if FLAGS.rank is not None:
                dist.init_process_group(
                    "gloo", init_method="file://" + FLAGS.file,
                    world_size=FLAGS.world_size, group_name=FLAGS.group)
            else:
                dist.init_process_group(
                    "gloo",
                    init_method="file://" + FLAGS.file,
                    rank=FLAGS.rank,
                    world_size=FLAGS.world_size,
                    group_name=FLAGS.group)
        elif FLAGS.method == "env-vars":
            os.environ["MASTER_ADDR"] = FLAGS.master_addr
            os.environ["MASTER_PORT"] = str(FLAGS.master_port)
            dist.init_process_group(
                "gloo", rank=FLAGS.rank, world_size=FLAGS.world_size)
    else:
        print("Training on {}local {}".format(
            "" if FLAGS.cpu else str(torch.cuda.device_count()) + " ",
            "CPU" if FLAGS.cpu else
            "GPU" if torch.cuda.device_count() == 1 else "GPUs"))

    train_loader, valid_loader = get_data(
        batch_size=FLAGS.batch_size,
        distributed_training=FLAGS.distributed)

    model = Net()
    if not FLAGS.cpu:
        if not FLAGS.distributed:
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.cuda()
        else:
            BATCH_DIM = 0  # (Distributed)DataParallel needs to know where to slice
            model = model.cuda()
            model = nn.parallel.DistributedDataParallel(model, dim=BATCH_DIM)

    train(model, train_loader, cpu=FLAGS.cpu)
