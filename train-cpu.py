#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
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
    # gloo/nccl backend, DistributedDataParallel, and multiple Dataloader
    # workers require you to change multiprocessing start method to 
    # `forkserver` (Python 3) or `spawn` since gloo/nccl aren't fork safe.
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(not distributed_training),
        num_workers=1,          pin_memory=True,
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
        return F.log_softmax(x)


def train(model, train_loader):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_loader:
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.data[0]
            loss.backward()
            optimizer.step()
        print("Epoch {epoch}: total loss={epoch_loss}".format(
            epoch=epoch, epoch_loss=epoch_loss))


if __name__ == "__main__":
    # We'll be running the train dataset distributed, but the valid locally
    DISTRIBUTED_TRAIN = True  # Set to false for no distributed train.
    WORLD_SIZE = 2
    if DISTRIBUTED_TRAIN:
        # you can't do anything (including distributed sampler) without
        # initializing first.
        os.environ["MASTER_ADDR"] = "192.168.1.30"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(
            "gloo", rank=int(os.environ["RANK"]), world_size=WORLD_SIZE)
        # using DistributedDataParallel REQUIRES backends `nccl` or `gloo`
    train_loader, valid_loader = get_data(
        batch_size=32, distributed_training=DISTRIBUTED_TRAIN)
    model = Net()
    if DISTRIBUTED_TRAIN:
        BATCH_DIM = 0  # (Distributed)DataParallel needs to know where to slice
        model = nn.parallel.DistributedDataParallel(model, dim=BATCH_DIM)
    train(model, train_loader)
