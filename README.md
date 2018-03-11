# Exploring Distributed PyTorch
Today I learned about using DistributedDataParallel and DistributedSampler in PyTorch to run experiments over a network instead of just locally. You'll find my journal below. I intend to clean up my notes soon.

# Journal:
My goal today is to learn about distributed Pytorch. I have a laptop and desktop, both with GPUs, to play with. The laptop is Kubuntu 17.10 and has an Nvidia Quadro M2200 with driver version 390.25 and Cuda compiler 9.1. It has Python 3.6.4 (Anaconda) and Pytorch 0.3.1.post2. The desktop uses Ubuntu Gnome 16.04 and has an Nvidia GeForce GTX 1080.  It has driver version 387.34 and Cuda 9.0. Here we again have Python 3.6.4 (Anaconda) but with Pytorch 0.3.0.post4. I'm on a relatively open network. I can use SSH to access either machine from the other.

My resources include [Seb Arnold's tutorial](http://pytorch.org/tutorials/intermediate/dist_tuto.html), the [Imagenet example](https://github.com/pytorch/examples/blob/master/imagenet/main.py), and of course the [distributed package docs](http://pytorch.org/docs/master/distributed.html).

My goal is to get the laptop and the desktop to communicate by adapting Seb Arnold's tutorial. If I succeed, the next step will be training a (toy) model across the machines.

## Arnold's Setup
Seb Arnold says I should use a coordination tool to run on a cluster. While that sounds like a great idea, we're going to see how far we can get with SSH. 

## Point to Point Experiment
I'm going to copy Seb's Blocking Point-to-Point code. I'll be verifying that everything goes smoothly when ran with multiprocessing to _emulate_ multiple machines. After checking both my machines, I'll explore the cross-communication piece.

I find that I can run my `blocking-local.py` script on either machine. Looking good! Now I'm going to see how I can cross-communicate. I'm going to use my laptop as the rank 0 (master) node and my desktop as the rank 1 node. I'm expecting that if I copy the `blocking-local.py` code, delete the `__main__` block, and replace it with a call to `init_processes`, then I'll be able to copy the script over to the desktop, change the rank, and the machines will communicate. Let's find out if that rough idea is correct.

*TIP:* If you're using SSH like I am, the Linux tool `scp` will help you copy files between machines easily.

It really is that simple! I've altered the code to except `RANK` as an environment variable and I'm calling my scripts like:
```
(laptop )$ RANK=0 ./blocking-remote.py
(desktop)$ RANK=1 ./blocking-remote.py
```
and I get the RANK 0 console output on my laptop and the RANK 1 console output on my (SSH session to) my Desktop! Since we're using the blocking commands `send` and `recv`, you cannot `Ctrl-C` out of the script until both machines are executing it. And, you will not see any outputs until both machines execute.

What behavior should we expect with non-blocking communication? For this, we will use the commands `isend` and `irecv`. Arnold gives a couple caveats on using these commands, but since we won't be dwelling on low-level point-to-point comms for very long, I just skim them.

The behavior is actually quite similar since we're still waiting for the `DistributedRequest`.


## Training
I'm bored. Let's train something!

At this point, I'm going to diverge from Arnold. While it's great to get a better understanding of the inner workings of `DistributedDataParallel`, I don't feel like re-inventing the wheel. I just want to know how to use the tools that Pytorch gives me. So, I'm going to try to emulate what Arnold did using `DistributedDataParallel` and `torch.utils.data.distributed.DistributedSampler`.

Take a quick look at the Docs for DistributedDataParallel before digging in. We're going to switch to the `gloo` backend because `DistributedDataParallel` only works with `gloo` and `nccl`. This is fine because we're going to try to use multiple GPUs anyways.

I'm copying the CNN that Arnold has hidden from the tutorial in his [repo](https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py). Then I'm going to test that it works on one machine (my laptop), and then I'll wrap it up in distributed goodness!

Now that I've verified my code works on local CPU by setting `DISTRIBUTED_TRAIN` to `False`, I'm going to try it over the network.

Doesn't work. We get `AttributeError: 'torch.FloatTensor' object has no attribute 'get_device'` trying to run over the network. A little research suggests that cuda tensors do have `get_device`. Not really sure why we wouldn't be able to do this over CPU. Sounds like something to figure out another time. You can read [this issue](https://github.com/pytorch/pytorch/issues/741) for a (little) more insight. On a quick pass through the docs, it looks like DistributedDataParallel is really only meant for GPUs anyways. That's fine.

Either way, the script runs fine locally on GPU with the slight necessary adjustments. Let's see what happens over network.

It works! Interestingly, the beefier but remote 1080 seems underutilized. There's certainly a lot of overhead, too. As far as behavior, both the laptop and the desktop finished their epoch at nearly the same time. The loss values were slightly different.

*NOTE:* The script is going to throw a messy warning on shutdown. This is expected. See issue [2530](https://github.com/pytorch/pytorch/issues/2530).

Now we're going to put in a little polish. We're going to make a CLI for whether or not to use distributed and the batch size. We're also going to assert that the batch size is an integer multiple of the number of GPUs, equal or greater than 1, so that each GPU is used and each GPU has the same chunk of the batch. And finally, we're going to look into changing the multiprocessing start method so that we can have multiple workers.

*NOTE:* It looks like setting the multiprocessing start method (mentioned on the DistributedDataParallel docs) isn't really necessary. I'm basing that on the ImageNet example, which uses 4 workers by default, as well as all the other preconditions that the docs warn.


## Other methods
That seems to be working well. I'm not going to offer suggestions on the correct way to take arguments like the master IP or port. The scripts here are too simple to really gain anything from setting up anything fancy. Instead, next I'm going to explore other ways of accomplishing distributed training.

### Shared Filesystem
It may be (okay, it _probably is_) the case that you have some files shared between your master and the other nodes. I'm going to emulate that by mounting a folder on my laptop onto my desktop and we're going to try out the shared filesystem method.

All you need is to be able to specify a path that all the machines can access. I've simply created a file on my laptop and mounted it as an NFS folder on my desktop. To be precise, I'm treating my laptop as the server and the desktop as the client. But it really doesn't matter. Any shared file system will suffice. I was tempted to use my Raspberry Pi as a server!

*TIP:* When you specify a shared file, be sure to reference a non-existant file, not an existing folder.
