import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from orbit_q.engine.models.autoencoder import PyTorchAutoencoder
import numpy as np
import time


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)  # use gloo or nccl


def cleanup():
    dist.destroy_process_group()


def train_model(rank, world_size, X_tensor, epochs=20):
    setup(rank, world_size)

    # Create model and move it to GPU with id rank
    model = PyTorchAutoencoder(input_dim=X_tensor.shape[1]).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)

    # Simple chunking for demo
    chunk_size = len(X_tensor) // world_size
    local_X = X_tensor[rank * chunk_size : (rank + 1) * chunk_size].to(rank)

    ddp_model.train()
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = ddp_model(local_X)
        loss = loss_fn(outputs, local_X)
        loss.backward()
        optimizer.step()

    duration = time.time() - start_time
    if rank == 0:
        print(f"DDP Training completed in {duration:.2f}s across {world_size} GPUs/processes")

    cleanup()


def run_ddp_training(X, world_size=2):
    """Entry point for DDP simulated training across CPUs/GPUs."""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    mp.spawn(train_model, args=(world_size, X_tensor, 20), nprocs=world_size, join=True)


if __name__ == "__main__":
    print("Running dummy DDP load test...")
    dummy_data = np.random.normal(0, 1, (10000, 5))
    run_ddp_training(dummy_data, world_size=2)
