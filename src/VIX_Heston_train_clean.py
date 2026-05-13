#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import argparse
import time
from VIX_Heston_util import *

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"running on {device}")


def alpha_learning_rate(epoch):
    if epoch < 200:
        return 1
    elif epoch < 400:
        return 0.1
    elif epoch < 600:
        return 0.01
    else:
        return 0.001


def epoch_loader(loader, network, loss_fn, K, T, dt, sequence_length, opt=None):
    """
    Runs one epoch of clean training for VIX hedging.

    Args:
        loader: DataLoader providing batches of (S, V, VIX, F1).
        network: VIX hedging network.
        loss_fn: CVaR loss function.
        K, T, dt, sequence_length: model parameters for feature construction.
        opt: Optimizer (if training, None for evaluation).

    Returns:
        Average loss for the epoch.
    """
    total_loss = 0.
    for S, V, VIX, F1 in loader:
        S, V = S.to(device), V.to(device)
        VIX, F1 = VIX.to(device), F1.to(device)

        # Construct input features
        features = construct_features(S, VIX, F1, K, T, dt, sequence_length)
        holding = network(features)
        loss = loss_fn(holding, S, F1, VIX, p0=p0_clean)

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_loss += loss.item()
    total_loss /= len(loader)
    return total_loss


# Model parameters (same as Heston pipeline)
sequence_length = 30
dt = 1/365
learning_rate = 0.05
batch_size = 10000
epoch_num = 700

K = 100
s0 = 100
alpha_loss = 0.5

# Parse arguments
parser = argparse.ArgumentParser(description="VIX Heston clean training.")
parser.add_argument("--N", type=int, default=10000, help="number of samples.")
parser.add_argument("--transaction_cost_rate", type=float, default=0.0,
                    help="transaction cost rate.")
args = parser.parse_args()
args_dict = vars(args)
print(args_dict)
N = args.N
T = dt * sequence_length
transaction_cost_rate = args.transaction_cost_rate

name = f"VIX_HestonClean_N{N:.0e}_tran{transaction_cost_rate:.0e}".replace("+0", "").replace("-0", "-")

# Load training data: (S, V, VIX, F1)
data_train = torch.load('../Data/VIX_Heston_train.pt')

# Loop over data partitions
for part in range(0, int(1e5 / N)):
    idx_start = int(part * N)
    idx_end = int((part + 1) * N)

    # Prepare partitioned training data and loader
    train_data = torch.utils.data.TensorDataset(
        data_train[0][idx_start:idx_end],  # S
        data_train[1][idx_start:idx_end],  # V
        data_train[2][idx_start:idx_end],  # VIX
        data_train[3][idx_start:idx_end],  # F1
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)

    # Initialize network and loss
    network = RNN_BN_simple_VIX(
        sequence_length=sequence_length,
        input_size=6, output_size=2, hidden_size=64
    ).to(device)

    loss_fn = loss_CVAR_VIX(
        Strike_price=K, alpha_loss=alpha_loss,
        trans_cost_rate=transaction_cost_rate, p0_mode='given'
    ).to(device)

    p0_clean = nn.Parameter(torch.tensor(1.69))
    opt = torch.optim.Adam([
        {'params': network.parameters()},
        {'params': [p0_clean]}
    ], lr=learning_rate)
    LR_scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=alpha_learning_rate)

    # Training loop
    print(f'Start Running {name}_part{int(part)}')
    for i in range(epoch_num):
        time1 = time.time()
        network.train()
        train_result = epoch_loader(
            train_loader, network, loss_fn, K, T, dt, sequence_length, opt=opt)
        time2 = time.time()
        print(f"epoch {i}, train loss: {train_result:.6f}, "
              f"time: {time2 - time1:.1f}s, p0: {p0_clean.item():.4f}")
        LR_scheduler.step()

    # Save trained network
    network.to('cpu')
    network.device = 'cpu'
    torch.save(network.state_dict(),
               f"../Result/{name}_part{int(part) + 1}.pth")
