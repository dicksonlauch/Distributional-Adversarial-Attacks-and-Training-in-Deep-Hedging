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
    elif epoch < 500:
        return 0.1
    elif epoch < 600:
        return 0.01
    else:
        return 0.001


def epoch_att_loader(loader, attacker, network, loss_fn,
                     K, T, dt, sequence_length,
                     delta, alpha1, alpha2, opt=None):
    """
    Runs one epoch of adversarial training for VIX hedging.

    Args:
        loader: DataLoader providing batches of (S, V, VIX, F1).
        attacker: VIX_Heston_Attacker.
        network: VIX hedging network.
        loss_fn: CVaR loss function.
        K, T, dt, sequence_length: model parameters.
        delta: attack budget.
        alpha1: weight for clean loss.
        alpha2: weight for adversarial loss.
        opt: Optimizer.

    Returns:
        Tuple (clean_loss, att_loss).
    """
    total_loss_att, total_loss_clean = 0., 0.
    for S, V, VIX, F1 in loader:
        S, V = S.to(device), V.to(device)
        VIX, F1 = VIX.to(device), F1.to(device)

        # Clean loss
        network.train()
        features = construct_features(S, VIX, F1, K, T, dt, sequence_length)
        holding = network(features)
        loss_clean = loss_fn(holding, S, F1, p0=p0_clean)

        # Adversarial loss
        if alpha2 > 0:
            network.eval()
            S_att, VIX_att, F1_att = attacker.S_budget_attack(
                network, S, VIX, F1, delta, 4, 20)
            features_att = construct_features(
                S_att, VIX_att, F1_att, K, T, dt, sequence_length)
            holding_att = network(features_att)
            loss_att = loss_fn(holding_att, S_att, F1_att, p0=p0_att)
        else:
            loss_att = torch.tensor([0.]).to(device)

        loss = alpha1 * loss_clean + alpha2 * loss_att
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_loss_att += loss_att.item()
        total_loss_clean += loss_clean.item()
    total_loss_att /= len(loader)
    total_loss_clean /= len(loader)
    return total_loss_clean, total_loss_att


# Model parameters
sequence_length = 30
dt = 1/365
learning_rate = 0.005
batch_size = 10000
epoch_num = 700

K = 100
s0 = 100
alpha_loss = 0.5

# Parse arguments
parser = argparse.ArgumentParser(description="VIX Heston adversarial training.")
parser.add_argument("--N", type=int, default=10000, help="number of samples.")
parser.add_argument("--delta", type=float, default=0., help="attack delta.")
parser.add_argument("--alpha", type=float, default=1.0, help="alpha.")
parser.add_argument("--transaction_cost_rate", type=float, default=0.0,
                    help="transaction cost rate.")
args = parser.parse_args()
args_dict = vars(args)
print(args_dict)

N = args.N
T = dt * sequence_length
delta = args.delta
alpha1 = args.alpha
alpha2 = 1.0
transaction_cost_rate = args.transaction_cost_rate

name = (f"VIX_Heston_Satt_N{N:.0e}_delta{delta}_alpha{int(alpha1)}"
        f"_tran{transaction_cost_rate:.0e}").replace("+0", "").replace("+", "")

# Load training data
data_train = torch.load('../Data/VIX_Heston_train.pt')

# Loop over data partitions
for part in range(0, int(1e5 / N)):
    idx_start = int(part * N)
    idx_end = int((part + 1) * N)

    train_data = torch.utils.data.TensorDataset(
        data_train[0][idx_start:idx_end],  # S
        data_train[1][idx_start:idx_end],  # V
        data_train[2][idx_start:idx_end],  # VIX
        data_train[3][idx_start:idx_end],  # F1
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)

    # Initialize network
    network = RNN_BN_simple_VIX(
        sequence_length=sequence_length,
        input_size=6, output_size=2, hidden_size=64
    ).to(device)

    # Loss function
    loss_fn = loss_CVAR_VIX(
        Strike_price=K, alpha_loss=alpha_loss,
        trans_cost_rate=transaction_cost_rate, p0_mode='given'
    ).to(device)

    # Trainable p0 parameters
    p0_clean = nn.Parameter(torch.tensor(1.69))
    p0_att = nn.Parameter(torch.tensor(1.69))

    opt = torch.optim.Adam([
        {'params': network.parameters()},
        {'params': [p0_clean, p0_att]}
    ], lr=learning_rate)
    LR_scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=alpha_learning_rate)

    # Attacker
    attacker = VIX_Heston_Attacker(
        loss_fn=loss_CVAR_VIX(
            Strike_price=K, alpha_loss=alpha_loss,
            trans_cost_rate=transaction_cost_rate, p0_mode='search'
        ).to(device),
        K=K, T=T, dt=dt, sequence_length=sequence_length
    )

    for module in network.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.momentum = 1.0

    print(f'Start Running {name}_part{int(part)}')

    # Training loop
    for i in range(epoch_num):
        time1 = time.time()
        network.train()
        if i < 300:
            # Warm-up: clean loss only
            train_result = epoch_att_loader(
                train_loader, attacker, network, loss_fn,
                K, T, dt, sequence_length,
                delta, 1., 0., opt)
        else:
            if i == 300:
                with torch.no_grad():
                    p0_att.copy_(p0_clean.detach().clone())
            train_result = epoch_att_loader(
                train_loader, attacker, network, loss_fn,
                K, T, dt, sequence_length,
                delta, alpha1, alpha2, opt)

        time2 = time.time()
        print(f"epoch{i}, clean_loss: {train_result[0]:.6f}, "
              f"att_loss: {train_result[1]:.6f}, time: {time2 - time1:.1f}s, "
              f"p0_clean: {p0_clean.item():.4f}, p0_att: {p0_att.item():.4f}")
        LR_scheduler.step()

    # Save trained network
    network.to('cpu')
    network.device = 'cpu'
    torch.save(network.state_dict(),
               f"../Result/{name}_part{int(part) + 1}.pth")
