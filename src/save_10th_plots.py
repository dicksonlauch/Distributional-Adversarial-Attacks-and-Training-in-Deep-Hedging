import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as si
from VIX_Heston_util import construct_features, RNN_BN_simple_VIX

def bs_put_delta(S, K, T, t, sigma, r=0.0):
    tau = T - t
    tau = np.maximum(tau, 1e-6)
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*tau) / (sigma * np.sqrt(tau))
    return si.norm.cdf(d1) - 1.0

def bs_vega_hedge(S, K, T, t, sigma, r=0.0):
    tau = T - t
    tau = np.maximum(tau, 1e-6)
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*tau) / (sigma * np.sqrt(tau))
    vega = S * np.sqrt(tau) * si.norm.pdf(d1)
    return vega / 100.0

def save_plots():
    device = torch.device("cpu")
    mdl_path = '../Result/VIX_HestonClean_mse_Put_N1e5_tran2e-3_part1.pth'
    data_path = '../Data/VIX_Heston_val.pt'
    K=100
    T=30/365
    dt=1/365
    seq_len=30
    
    data_test = torch.load(data_path, map_location=device, weights_only=False)
    S, V, VIX, F1 = [t.to(device) for t in data_test]
    
    mdl = RNN_BN_simple_VIX(sequence_length=seq_len, input_size=6, output_size=2, hidden_size=64, device=device)
    mdl.load_state_dict(torch.load(mdl_path, map_location=device, weights_only=False))
    mdl.eval()

    with torch.no_grad():
        feats = construct_features(S, VIX, F1, K, T, dt, seq_len)
        h = mdl(feats)
        dS = h[:, :, 0].numpy()
        dF1 = h[:, :, 1].numpy()
        
    S = S.numpy()
    VIX = VIX.numpy()

    plt.style.use('seaborn-v0_8-whitegrid')
    t_steps = np.arange(seq_len)
    t_arr = t_steps * dt

    rets = S[:, -1] / S[:, 0]
    sorted_rets_indices = np.argsort(rets)
    abs_diff = np.abs(S[:, -1] - K)
    sorted_pin_indices = np.argsort(abs_diff)

    # 1. 10th Worst Crash Scenario
    idx_c = sorted_rets_indices[9]
    S_path = S[idx_c, :-1]
    VIX_path = VIX[idx_c, :-1]
    sigma_path = VIX_path / 100.0
    theo_delta = bs_put_delta(S_path, K, T, t_arr, sigma_path)
    theo_vega = bs_vega_hedge(S_path, K, T, t_arr, sigma_path)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax2 = ax.twinx()
    l1 = ax.plot(t_steps, dS[idx_c], 'g-', label='Agent d_S')
    l2 = ax.plot(t_steps, theo_delta, 'g--', alpha=0.5, label='BS Delta')
    l3 = ax2.plot(t_steps, dF1[idx_c], 'purple', linestyle='-', label='Agent d_F1')
    l4 = ax2.plot(t_steps, theo_vega, 'purple', linestyle='--', alpha=0.5, label='BS Vega Hedge')
    ax.set_xlabel('Days')
    ax.set_ylabel('Stock Delta', color='g')
    ax2.set_ylabel('VIX Hedge', color='purple')
    ax.set_title('10th Worst Crash Scenario')
    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right', bbox_to_anchor=(1.35, 1))
    plt.tight_layout()
    plt.savefig('traj_crash_10.png')

    # 2. 10th Best Up Scenario
    idx_u = sorted_rets_indices[-10]
    S_path = S[idx_u, :-1]
    VIX_path = VIX[idx_u, :-1]
    sigma_path = VIX_path / 100.0
    theo_delta = bs_put_delta(S_path, K, T, t_arr, sigma_path)
    theo_vega = bs_vega_hedge(S_path, K, T, t_arr, sigma_path)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax2 = ax.twinx()
    l1 = ax.plot(t_steps, dS[idx_u], 'g-', label='Agent d_S')
    l2 = ax.plot(t_steps, theo_delta, 'g--', alpha=0.5, label='BS Delta')
    l3 = ax2.plot(t_steps, dF1[idx_u], 'purple', linestyle='-', label='Agent d_F1')
    l4 = ax2.plot(t_steps, theo_vega, 'purple', linestyle='--', alpha=0.5, label='BS Vega Hedge')
    ax.set_xlabel('Days')
    ax.set_ylabel('Stock Delta', color='g')
    ax2.set_ylabel('VIX Hedge', color='purple')
    ax.set_title('10th Best Up Scenario')
    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right', bbox_to_anchor=(1.35, 1))
    plt.tight_layout()
    plt.savefig('traj_up_10.png')

    # 3. 10th Tightest Pin Risk Scenario
    idx_p = sorted_pin_indices[9]
    S_path = S[idx_p, :-1]
    VIX_path = VIX[idx_p, :-1]
    sigma_path = VIX_path / 100.0
    theo_delta = bs_put_delta(S_path, K, T, t_arr, sigma_path)
    theo_vega = bs_vega_hedge(S_path, K, T, t_arr, sigma_path)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax2 = ax.twinx()
    l1 = ax.plot(t_steps, dS[idx_p], 'g-', label='Agent d_S')
    l2 = ax.plot(t_steps, theo_delta, 'g--', alpha=0.5, label='BS Delta')
    l3 = ax2.plot(t_steps, dF1[idx_p], 'purple', linestyle='-', label='Agent d_F1')
    l4 = ax2.plot(t_steps, theo_vega, 'purple', linestyle='--', alpha=0.5, label='BS Vega Hedge')
    ax.set_xlabel('Days')
    ax.set_ylabel('Stock Delta', color='g')
    ax2.set_ylabel('VIX Hedge', color='purple')
    ax.set_title('10th Tightest Pin Risk Scenario')
    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right', bbox_to_anchor=(1.35, 1))
    plt.tight_layout()
    plt.savefig('traj_pin_10.png')

if __name__ == "__main__":
    save_plots()
