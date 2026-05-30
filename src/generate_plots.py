import torch
import matplotlib.pyplot as plt
import numpy as np
from VIX_Heston_util import construct_features, RNN_BN_simple_VIX

def plot_agent_behavior(mdl_path, data_path, K=100, T=30/365, dt=1/365, seq_len=30):
    device = torch.device("cpu")
    
    data_test = torch.load(data_path, map_location=device)
    S, V, VIX, F1 = [t.to(device) for t in data_test]
    
    mdl = RNN_BN_simple_VIX(sequence_length=seq_len, input_size=6, output_size=2, hidden_size=64, device=device)
    mdl.load_state_dict(torch.load(mdl_path, map_location=device))
    mdl.to(device)
    mdl.eval()

    with torch.no_grad():
        feats = construct_features(S, VIX, F1, K, T, dt, seq_len)
        h = mdl(feats)
        dS = h[:, :, 0].cpu()
        dF1 = h[:, :, 1].cpu()
        
    S, VIX = S.cpu(), VIX.cpu()
    plt.style.use('seaborn-v0_8-whitegrid')
    
    tau = T - np.arange(seq_len) * dt
    tau_matrix = np.tile(tau, (S.shape[0], 1))
    mny = (np.log(S[:, :-1].numpy() / K) / np.sqrt(tau_matrix)).flatten()
    t_flat = np.tile(np.arange(seq_len), (S.shape[0], 1)).flatten()
    v_lvl = VIX[:, :-1].flatten().numpy()
    dS_flat = dS.flatten().numpy()
    dF1_flat = dF1.flatten().numpy()
    
    idx_s = np.random.choice(len(mny), 10000, replace=False)

    # Plot 1: Stock Delta
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111, projection='3d')
    sc1 = ax1.scatter(mny[idx_s], t_flat[idx_s], dS_flat[idx_s], c=v_lvl[idx_s], cmap='coolwarm', alpha=0.5, s=2)
    ax1.set_xlabel('Moneyness')
    ax1.set_ylabel('Time Step (Days)')
    ax1.set_zlabel('d_S')
    ax1.set_title('Stock Delta vs. Moneyness and Time')
    plt.colorbar(sc1, ax=ax1, label='VIX', pad=0.1)
    
    # Save the stock delta plot
    # Set view angle to see the moneyness/d_S relationship clearly
    ax1.view_init(elev=20, azim=45)
    plt.savefig('stock_delta_3d.png')
    plt.close()

    # Plot 2: VIX Hedging
    fig = plt.figure(figsize=(10, 8))
    ax2 = fig.add_subplot(111, projection='3d')
    sc2 = ax2.scatter(mny[idx_s], t_flat[idx_s], dF1_flat[idx_s], c=v_lvl[idx_s], cmap='viridis', alpha=0.5, s=2)
    ax2.set_xlabel('Moneyness')
    ax2.set_ylabel('Time Step (Days)')
    ax2.set_zlabel('d_F1')
    ax2.set_title('VIX Hedging vs. Moneyness and Time')
    plt.colorbar(sc2, ax=ax2, label='VIX', pad=0.1)
    
    # Save the VIX hedging plot
    ax2.view_init(elev=20, azim=45)
    plt.savefig('vix_hedging_3d.png')
    plt.close()
    
    print("Plots saved.")

plot_agent_behavior('../Result/VIX_HestonClean_mse_N1e4_tran2e-3_part3.pth', '../Data/VIX_Heston_val.pt')
