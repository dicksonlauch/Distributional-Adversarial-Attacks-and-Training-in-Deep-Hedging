import torch
import matplotlib.pyplot as plt
import numpy as np
from VIX_Heston_util import construct_features, RNN_BN_simple_VIX

def run_experiment():
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
    mdl.to(device)
    mdl.eval()

    with torch.no_grad():
        feats = construct_features(S, VIX, F1, K, T, dt, seq_len)
        h = mdl(feats)
        dS = h[:, :, 0].cpu()
        dF1 = h[:, :, 1].cpu()
        
    S, VIX = S.cpu(), VIX.cpu()
    
    tau = T - np.arange(seq_len) * dt
    tau_matrix = np.tile(tau, (S.shape[0], 1))
    mny = (np.log(S[:, :-1].numpy() / K) / np.sqrt(tau_matrix)).flatten()
    
    t_flat = np.tile(np.arange(seq_len), (S.shape[0], 1)).flatten()
    v_lvl = VIX[:, :-1].flatten().numpy()
    dS_flat = dS.flatten().numpy()
    dF1_flat = dF1.flatten().numpy()
    
    # Filter for Time Step 15
    t_mask = (t_flat == 15)
    
    mny_t15 = mny[t_mask]
    v_lvl_t15 = v_lvl[t_mask]
    dS_t15 = dS_flat[t_mask]
    dF1_t15 = dF1_flat[t_mask]
    
    # Define Moneyness Bins (Adjusted OTM to capture data)
    mask_itm = mny_t15 < -1.0
    mask_atm = (mny_t15 > -0.2) & (mny_t15 < 0.2)
    mask_otm = mny_t15 > 0.4
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot ITM
    axes[0,0].scatter(v_lvl_t15[mask_itm], dS_t15[mask_itm], alpha=0.5, s=5, c='blue')
    axes[0,0].set_title('Stock Delta (ITM: Mny < -1)')
    axes[0,0].set_ylabel('d_S')
    axes[0,0].set_xlabel('VIX')
    
    axes[1,0].scatter(v_lvl_t15[mask_itm], dF1_t15[mask_itm], alpha=0.5, s=5, c='purple')
    axes[1,0].set_title('VIX Hedging (ITM: Mny < -1)')
    axes[1,0].set_ylabel('d_F1')
    axes[1,0].set_xlabel('VIX')
    
    # Plot ATM
    axes[0,1].scatter(v_lvl_t15[mask_atm], dS_t15[mask_atm], alpha=0.5, s=5, c='green')
    axes[0,1].set_title('Stock Delta (ATM: Mny ≈ 0)')
    axes[0,1].set_xlabel('VIX')
    
    axes[1,1].scatter(v_lvl_t15[mask_atm], dF1_t15[mask_atm], alpha=0.5, s=5, c='orange')
    axes[1,1].set_title('VIX Hedging (ATM: Mny ≈ 0)')
    axes[1,1].set_xlabel('VIX')
    
    # Plot OTM
    axes[0,2].scatter(v_lvl_t15[mask_otm], dS_t15[mask_otm], alpha=0.5, s=5, c='red')
    axes[0,2].set_title('Stock Delta (OTM: Mny > 0.4)')
    axes[0,2].set_xlabel('VIX')
    
    axes[1,2].scatter(v_lvl_t15[mask_otm], dF1_t15[mask_otm], alpha=0.5, s=5, c='brown')
    axes[1,2].set_title('VIX Hedging (OTM: Mny > 0.4)')
    axes[1,2].set_xlabel('VIX')
    
    plt.tight_layout()
    plt.savefig('vix_experiment.png')

if __name__ == "__main__":
    run_experiment()
