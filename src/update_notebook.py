import json

def update_notebook():
    with open('VIX_Heston_charts.ipynb', 'r') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if 'def plot_agent_behavior' in source:
                # Replace moneyness definition
                old_mny = "mny = (S[:, :-1] / K).flatten().numpy()"
                new_mny = """tau = T - np.arange(seq_len) * dt
    tau_matrix = np.tile(tau, (S.shape[0], 1))
    mny = (np.log(S[:, :-1].numpy() / K) / np.sqrt(tau_matrix)).flatten()
    t_flat = np.tile(np.arange(seq_len), (S.shape[0], 1)).flatten()"""
                
                source = source.replace(old_mny, new_mny)
                
                # Replace Plot 2 and Plot 3 with 3D plots
                old_plot23 = """    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sc1 = axes[0].scatter(mny[idx_s], dS_flat[idx_s], c=v_lvl[idx_s], cmap='coolwarm', alpha=0.5, s=2)
    axes[0].set_xlabel('Moneyness')
    axes[0].set_ylabel('d_S')
    axes[0].set_title('Stock Delta vs. Moneyness')
    plt.colorbar(sc1, ax=axes[0], label='VIX')

    sc2 = axes[1].scatter(v_lvl[idx_s], dF1_flat[idx_s], c=mny[idx_s], cmap='viridis', alpha=0.5, s=2)
    axes[1].set_xlabel('VIX')
    axes[1].set_ylabel('d_F1')
    axes[1].set_title('VIX Hedging vs. VIX Level')
    plt.colorbar(sc2, ax=axes[1], label='Moneyness')
    
    plt.tight_layout()
    plt.show()


    # ==========================================
    # Plot 3: Additional Scatter (d_F1 vs Moneyness and d_S vs VIX)
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sc3 = axes[0].scatter(mny[idx_s], dF1_flat[idx_s], c=v_lvl[idx_s], cmap='coolwarm', alpha=0.5, s=2)
    axes[0].set_xlabel('Moneyness')
    axes[0].set_ylabel('d_F1')
    axes[0].set_title('VIX Hedging vs. Moneyness')
    plt.colorbar(sc3, ax=axes[0], label='VIX')

    sc4 = axes[1].scatter(v_lvl[idx_s], dS_flat[idx_s], c=mny[idx_s], cmap='viridis', alpha=0.5, s=2)
    axes[1].set_xlabel('VIX')
    axes[1].set_ylabel('d_S')
    axes[1].set_title('Stock Delta vs. VIX Level')
    plt.colorbar(sc4, ax=axes[1], label='Moneyness')
    
    plt.tight_layout()
    plt.show()"""

                new_plot23 = """    fig = plt.figure(figsize=(14, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(mny[idx_s], t_flat[idx_s], dS_flat[idx_s], c=v_lvl[idx_s], cmap='coolwarm', alpha=0.5, s=2)
    ax1.set_xlabel('Moneyness')
    ax1.set_ylabel('Time Step (Days)')
    ax1.set_zlabel('d_S')
    ax1.set_title('Stock Delta vs. Moneyness and Time')
    plt.colorbar(sc1, ax=ax1, label='VIX', pad=0.1)

    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(mny[idx_s], t_flat[idx_s], dF1_flat[idx_s], c=v_lvl[idx_s], cmap='viridis', alpha=0.5, s=2)
    ax2.set_xlabel('Moneyness')
    ax2.set_ylabel('Time Step (Days)')
    ax2.set_zlabel('d_F1')
    ax2.set_title('VIX Hedging vs. Moneyness and Time')
    plt.colorbar(sc2, ax=ax2, label='VIX', pad=0.1)
    
    plt.tight_layout()
    plt.show()"""
                
                source = source.replace(old_plot23, new_plot23)
                
                # Update cell source
                cell['source'] = [line + '\n' for line in source.split('\n')]
                # Fix last element newline
                if cell['source']:
                    cell['source'][-1] = cell['source'][-1].rstrip('\n')
                break
                
    with open('VIX_Heston_charts.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    update_notebook()
