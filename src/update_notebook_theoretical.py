import json

def update_notebook():
    with open('VIX_Heston_charts.ipynb', 'r') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if 'def plot_agent_behavior' in source:
                # Add scipy import if not present
                if 'import scipy.stats as si' not in source:
                    source = source.replace('import numpy as np', 'import numpy as np\nimport scipy.stats as si')
                
                # Add theoretical functions
                theo_funcs = """
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

def plot_agent_behavior"""
                if 'def bs_put_delta' not in source:
                    source = source.replace('def plot_agent_behavior', theo_funcs)
                
                # Replace Crash Scenario
                old_crash_plot = """    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t_steps, dS[idx_c].numpy(), 'g-', label='d_S')
    ax.plot(t_steps, dF1[idx_c].numpy(), 'purple', linestyle='-.', label='d_F1')
    ax.set_xlabel('Days')
    ax.set_ylabel('Position')
    ax.set_title('Agent Response')
    ax.legend()
    plt.show()"""
                
                new_crash_plot = """    S_path = S[idx_c, :-1].numpy()
    VIX_path = VIX[idx_c, :-1].numpy()
    sigma_path = VIX_path / 100.0
    t_arr = t_steps * dt
    theo_delta = bs_put_delta(S_path, K, T, t_arr, sigma_path)
    theo_vega = bs_vega_hedge(S_path, K, T, t_arr, sigma_path)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t_steps, dS[idx_c].numpy(), 'g-', label='Agent d_S')
    ax.plot(t_steps, theo_delta, 'g--', alpha=0.5, label='BS Delta')
    ax.plot(t_steps, dF1[idx_c].numpy(), 'purple', linestyle='-', label='Agent d_F1')
    ax.plot(t_steps, theo_vega, 'purple', linestyle='--', alpha=0.5, label='BS Vega Hedge')
    ax.set_xlabel('Days')
    ax.set_ylabel('Position')
    ax.set_title('Agent Response vs Theoretical (Crash Scenario)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.tight_layout()
    plt.show()"""
                
                source = source.replace(old_crash_plot, new_crash_plot)

                # Replace Up Scenario
                old_up_plot = """    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t_steps, dS[idx_u].numpy(), 'g-', label='d_S')
    ax.plot(t_steps, dF1[idx_u].numpy(), 'purple', linestyle='-.', label='d_F1')
    ax.set_xlabel('Days')
    ax.set_ylabel('Position')
    ax.set_title('Agent Response (Up Scenario)')
    ax.legend()
    plt.show()"""
                
                new_up_plot = """    S_path = S[idx_u, :-1].numpy()
    VIX_path = VIX[idx_u, :-1].numpy()
    sigma_path = VIX_path / 100.0
    theo_delta = bs_put_delta(S_path, K, T, t_arr, sigma_path)
    theo_vega = bs_vega_hedge(S_path, K, T, t_arr, sigma_path)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t_steps, dS[idx_u].numpy(), 'g-', label='Agent d_S')
    ax.plot(t_steps, theo_delta, 'g--', alpha=0.5, label='BS Delta')
    ax.plot(t_steps, dF1[idx_u].numpy(), 'purple', linestyle='-', label='Agent d_F1')
    ax.plot(t_steps, theo_vega, 'purple', linestyle='--', alpha=0.5, label='BS Vega Hedge')
    ax.set_xlabel('Days')
    ax.set_ylabel('Position')
    ax.set_title('Agent Response vs Theoretical (Up Scenario)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.tight_layout()
    plt.show()"""
                
                source = source.replace(old_up_plot, new_up_plot)

                # Replace Pin Risk Scenario
                old_pin_plot = """    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t_steps, dS[idx_p].numpy(), 'g-', label='d_S')
    ax.plot(t_steps, dF1[idx_p].numpy(), 'purple', linestyle='-.', label='d_F1')
    ax.set_xlabel('Days')
    ax.set_ylabel('Position')
    ax.set_title('Agent Response (Pin Risk Scenario)')
    ax.legend()
    plt.show()"""
                
                new_pin_plot = """    S_path = S[idx_p, :-1].numpy()
    VIX_path = VIX[idx_p, :-1].numpy()
    sigma_path = VIX_path / 100.0
    theo_delta = bs_put_delta(S_path, K, T, t_arr, sigma_path)
    theo_vega = bs_vega_hedge(S_path, K, T, t_arr, sigma_path)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t_steps, dS[idx_p].numpy(), 'g-', label='Agent d_S')
    ax.plot(t_steps, theo_delta, 'g--', alpha=0.5, label='BS Delta')
    ax.plot(t_steps, dF1[idx_p].numpy(), 'purple', linestyle='-', label='Agent d_F1')
    ax.plot(t_steps, theo_vega, 'purple', linestyle='--', alpha=0.5, label='BS Vega Hedge')
    ax.set_xlabel('Days')
    ax.set_ylabel('Position')
    ax.set_title('Agent Response vs Theoretical (Pin Risk Scenario)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.tight_layout()
    plt.show()"""
                
                source = source.replace(old_pin_plot, new_pin_plot)

                # Update cell source
                cell['source'] = [line + '\n' for line in source.split('\n')]
                if cell['source']:
                    cell['source'][-1] = cell['source'][-1].rstrip('\n')
                break
                
    with open('VIX_Heston_charts.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    update_notebook()
