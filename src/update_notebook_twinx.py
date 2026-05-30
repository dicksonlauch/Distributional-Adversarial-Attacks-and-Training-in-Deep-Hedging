import json

def update_notebook():
    with open('VIX_Heston_charts.ipynb', 'r') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if 'def plot_agent_behavior' in source:
                
                # Crash Scenario
                old_crash_plot = """    fig, ax = plt.subplots(figsize=(10, 3))
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
                
                new_crash_plot = """    fig, ax = plt.subplots(figsize=(10, 3))
    ax2 = ax.twinx()
    l1 = ax.plot(t_steps, dS[idx_c].numpy(), 'g-', label='Agent d_S')
    l2 = ax.plot(t_steps, theo_delta, 'g--', alpha=0.5, label='BS Delta')
    l3 = ax2.plot(t_steps, dF1[idx_c].numpy(), 'purple', linestyle='-', label='Agent d_F1')
    l4 = ax2.plot(t_steps, theo_vega, 'purple', linestyle='--', alpha=0.5, label='BS Vega Hedge')
    ax.set_xlabel('Days')
    ax.set_ylabel('Stock Delta', color='g')
    ax2.set_ylabel('VIX Hedge', color='purple')
    ax.set_title('Agent Response vs Theoretical (Crash Scenario)')
    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right', bbox_to_anchor=(1.35, 1))
    plt.tight_layout()
    plt.show()"""
                
                source = source.replace(old_crash_plot, new_crash_plot)

                # Up Scenario
                old_up_plot = """    fig, ax = plt.subplots(figsize=(10, 3))
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
                
                new_up_plot = """    fig, ax = plt.subplots(figsize=(10, 3))
    ax2 = ax.twinx()
    l1 = ax.plot(t_steps, dS[idx_u].numpy(), 'g-', label='Agent d_S')
    l2 = ax.plot(t_steps, theo_delta, 'g--', alpha=0.5, label='BS Delta')
    l3 = ax2.plot(t_steps, dF1[idx_u].numpy(), 'purple', linestyle='-', label='Agent d_F1')
    l4 = ax2.plot(t_steps, theo_vega, 'purple', linestyle='--', alpha=0.5, label='BS Vega Hedge')
    ax.set_xlabel('Days')
    ax.set_ylabel('Stock Delta', color='g')
    ax2.set_ylabel('VIX Hedge', color='purple')
    ax.set_title('Agent Response vs Theoretical (Up Scenario)')
    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right', bbox_to_anchor=(1.35, 1))
    plt.tight_layout()
    plt.show()"""
                
                source = source.replace(old_up_plot, new_up_plot)

                # Pin Risk Scenario
                old_pin_plot = """    fig, ax = plt.subplots(figsize=(10, 3))
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
                
                new_pin_plot = """    fig, ax = plt.subplots(figsize=(10, 3))
    ax2 = ax.twinx()
    l1 = ax.plot(t_steps, dS[idx_p].numpy(), 'g-', label='Agent d_S')
    l2 = ax.plot(t_steps, theo_delta, 'g--', alpha=0.5, label='BS Delta')
    l3 = ax2.plot(t_steps, dF1[idx_p].numpy(), 'purple', linestyle='-', label='Agent d_F1')
    l4 = ax2.plot(t_steps, theo_vega, 'purple', linestyle='--', alpha=0.5, label='BS Vega Hedge')
    ax.set_xlabel('Days')
    ax.set_ylabel('Stock Delta', color='g')
    ax2.set_ylabel('VIX Hedge', color='purple')
    ax.set_title('Agent Response vs Theoretical (Pin Risk Scenario)')
    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right', bbox_to_anchor=(1.35, 1))
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
