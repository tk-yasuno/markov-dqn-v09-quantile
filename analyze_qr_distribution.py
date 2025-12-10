"""
QR-DQN Return Distribution Analysis Tool (v0.9)

Detailed analysis of return distributions learned by QR-DQN:
- Distribution statistics (mean, variance, quantiles)
- Per-action distribution comparison
- State-dependent distribution analysis
- Risk analysis (worst-case scenarios via quantiles)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from train_markov_fleet import FleetQRDQN
from markov_fleet_environment import MarkovFleetEnvironment


def analyze_distribution_statistics(checkpoint_path: str, save_dir: str = "outputs_qrdqn_analysis"):
    """
    Compute detailed statistics of QR-DQN return distributions.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        save_dir: Directory to save analysis results
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    # Create agent
    n_bridges = config.get('n_urban', 20) + config.get('n_rural', 80)
    n_quantiles = config.get('n_quantiles', 200)
    
    agent = FleetQRDQN(n_bridges=n_bridges, n_actions=6, 
                       n_quantiles=n_quantiles)
    agent.load_state_dict(checkpoint['agent_state_dict'])
    agent.eval()
    
    tau = agent.tau.numpy()  # Quantile midpoints
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Sample multiple states for analysis
    n_samples = 1000
    states = np.random.randint(0, 3, (n_samples, n_bridges)).astype(np.float32)
    
    # Get quantile values for all samples
    with torch.no_grad():
        states_t = torch.FloatTensor(states)
        q_values, quantiles = agent(states_t)
    
    q_values = q_values.numpy()  # [n_samples, n_bridges, n_actions]
    quantiles = quantiles.numpy()  # [n_samples, n_bridges, n_actions, n_quantiles]
    
    # Action names
    action_names = ['None', 'Work31', 'Work33', 'Work34', 'Work35', 'Work38']
    
    print("\n" + "="*80)
    print("QR-DQN RETURN DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # ===== 1. Per-Action Distribution Statistics =====
    print("\n1. PER-ACTION DISTRIBUTION STATISTICS")
    print("-" * 80)
    
    stats_data = []
    
    for a_idx, action_name in enumerate(action_names):
        # Get all quantile values for this action
        action_quantiles = quantiles[:, :, a_idx, :]  # [n_samples, n_bridges, n_quantiles]
        action_quantiles = action_quantiles.reshape(-1, n_quantiles)  # [n_samples * n_bridges, n_quantiles]
        
        # Compute statistics from quantiles
        means = action_quantiles.mean(axis=1)  # Mean of quantiles
        
        # Variance from quantiles (approximate)
        variances = ((action_quantiles - means[:, np.newaxis])**2).mean(axis=1)
        stds = np.sqrt(variances)
        
        # Quantile-based risk measures (directly from quantiles!)
        q_05 = action_quantiles[:, int(0.05 * n_quantiles)]  # 5th percentile
        q_25 = action_quantiles[:, int(0.25 * n_quantiles)]  # 25th percentile
        q_50 = action_quantiles[:, int(0.50 * n_quantiles)]  # Median
        q_75 = action_quantiles[:, int(0.75 * n_quantiles)]  # 75th percentile
        q_95 = action_quantiles[:, int(0.95 * n_quantiles)]  # 95th percentile
        
        # VaR and CVaR
        var_5 = np.percentile(means, 5)  # Value at Risk (5%)
        cvar_5 = np.mean(means[means <= var_5])  # Conditional VaR
        
        print(f"\n{action_name}:")
        print(f"  Mean Return:    {np.mean(means):8.2f} ± {np.std(means):6.2f}")
        print(f"  Std Dev:        {np.mean(stds):8.2f} ± {np.std(stds):6.2f}")
        print(f"  Q05 (5%ile):    {np.mean(q_05):8.2f}")
        print(f"  Q25 (25%ile):   {np.mean(q_25):8.2f}")
        print(f"  Q50 (median):   {np.mean(q_50):8.2f}")
        print(f"  Q75 (75%ile):   {np.mean(q_75):8.2f}")
        print(f"  Q95 (95%ile):   {np.mean(q_95):8.2f}")
        print(f"  VaR (5%):       {var_5:8.2f}")
        print(f"  CVaR (5%):      {cvar_5:8.2f}")
        
        stats_data.append({
            'action': action_name,
            'mean': means,
            'std': stds,
            'q_05': q_05,
            'q_25': q_25,
            'q_50': q_50,
            'q_75': q_75,
            'q_95': q_95,
            'var_5': var_5,
            'cvar_5': cvar_5
        })
    
    # ===== 2. Distribution Shape Analysis =====
    print("\n\n2. DISTRIBUTION SHAPE ANALYSIS")
    print("-" * 80)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for a_idx, (action_name, ax) in enumerate(zip(action_names, axes)):
        data = stats_data[a_idx]
        
        # Plot mean distribution
        ax.hist(data['mean'], bins=50, alpha=0.7, color=f'C{a_idx}', edgecolor='black')
        ax.axvline(np.mean(data['mean']), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(data["mean"]):.2f}')
        ax.axvline(data['var_5'], color='orange', linestyle='--',
                   linewidth=2, label=f'VaR(5%): {data["var_5"]:.2f}')
        
        ax.set_xlabel('Expected Return')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{action_name}\n(std={np.mean(data["std"]):.2f}, median={np.mean(data["q_50"]):.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = save_path / "distribution_statistics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Distribution statistics plot saved to: {plot_path}")
    plt.close()
    
    # ===== 3. Uncertainty Analysis =====
    print("\n\n3. UNCERTAINTY ANALYSIS")
    print("-" * 80)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Variance comparison
    variances = [np.mean(data['std'])**2 for data in stats_data]
    ax1.bar(action_names, variances, color=['C0', 'C1', 'C2', 'C3', 'C4', 'C5'])
    ax1.set_ylabel('Variance')
    ax1.set_title('Return Variance by Action\n(Higher = More Uncertain)')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)
    
    # IQR (Interquartile Range) comparison
    iqrs = [np.mean(data['q_75'] - data['q_25']) for data in stats_data]
    ax2.bar(action_names, iqrs, color=['C0', 'C1', 'C2', 'C3', 'C4', 'C5'])
    ax2.set_ylabel('IQR (Q75 - Q25)')
    ax2.set_title('Interquartile Range by Action\n(Higher = More Spread Out)')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plot_path = save_path / "uncertainty_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Uncertainty analysis plot saved to: {plot_path}")
    plt.close()
    
    # ===== 4. Risk Profile =====
    print("\n\n4. RISK PROFILE")
    print("-" * 80)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(action_names))
    width = 0.25
    
    means = [np.mean(data['mean']) for data in stats_data]
    var_5s = [data['var_5'] for data in stats_data]
    cvar_5s = [data['cvar_5'] for data in stats_data]
    
    ax.bar(x - width, means, width, label='Mean Return', color='green', alpha=0.7)
    ax.bar(x, var_5s, width, label='VaR (5%)', color='orange', alpha=0.7)
    ax.bar(x + width, cvar_5s, width, label='CVaR (5%)', color='red', alpha=0.7)
    
    ax.set_xlabel('Action')
    ax.set_ylabel('Return Value')
    ax.set_title('Risk Profile by Action\n(VaR = Value at Risk, CVaR = Conditional Value at Risk)')
    ax.set_xticks(x)
    ax.set_xticklabels(action_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plot_path = save_path / "risk_profile.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Risk profile plot saved to: {plot_path}")
    plt.close()
    
    # ===== 5. Quantile Distribution Visualization =====
    print("\n\n5. QUANTILE DISTRIBUTION VISUALIZATION")
    print("-" * 80)
    
    # Analyze how quantile distributions change with bridge states
    state_conditions = [
        ("All Good", np.zeros(n_bridges)),
        ("All Fair", np.ones(n_bridges)),
        ("All Poor", np.ones(n_bridges) * 2),
        ("Mixed", np.random.randint(0, 3, n_bridges))
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for ax, (state_name, state) in zip(axes, state_conditions):
        state_t = torch.FloatTensor(state.astype(np.float32)).unsqueeze(0)
        
        with torch.no_grad():
            q_vals, quants = agent(state_t)
        
        q_vals = q_vals[0].numpy()  # [n_bridges, n_actions]
        quants = quants[0].numpy()  # [n_bridges, n_actions, n_quantiles]
        
        # Average over bridges
        avg_quants = quants.mean(axis=0)  # [n_actions, n_quantiles]
        avg_q = q_vals.mean(axis=0)  # [n_actions]
        
        # Plot quantile distributions for each action
        for a_idx, action_name in enumerate(action_names):
            quantile_values = avg_quants[a_idx]  # [n_quantiles]
            q_val = avg_q[a_idx]
            
            # Sort quantiles for proper visualization
            sorted_quantiles = np.sort(quantile_values)
            
            ax.plot(tau, sorted_quantiles, label=f'{action_name} (Q={q_val:.1f})', 
                   linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Quantile (τ)')
        ax.set_ylabel('Return Value')
        ax.set_title(f'State: {state_name}')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plot_path = save_path / "quantile_distributions.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Quantile distributions plot saved to: {plot_path}")
    plt.close()
    
    # ===== 6. Quantile Spread Analysis =====
    print("\n\n6. QUANTILE SPREAD ANALYSIS")
    print("-" * 80)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for a_idx, action_name in enumerate(action_names):
        data = stats_data[a_idx]
        ax.scatter(data['mean'], data['std'], alpha=0.3, s=20, 
                  label=action_name, color=f'C{a_idx}')
    
    ax.set_xlabel('Mean Return (Q-value)')
    ax.set_ylabel('Standard Deviation (Uncertainty)')
    ax.set_title('Return Distribution: Mean vs Uncertainty\n(Each point = one state-action pair)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = save_path / "mean_vs_uncertainty.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Mean vs uncertainty plot saved to: {plot_path}")
    plt.close()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll plots saved to: {save_path}")
    print("\nKey Insights:")
    print("1. Check 'distribution_statistics.png' for per-action return distributions")
    print("2. Check 'uncertainty_analysis.png' for variance and IQR comparison")
    print("3. Check 'risk_profile.png' for VaR and CVaR analysis")
    print("4. Check 'quantile_distributions.png' for how quantiles change with state")
    print("5. Check 'mean_vs_uncertainty.png' for relationship between return and uncertainty")
    
    return stats_data


def main():
    parser = argparse.ArgumentParser(description="Analyze QR-DQN Return Distributions")
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--save-dir', type=str, default='outputs_qrdqn_analysis',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("QR-DQN - RETURN DISTRIBUTION ANALYSIS")
    print("="*80 + "\n")
    
    # Run analysis
    stats_data = analyze_distribution_statistics(args.checkpoint, args.save_dir)
    
    print("\n✓ Analysis complete! Check the output directory for detailed plots.\n")


if __name__ == "__main__":
    main()
