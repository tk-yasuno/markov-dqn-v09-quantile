"""
Visualization Tool for Markov Fleet QR-DQN v0.9 (Quantile Regression)
Comprehensive training results visualization with QR-DQN quantile analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from train_markov_fleet import FleetQRDQN


def plot_training_curves_v09(checkpoint_path: str, save_dir: str = "outputs_markov_v09/plots"):
    """Plot comprehensive training curves for v0.9 with QR-DQN"""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    rewards_history = checkpoint['rewards_history']
    costs_history = checkpoint['costs_history']
    losses_history = checkpoint.get('losses_history', [])
    config = checkpoint.get('config', {})
    episodes = checkpoint.get('episodes', len(rewards_history))
    
    # QR-DQN parameters
    n_quantiles = config.get('n_quantiles', 200)
    kappa = config.get('kappa', 1.0)
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Compute moving averages
    window = 50
    def moving_average(data, w):
        if len(data) < w:
            return data
        return np.convolve(data, np.ones(w)/w, mode='valid')
    
    ma_rewards = moving_average(rewards_history, window)
    ma_costs = moving_average(costs_history, window)
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Episode Rewards
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(rewards_history, alpha=0.3, label='Raw', color='red')
    if len(ma_rewards) > 0:
        ax1.plot(range(window-1, episodes), ma_rewards, 
                label=f'MA({window})', color='darkred', linewidth=2)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title(f'Markov Fleet QR-DQN v0.9: Episode Rewards\n(Quantile Regression, {n_quantiles} quantiles)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Total Costs
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(costs_history, alpha=0.3, label='Raw', color='purple')
    if len(ma_costs) > 0:
        ax2.plot(range(window-1, episodes), ma_costs, 
                label=f'MA({window})', color='indigo', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Cost ($k)')
    ax2.set_title('Total Maintenance Cost\n(30-year horizon)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Reward-Cost Trade-off
    ax3 = plt.subplot(2, 3, 3)
    scatter = ax3.scatter(costs_history, rewards_history, c=range(episodes), 
                         cmap='plasma', alpha=0.6, s=20)
    ax3.set_xlabel('Total Cost ($k)')
    ax3.set_ylabel('Total Reward')
    ax3.set_title('Reward-Cost Trade-off\n(Color = Episode)')
    plt.colorbar(scatter, ax=ax3, label='Episode')
    ax3.grid(True, alpha=0.3)
    
    # 4. Training Loss (Cross-Entropy)
    ax4 = plt.subplot(2, 3, 4)
    if len(losses_history) > 0:
        ax4.plot(losses_history, alpha=0.3, color='orange', label='Raw')
        ma_loss = moving_average(losses_history, min(window, len(losses_history)//10))
        if len(ma_loss) > 0:
            offset = len(losses_history) - len(ma_loss)
            ax4.plot(range(offset, len(losses_history)), ma_loss, 
                    color='darkorange', linewidth=2, label='MA')
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Quantile Huber Loss')
        ax4.set_title(f'Training Loss (QR-DQN, κ={kappa})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
    else:
        ax4.text(0.5, 0.5, 'No loss data available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Training Loss')
    
    # 5. Learning Progress (Recent Performance)
    ax5 = plt.subplot(2, 3, 5)
    recent_window = min(100, len(rewards_history) // 4)
    if recent_window > 0:
        recent_rewards = [np.mean(rewards_history[max(0, i-recent_window):i+1]) 
                         for i in range(len(rewards_history))]
        ax5.plot(recent_rewards, color='green', linewidth=2)
        ax5.set_xlabel('Episode')
        ax5.set_ylabel(f'Average Reward (last {recent_window} eps)')
        ax5.set_title('Learning Progress\n(Rolling Average)')
        ax5.grid(True, alpha=0.3)
    
    # 6. Summary Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate statistics
    final_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
    final_cost = np.mean(costs_history[-100:]) if len(costs_history) >= 100 else np.mean(costs_history)
    max_reward = np.max(rewards_history)
    min_cost = np.min(costs_history)
    
    summary_text = f"""
    C51 Distributional DQN v0.8 Summary
    ═══════════════════════════════════
    
    Training Configuration:
    • Episodes: {episodes}
    • Quantiles: {n_quantiles}
    • Kappa (Huber): {kappa}
    • Quantile Spacing: {1.0/n_quantiles:.4f}
    
    Fleet Configuration:
    • Urban Bridges: {config.get('n_urban', 20)}
    • Rural Bridges: {config.get('n_rural', 80)}
    • Horizon: {config.get('horizon_years', 30)} years
    
    Performance (Last 100 episodes):
    • Avg Reward: {final_reward:.2f}
    • Avg Cost: {final_cost:.2f}k USD
    
    Best Performance:
    • Max Reward: {max_reward:.2f}
    • Min Cost: {min_cost:.2f}k USD
    
    Optimizations:
    ✓ C51 Distributional RL
    ✓ Noisy Networks
    ✓ Dueling DQN
    ✓ Double DQN
    ✓ PER (Prioritized Replay)
    ✓ N-step Learning
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = save_path / "training_curves_v09.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to: {plot_path}")
    
    plt.show()


def visualize_c51_distribution(checkpoint_path: str, state_sample: np.ndarray = None, 
                                save_dir: str = "outputs_markov_v08/plots"):
    """
    Visualize C51 return distributions for a sample state.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        state_sample: Sample state (100-dim). If None, uses a random state.
        save_dir: Directory to save plots
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    # Create agent
    n_bridges = config.get('n_urban', 20) + config.get('n_rural', 80)
    n_atoms = config.get('n_atoms', 51)
    v_min = config.get('v_min', -100.0)
    v_max = config.get('v_max', 100.0)
    
    agent = FleetC51(n_bridges=n_bridges, n_actions=6, 
                     n_atoms=n_atoms, v_min=v_min, v_max=v_max)
    agent.load_state_dict(checkpoint['agent_state_dict'])
    agent.eval()
    
    # Create sample state
    if state_sample is None:
        state_sample = np.random.randint(0, 3, n_bridges).astype(np.float32)
    
    state_t = torch.FloatTensor(state_sample).unsqueeze(0)
    
    # Get Q-values and distributions
    with torch.no_grad():
        q_values, distributions = agent(state_t)
    
    q_values = q_values[0].numpy()  # [n_bridges, n_actions]
    distributions = distributions[0].numpy()  # [n_bridges, n_actions, n_atoms]
    support = agent.support.numpy()
    
    # Create visualization
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Plot distributions for first few bridges
    n_bridges_to_plot = min(4, n_bridges)
    action_names = ['None', 'Work31', 'Work33', 'Work34', 'Work35', 'Work38']
    
    fig, axes = plt.subplots(n_bridges_to_plot, 1, figsize=(12, 3 * n_bridges_to_plot))
    
    if n_bridges_to_plot == 1:
        axes = [axes]
    
    for b_idx in range(n_bridges_to_plot):
        ax = axes[b_idx]
        
        # Plot distribution for each action
        for a_idx in range(6):
            dist = distributions[b_idx, a_idx]
            q_val = q_values[b_idx, a_idx]
            
            ax.plot(support, dist, label=f'{action_names[a_idx]} (Q={q_val:.2f})', 
                   linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Return Value')
        ax.set_ylabel('Probability')
        ax.set_title(f'Bridge {b_idx}: Return Distributions (State={int(state_sample[b_idx])})')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = save_path / "c51_distributions.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"C51 distributions saved to: {plot_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize Markov Fleet QR-DQN v0.9 Results")
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save plots (default: auto-detect from checkpoint path)')
    parser.add_argument('--plot-dist', action='store_true',
                       help='Also plot QR-DQN quantile distributions')
    
    args = parser.parse_args()
    
    # Auto-detect save directory from checkpoint path if not specified
    if args.save_dir is None:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.parent.name == 'models':
            # Checkpoint is in output_dir/models/, use output_dir/plots/
            args.save_dir = str(checkpoint_path.parent.parent / 'plots')
        else:
            # Default fallback
            args.save_dir = 'outputs_markov_v09/plots'
    
    print("\n" + "="*70)
    print("MARKOV FLEET QR-DQN v0.9 VISUALIZATION")
    print("="*70 + "\n")
    
    # Plot training curves
    plot_training_curves_v09(args.checkpoint, args.save_dir)
    
    # Plot QR-DQN distributions if requested
    if args.plot_dist:
        print("\nGenerating QR-DQN distribution plots...")
        visualize_qrdqn_distribution(args.checkpoint, save_dir=args.save_dir)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
