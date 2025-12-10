"""
Markov Fleet QR-DQN (Quantile Regression DQN) Training Script (v0.9)

Training 100-bridge fleet with:
- QR-DQN Distributional RL (Dabney et al., AAAI 2018)
- Quantile regression with learnable quantile values
- Quantile Huber loss for robust distributional Bellman update
- Unified Markov transition matrices (municipality-level)
- Urban 20 bridges (higher importance)
- Rural 80 bridges (standard importance)
- Vectorized parallel training with AsyncVectorEnv
- GPU acceleration with Mixed Precision Training (AMP)
- Noisy Networks for Exploration (ICLR 2018) - removes ε-greedy

Based on: v0.8 + QR-DQN "Distributional RL with Quantile Regression" (Dabney et al., AAAI 2018)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import argparse
from collections import deque
from tqdm import tqdm
import time
from torch.amp import autocast, GradScaler
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import yaml
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from markov_fleet_environment import MarkovFleetEnvironment


# ----- Noisy Linear Layer (Factorised Gaussian) -----

class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for exploration (ICLR 2018).
    
    Uses factorised Gaussian noise for efficient parameter-space exploration.
    Replaces ε-greedy exploration with learned stochastic policy.
    
    Paper: Fortunato et al., "Noisy Networks for Exploration" (ICLR 2018)
    """
    
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Noise buffers (not parameters)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize learnable parameters"""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        """Sample new noise for weight and bias"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Factorised Gaussian noise
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        """Generate scaled noise: sign(x) * sqrt(|x|)"""
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return nn.functional.linear(x, weight, bias)


# ----- QR-DQN (Quantile Regression DQN) Network for Fleet Actions -----

class FleetQRDQN(nn.Module):
    """
    QR-DQN (Quantile Regression DQN) with Dueling architecture and Noisy Networks for 100-bridge fleet maintenance.
    
    Based on: "Distributional Reinforcement Learning with Quantile Regression" (Dabney et al., AAAI 2018)
    
    Architecture:
        - Shared network: Input -> [512, 256]
        - Value stream: [256] -> [128] -> [n_quantiles] (Noisy layers)
        - Advantage stream: [256] -> [128] -> [100 * 6 * n_quantiles] (Noisy layers)
    
    Input: 100-dim state vector (discrete states 0-2)
    Output: [100 bridges × 6 actions × n_quantiles] quantile values
    
    Key features: 
        - Learnable quantile values (no fixed support like C51)
        - Quantile Huber loss for robust distributional learning
        - Uses NoisyLinear for automatic exploration without ε-greedy
        - More flexible than C51 (no need to specify V_min/V_max)
    """
    
    def __init__(self, n_bridges: int = 100, n_actions: int = 6, 
                 n_quantiles: int = 200):
        super().__init__()
        self.n_bridges = n_bridges
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        
        # Quantile midpoints (tau) - fixed uniform quantiles
        # τ_i = (i + 0.5) / N for i = 0, 1, ..., N-1
        tau = torch.arange(0, n_quantiles, dtype=torch.float32) + 0.5
        tau = tau / n_quantiles
        self.register_buffer('tau', tau)  # [n_quantiles]
        
        # Shared feature extractor (standard Linear)
        self.shared = nn.Sequential(
            nn.Linear(n_bridges, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        # Value stream (with Noisy layers) - outputs quantile values
        self.value_fc1 = NoisyLinear(256, 128)
        self.value_fc2 = NoisyLinear(128, n_quantiles)
        
        # Advantage stream (with Noisy layers) - outputs quantile values for each action
        self.advantage_fc1 = NoisyLinear(256, 128)
        self.advantage_fc2 = NoisyLinear(128, n_bridges * n_actions * n_quantiles)
    
    def forward(self, x):
        """
        Forward pass with Noisy Networks and QR-DQN quantile output
        
        Args:
            x: State tensor [batch_size, n_bridges]
        
        Returns:
            Tuple of (q_values, quantiles):
                - q_values: Expected Q-values [batch_size, n_bridges, n_actions]
                - quantiles: Quantile values [batch_size, n_bridges, n_actions, n_quantiles]
        """
        batch_size = x.size(0)
        
        # Shared features
        features = self.shared(x)
        
        # Value stream (with NoisyLinear) - outputs quantile values
        value = torch.relu(self.value_fc1(features))
        value_quantiles = self.value_fc2(value)  # [B, n_quantiles]
        
        # Advantage stream (with NoisyLinear) - outputs quantile values
        advantage = torch.relu(self.advantage_fc1(features))
        advantage_quantiles = self.advantage_fc2(advantage)  # [B, n_bridges * n_actions * n_quantiles]
        
        # Reshape advantage
        advantage_quantiles = advantage_quantiles.view(batch_size, self.n_bridges, self.n_actions, self.n_quantiles)
        
        # Dueling architecture for quantiles
        # value_quantiles: [B, n_quantiles] -> [B, 1, 1, n_quantiles] for broadcasting
        value_quantiles = value_quantiles.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, n_quantiles]
        
        # Combine value and advantage quantiles
        # Q_quantiles = V_quantiles + (A_quantiles - mean(A_quantiles))
        advantage_mean = advantage_quantiles.mean(dim=2, keepdim=True)  # [B, n_bridges, 1, n_quantiles]
        quantiles = value_quantiles + (advantage_quantiles - advantage_mean)  # [B, n_bridges, n_actions, n_quantiles]
        
        # Compute expected Q-values from quantiles (mean over quantiles)
        q_values = quantiles.mean(dim=-1)  # [B, n_bridges, n_actions]
        
        return q_values, quantiles
    
    def reset_noise(self):
        """Reset noise for all NoisyLinear layers"""
        self.value_fc1.reset_noise()
        self.value_fc2.reset_noise()
        self.advantage_fc1.reset_noise()
        self.advantage_fc2.reset_noise()


# Keep FleetDQN for reference (commented out)
# class FleetDQN(nn.Module):
#     ... (original v0.7 implementation)


# ----- Prioritized N-Step Replay Buffer -----

class PrioritizedNStepBuffer:
    """Prioritized N-step replay buffer for fleet transitions"""
    
    def __init__(self, capacity: int, n_steps: int = 3, gamma: float = 0.99,
                 alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        self.capacity = capacity
        self.n_steps = n_steps
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.n_step_buffer = deque(maxlen=n_steps)
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def push(self, state, actions, reward, next_state, done):
        """Add experience to n-step buffer"""
        self.n_step_buffer.append((state, actions, reward, next_state, done))
        
        if len(self.n_step_buffer) == self.n_steps or done:
            # Calculate n-step return
            n_step_reward = 0.0
            for i, (_, _, r, _, _) in enumerate(self.n_step_buffer):
                n_step_reward += (self.gamma ** i) * r
            
            # Get first and last transitions
            s0, a0 = self.n_step_buffer[0][:2]
            sn, done_n = self.n_step_buffer[-1][3:5]
            
            experience = (s0, a0, n_step_reward, sn, done_n)
            
            # Assign max priority for new experience
            max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0
            
            if len(self.buffer) < self.capacity:
                self.buffer.append(experience)
            else:
                self.buffer[self.position] = experience
            
            self.priorities[self.position] = max_priority
            self.position = (self.position + 1) % self.capacity
            self.size = len(self.buffer)
            
            if done:
                self.n_step_buffer.clear()
    
    def sample(self, batch_size: int):
        """Sample batch with prioritized sampling"""
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs_sum = probs.sum()
        
        if probs_sum == 0 or np.isnan(probs_sum) or np.isinf(probs_sum):
            probs = np.ones(self.size) / self.size
        else:
            probs /= probs_sum
            probs = np.clip(probs, 1e-8, 1.0)
            probs /= probs.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # Importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = [self.buffer[i] for i in indices]
        
        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.int64)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD-errors"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6
    
    def __len__(self):
        return len(self.buffer)


# ----- QR-DQN Quantile Huber Loss Function -----

def quantile_huber_loss(agent, target_net, s_b_t, a_b_t, r_b_t, sn_b_t, d_b_t, w_b_t, gamma, kappa=1.0, n_steps=3):
    """
    QR-DQN Quantile Huber Loss with distributional Bellman update.
    
    Based on: "Distributional Reinforcement Learning with Quantile Regression" (Dabney et al., AAAI 2018)
    
    Args:
        agent: Current QR-DQN network
        target_net: Target QR-DQN network
        s_b_t: State batch [B, n_bridges]
        a_b_t: Action batch [B, n_bridges]
        r_b_t: Reward batch [B]
        sn_b_t: Next state batch [B, n_bridges]
        d_b_t: Done batch [B]
        w_b_t: Importance sampling weights [B]
        gamma: Discount factor
        kappa: Huber loss threshold (default: 1.0)
        n_steps: N-step returns
    
    Returns:
        loss: Quantile Huber loss
        td_errors: TD errors for PER priority update
    """
    batch_size = s_b_t.size(0)
    n_quantiles = agent.n_quantiles
    tau = agent.tau  # [n_quantiles]
    
    # Get current quantiles: Q(s, a) for taken actions
    _, current_quantiles = agent(s_b_t)  # [B, n_bridges, n_actions, n_quantiles]
    
    # Gather quantiles for taken actions
    # a_b_t: [B, n_bridges] -> [B, n_bridges, 1, 1] for gathering
    a_b_t_expanded = a_b_t.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, n_quantiles)
    
    # current_quantiles: [B, n_bridges, n_actions, n_quantiles]
    # Gather on action dimension (dim=2)
    current_quantiles = current_quantiles.gather(2, a_b_t_expanded).squeeze(2)  # [B, n_bridges, n_quantiles]
    
    # Average over bridges for fleet-level Q-quantiles
    current_quantiles = current_quantiles.mean(dim=1)  # [B, n_quantiles]
    
    with torch.no_grad():
        # Double DQN: select actions with online network
        next_q_values, _ = agent(sn_b_t)  # [B, n_bridges, n_actions]
        next_actions = next_q_values.argmax(dim=2)  # [B, n_bridges]
        
        # Get target quantiles with target network
        _, target_next_quantiles = target_net(sn_b_t)  # [B, n_bridges, n_actions, n_quantiles]
        
        # Gather quantiles for selected actions
        next_actions_expanded = next_actions.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, n_quantiles)
        target_next_quantiles = target_next_quantiles.gather(2, next_actions_expanded).squeeze(2)  # [B, n_bridges, n_quantiles]
        
        # Average over bridges
        target_next_quantiles = target_next_quantiles.mean(dim=1)  # [B, n_quantiles]
        
        # Distributional Bellman update (no projection needed for QR-DQN!)
        # T_θ = r + gamma^n * θ * (1 - done)
        gamma_n = gamma ** n_steps
        
        # Expand rewards for broadcasting
        r_b_t_expanded = r_b_t.unsqueeze(1)  # [B, 1]
        d_b_t_expanded = d_b_t.unsqueeze(1)  # [B, 1]
        
        # Bellman target quantiles
        target_quantiles = r_b_t_expanded + gamma_n * target_next_quantiles * (1 - d_b_t_expanded)  # [B, n_quantiles]
    
    # Quantile Huber Loss (Dabney et al., AAAI 2018)
    # current_quantiles: [B, n_quantiles] -> [B, n_quantiles, 1]
    # target_quantiles: [B, n_quantiles] -> [B, 1, n_quantiles]
    # TD errors: [B, n_quantiles, n_quantiles]
    current_quantiles_expanded = current_quantiles.unsqueeze(2)  # [B, n_quantiles, 1]
    target_quantiles_expanded = target_quantiles.unsqueeze(1)  # [B, 1, n_quantiles]
    
    # Bellman errors: θ_i - T_θ_j for all i, j pairs
    td_errors_matrix = target_quantiles_expanded - current_quantiles_expanded  # [B, n_quantiles, n_quantiles]
    
    # Huber loss: ρ_κ(u) = 0.5 * u^2 if |u| ≤ κ, else κ * (|u| - 0.5 * κ)
    abs_td_errors = td_errors_matrix.abs()  # [B, n_quantiles, n_quantiles]
    huber_loss = torch.where(
        abs_td_errors <= kappa,
        0.5 * td_errors_matrix ** 2,
        kappa * (abs_td_errors - 0.5 * kappa)
    )  # [B, n_quantiles, n_quantiles]
    
    # Quantile regression weights: |τ_i - 1{u < 0}|
    # tau: [n_quantiles] -> [1, n_quantiles, 1]
    tau_expanded = tau.view(1, n_quantiles, 1)  # [1, n_quantiles, 1]
    
    # Indicator: 1{target - current < 0}
    indicator = (td_errors_matrix < 0).float()  # [B, n_quantiles, n_quantiles]
    
    # Quantile weights
    quantile_weights = torch.abs(tau_expanded - indicator)  # [B, n_quantiles, n_quantiles]
    
    # Final quantile Huber loss
    quantile_huber = quantile_weights * huber_loss  # [B, n_quantiles, n_quantiles]
    
    # Sum over target quantiles (j), average over current quantiles (i)
    loss = quantile_huber.sum(dim=2).mean(dim=1)  # [B]
    
    # Weighted loss with importance sampling
    loss = (w_b_t * loss).mean()
    
    # TD errors for PER (use mean absolute TD error as proxy)
    with torch.no_grad():
        td_errors = abs_td_errors.mean(dim=(1, 2))  # [B]
    
    return loss, td_errors


# ----- Vectorized Environment Wrapper -----

def make_env(n_urban: int, n_rural: int, horizon_years: int, cost_lambda: float, seed: int):
    """Create a single environment instance"""
    def _init():
        env = MarkovFleetEnvironment(
            n_urban=n_urban,
            n_rural=n_rural,
            horizon_years=horizon_years,
            cost_lambda=cost_lambda,
            use_onehot=False,
            seed=seed
        )
        return env
    return _init


# ----- Training Function -----

def train_markov_fleet(
    n_episodes: int = 1000,
    n_envs: int = 4,
    n_urban: int = 20,
    n_rural: int = 80,
    horizon_years: int = 30,
    cost_lambda: float = 1e-3,
    gamma: float = 0.95,
    lr: float = 1.5e-3,
    buffer_capacity: int = 10000,
    batch_size: int = 64,
    target_sync_steps: int = 500,
    n_quantiles: int = 200,
    kappa: float = 1.0,
    device: str = 'cuda',
    seed: int = 42,
    output_dir: str = 'outputs_markov',
    verbose: bool = True,
):
    """
    Train QR-DQN (Quantile Regression DQN) agent with Noisy Networks for Markov fleet maintenance.
    
    Args:
        n_episodes: Number of training episodes
        n_envs: Number of parallel environments
        n_urban: Number of urban bridges
        n_rural: Number of rural bridges
        horizon_years: Episode length (years)
        cost_lambda: Cost penalty scaling
        gamma: Discount factor
        lr: Learning rate
        buffer_capacity: Replay buffer size
        batch_size: Batch size for training
        target_sync_steps: Target network update frequency
        n_quantiles: Number of quantiles in QR-DQN (default: 200)
        kappa: Huber loss threshold for QR-DQN (default: 1.0)
        device: 'cuda' or 'cpu'
        seed: Random seed
        output_dir: Output directory
        verbose: Print progress
    
    Returns:
        agent: Trained QR-DQN agent with Noisy Networks
        rewards_history: Episode rewards
        costs_history: Episode costs
    """
    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "models").mkdir(exist_ok=True)
    
    n_bridges = n_urban + n_rural
    
    if verbose:
        print("\n" + "="*80)
        print("MARKOV FLEET QR-DQN TRAINING (v0.9)")
        print("="*80)
        print(f"Fleet Configuration:")
        print(f"  Total Bridges: {n_bridges} (Urban: {n_urban}, Rural: {n_rural})")
        print(f"  Episode Horizon: {horizon_years} years")
        print(f"  Cost Lambda: {cost_lambda}")
        print(f"\nTraining Configuration:")
        print(f"  Episodes: {n_episodes}")
        print(f"  Parallel Envs: {n_envs}")
        print(f"  Device: {device}")
        print(f"  Gamma: {gamma}, LR: {lr}")
        print(f"  Buffer: {buffer_capacity}, Batch: {batch_size}")
        print(f"  Target Sync: {target_sync_steps} steps")
        print(f"\nQR-DQN Configuration (Dabney et al., AAAI 2018):")
        print(f"  N_quantiles: {n_quantiles}")
        print(f"  Kappa (Huber threshold): {kappa}")
        print(f"  Quantile spacing: {1.0/n_quantiles:.4f}")
        print(f"\nOptimizations:")
        print(f"  ✓ QR-DQN Distributional RL (quantile regression)")
        print(f"  ✓ Quantile Huber loss (robust learning)")
        print(f"  ✓ Mixed Precision Training (AMP)")
        print(f"  ✓ Double DQN")
        print(f"  ✓ Dueling DQN")
        print(f"  ✓ N-step Learning (n=3)")
        print(f"  ✓ Prioritized Experience Replay (PER)")
        print(f"  ✓ AsyncVectorEnv ({n_envs}x speedup)")
        print(f"  ✓ Noisy Networks (no ε-greedy needed!)")
        print("="*80 + "\n")
    
    # Create vectorized environments
    env_fns = [make_env(n_urban, n_rural, horizon_years, cost_lambda, seed + i) 
               for i in range(n_envs)]
    envs = AsyncVectorEnv(env_fns)
    
    # Initialize QR-DQN networks
    agent = FleetQRDQN(n_bridges=n_bridges, n_actions=6, 
                       n_quantiles=n_quantiles).to(device)
    target_net = FleetQRDQN(n_bridges=n_bridges, n_actions=6,
                            n_quantiles=n_quantiles).to(device)
    target_net.load_state_dict(agent.state_dict())
    target_net.eval()
    
    optimizer = optim.AdamW(agent.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler('cuda') if device == 'cuda' else None
    buffer = PrioritizedNStepBuffer(
        buffer_capacity, n_steps=3, gamma=gamma,
        alpha=0.6, beta=0.4, beta_increment=0.001
    )
    
    # Training tracking
    rewards_history = []
    costs_history = []
    losses_history = []
    total_steps = 0
    episodes_completed = 0
    start_time = time.time()
    
    # Reset environments
    observations, infos = envs.reset()
    states = observations.astype(np.float32)
    
    # Episode tracking per environment
    env_episode_rewards = np.zeros(n_envs)
    env_episode_costs = np.zeros(n_envs)
    
    pbar = tqdm(total=n_episodes, desc="Training Markov Fleet DQN") if verbose else None
    
    while episodes_completed < n_episodes:
        # Reset noise at the start of each episode for exploration
        # (Noisy Networks provide automatic exploration without ε-greedy)
        for i in range(n_envs):
            if env_episode_rewards[i] == 0:  # Start of new episode
                agent.reset_noise()
                target_net.reset_noise()
        
        # Select actions using C51 noisy network (no ε-greedy needed)
        actions_batch = []
        
        for i in range(n_envs):
            # Actions from QR-DQN noisy network (exploration built-in)
            # Use expected Q-values from quantiles
            with torch.no_grad():
                state_t = torch.FloatTensor(states[i]).unsqueeze(0).to(device)
                q_values, _ = agent(state_t)  # q_values: [1, n_bridges, n_actions]
                q_values = q_values[0]  # [n_bridges, n_actions]
                actions = q_values.argmax(dim=1).cpu().numpy()
            
            actions_batch.append(actions)
        
        actions_batch = np.array(actions_batch)
        
        # Step all environments
        next_observations, rewards, terminateds, truncateds, infos = envs.step(actions_batch)
        next_states = next_observations.astype(np.float32)
        
        # Calculate costs directly from actions (since AsyncVectorEnv doesn't return step info reliably)
        from src.markov_fleet_environment import ACTION_COST_KUSD
        
        # Store transitions and accumulate costs
        for i in range(n_envs):
            buffer.push(
                states[i], actions_batch[i], rewards[i],
                next_states[i], terminateds[i] or truncateds[i]
            )
            
            env_episode_rewards[i] += rewards[i]
            
            # Calculate cost directly from actions taken
            step_cost = np.sum(ACTION_COST_KUSD[actions_batch[i]])
            env_episode_costs[i] += step_cost
            
            # Episode completion
            if terminateds[i] or truncateds[i]:
                rewards_history.append(env_episode_rewards[i])
                costs_history.append(env_episode_costs[i])
                
                env_episode_rewards[i] = 0
                env_episode_costs[i] = 0
                
                episodes_completed += 1
                if pbar:
                    pbar.update(1)
                
                if episodes_completed >= n_episodes:
                    break
        
        states = next_states
        total_steps += n_envs
        
        # Optimization step
        if len(buffer) >= batch_size:
            s_b, a_b, r_b, sn_b, d_b, indices, weights = buffer.sample(batch_size)
            
            # Convert to tensors
            s_b_t = torch.FloatTensor(s_b).to(device)  # [B, n_bridges]
            a_b_t = torch.LongTensor(a_b).to(device)  # [B, n_bridges]
            r_b_t = torch.FloatTensor(r_b).to(device)  # [B]
            sn_b_t = torch.FloatTensor(sn_b).to(device)  # [B, n_bridges]
            d_b_t = torch.FloatTensor(d_b).to(device)  # [B]
            w_b_t = torch.FloatTensor(weights).to(device)  # [B]
            
            # Mixed precision training with QR-DQN quantile Huber loss
            if scaler:
                with autocast('cuda'):
                    # QR-DQN quantile Huber loss
                    loss, td_errors = quantile_huber_loss(
                        agent, target_net, s_b_t, a_b_t, r_b_t, sn_b_t, d_b_t, w_b_t, gamma, kappa=kappa, n_steps=3
                    )
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training with QR-DQN quantile Huber loss
                loss, td_errors = quantile_huber_loss(
                    agent, target_net, s_b_t, a_b_t, r_b_t, sn_b_t, d_b_t, w_b_t, gamma, kappa=kappa, n_steps=3
                )
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Update priorities (using KL divergence as TD error proxy)
            buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
            losses_history.append(loss.item())
        
        # Sync target network
        if total_steps % target_sync_steps == 0:
            target_net.load_state_dict(agent.state_dict())
    
    if pbar:
        pbar.close()
    
    envs.close()
    
    # Training summary
    elapsed_time = time.time() - start_time
    
    if verbose:
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Total Episodes: {episodes_completed}")
        print(f"Total Time: {elapsed_time:.2f} sec ({elapsed_time/60:.2f} min)")
        print(f"Time per Episode: {elapsed_time/episodes_completed:.3f} sec")
        print(f"Final Reward (last 100): {np.mean(rewards_history[-100:]):.2f}")
        print(f"Final Cost (last 100): {np.mean(costs_history[-100:]):.2f}k USD")
        print("="*80 + "\n")
    
    # Save model
    model_path = output_path / "models" / f"markov_fleet_qrdqn_final_{n_episodes}ep.pt"
    torch.save({
        'agent_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episodes': episodes_completed,
        'rewards_history': rewards_history,
        'costs_history': costs_history,
        'losses_history': losses_history,
        'config': {
            'n_urban': n_urban,
            'n_rural': n_rural,
            'horizon_years': horizon_years,
            'cost_lambda': cost_lambda,
            'gamma': gamma,
            'lr': lr,
            'n_quantiles': n_quantiles,
            'kappa': kappa,
        }
    }, model_path)
    
    if verbose:
        print(f"Model saved to: {model_path}")
    
    return agent, rewards_history, costs_history


# ----- Main Entry Point -----

def main():
    parser = argparse.ArgumentParser(description="Train Markov Fleet QR-DQN")
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--n-envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--n-urban', type=int, default=20, help='Number of urban bridges')
    parser.add_argument('--n-rural', type=int, default=80, help='Number of rural bridges')
    parser.add_argument('--horizon', type=int, default=30, help='Episode horizon (years)')
    parser.add_argument('--cost-lambda', type=float, default=1e-3, help='Cost penalty scaling')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--lr', type=float, default=1.5e-3, help='Learning rate')
    parser.add_argument('--buffer-size', type=int, default=10000, help='Replay buffer size')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--target-sync', type=int, default=500, help='Target network sync steps')
    parser.add_argument('--n-quantiles', type=int, default=200, help='QR-DQN number of quantiles')
    parser.add_argument('--kappa', type=float, default=1.0, help='QR-DQN Huber loss threshold')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='outputs_markov', help='Output directory')
    
    args = parser.parse_args()
    
    # Train
    agent, rewards, costs = train_markov_fleet(
        n_episodes=args.episodes,
        n_envs=args.n_envs,
        n_urban=args.n_urban,
        n_rural=args.n_rural,
        horizon_years=args.horizon,
        cost_lambda=args.cost_lambda,
        gamma=args.gamma,
        lr=args.lr,
        buffer_capacity=args.buffer_size,
        batch_size=args.batch_size,
        target_sync_steps=args.target_sync,
        n_quantiles=args.n_quantiles,
        kappa=args.kappa,
        device=args.device,
        seed=args.seed,
        output_dir=args.output,
        verbose=True,
    )
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
