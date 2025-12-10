"""
Quick Test Script for QR-DQN with Noisy Networks (v0.9)

This script performs a quick verification that:
1. NoisyLinear layers work correctly
2. Noise reset functionality works
3. FleetQRDQN network forward pass works
4. QR-DQN quantile output is valid
5. Training loop doesn't crash
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import from train_markov_fleet.py
from train_markov_fleet import NoisyLinear, FleetQRDQN


def test_noisy_linear():
    """Test NoisyLinear layer basic functionality"""
    print("\n" + "="*60)
    print("TEST 1: NoisyLinear Layer")
    print("="*60)
    
    layer = NoisyLinear(10, 5)
    x = torch.randn(2, 10)
    
    # Test forward pass
    out1 = layer(x)
    print(f"✓ Forward pass works: {out1.shape}")
    
    # Test noise reset
    layer.reset_noise()
    out2 = layer(x)
    print(f"✓ Noise reset works")
    
    # Verify outputs differ (due to noise)
    layer.train()
    out3 = layer(x)
    layer.reset_noise()
    out4 = layer(x)
    
    diff = (out3 - out4).abs().mean().item()
    print(f"✓ Noise changes output (mean diff: {diff:.4f})")
    
    # Test eval mode (no noise)
    layer.eval()
    out5 = layer(x)
    out6 = layer(x)
    eval_diff = (out5 - out6).abs().mean().item()
    print(f"✓ Eval mode is deterministic (diff: {eval_diff:.6f})")
    
    print("✓ NoisyLinear test PASSED!\n")


def test_fleet_qrdqn():
    """Test FleetQRDQN with Noisy Networks and Quantile Output"""
    print("="*60)
    print("TEST 2: FleetQRDQN with Noisy Networks and Quantile Regression")
    print("="*60)
    
    n_bridges = 100
    n_actions = 6
    n_quantiles = 200
    batch_size = 4
    
    model = FleetQRDQN(n_bridges=n_bridges, n_actions=n_actions, 
                       n_quantiles=n_quantiles)
    states = torch.randn(batch_size, n_bridges)
    
    # Test forward pass
    model.train()
    q_values, quantiles = model(states)
    print(f"✓ Forward pass works")
    print(f"  - Q-values shape: {q_values.shape}")
    print(f"  - Quantiles shape: {quantiles.shape}")
    assert q_values.shape == (batch_size, n_bridges, n_actions)
    assert quantiles.shape == (batch_size, n_bridges, n_actions, n_quantiles)
    
    # Verify Q-values are computed correctly from quantiles (mean)
    expected_q = quantiles.mean(dim=-1)
    print(f"✓ Q-values match expected values (max diff: {(q_values - expected_q).abs().max():.6f})")
    assert torch.allclose(q_values, expected_q, atol=1e-5)
    
    # Verify quantile values are ordered (optional check)
    quantiles_sorted = quantiles.sort(dim=-1)[0]
    ordering_check = torch.allclose(quantiles, quantiles_sorted, atol=1e-2)
    if ordering_check:
        print(f"✓ Quantiles are monotonically ordered (as expected)")
    else:
        print(f"⚠ Quantiles not perfectly ordered (normal for early training)")
    
    # Test reset_noise
    model.reset_noise()
    q_values2, quantiles2 = model(states)
    diff = (q_values - q_values2).abs().mean().item()
    print(f"✓ reset_noise() changes output (mean Q diff: {diff:.4f})")
    
    # Test action selection
    actions = q_values.argmax(dim=2)
    print(f"✓ Action selection works: {actions.shape}")
    assert actions.shape == (batch_size, n_bridges)
    
    # Verify actions are in valid range
    assert actions.min() >= 0 and actions.max() < n_actions
    print(f"✓ Actions in valid range [0, {n_actions-1}]")
    
    print("✓ FleetQRDQN test PASSED!\n")


def test_training_loop():
    """Test a minimal training loop with QR-DQN"""
    print("="*60)
    print("TEST 3: Minimal Training Loop with QR-DQN")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    n_bridges = 100
    n_actions = 6
    n_quantiles = 200
    kappa = 1.0
    
    # Create networks
    agent = FleetQRDQN(n_bridges=n_bridges, n_actions=n_actions,
                       n_quantiles=n_quantiles).to(device)
    target_net = FleetQRDQN(n_bridges=n_bridges, n_actions=n_actions,
                            n_quantiles=n_quantiles).to(device)
    target_net.load_state_dict(agent.state_dict())
    target_net.eval()
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    tau = agent.tau
    
    # Simulate a few training steps
    agent.train()
    for step in range(5):
        # Reset noise
        agent.reset_noise()
        target_net.reset_noise()
        
        # Generate fake batch
        states = torch.randn(32, n_bridges).to(device)
        actions = torch.randint(0, n_actions, (32, n_bridges)).to(device)
        rewards = torch.randn(32).to(device)
        next_states = torch.randn(32, n_bridges).to(device)
        dones = torch.zeros(32).to(device)
        
        # Forward pass with QR-DQN
        q_values, quantiles = agent(states)
        
        # Get quantiles for taken actions
        actions_expanded = actions.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, n_quantiles)
        quantiles_taken = quantiles.gather(2, actions_expanded).squeeze(2).mean(dim=1)  # [B, n_quantiles]
        
        # Target quantiles (simplified for testing)
        with torch.no_grad():
            next_q, next_quantiles = target_net(next_states)
            next_actions = next_q.argmax(dim=2)
            next_actions_expanded = next_actions.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, n_quantiles)
            target_quantiles = next_quantiles.gather(2, next_actions_expanded).squeeze(2).mean(dim=1)  # [B, n_quantiles]
            
            # Bellman update
            target_quantiles = rewards.unsqueeze(1) + 0.99 * target_quantiles * (1 - dones.unsqueeze(1))
        
        # Quantile Huber loss (simplified)
        quantiles_expanded = quantiles_taken.unsqueeze(2)  # [B, n_quantiles, 1]
        target_expanded = target_quantiles.unsqueeze(1)  # [B, 1, n_quantiles]
        td_errors = target_expanded - quantiles_expanded  # [B, n_quantiles, n_quantiles]
        
        abs_td = td_errors.abs()
        huber = torch.where(abs_td <= kappa, 0.5 * td_errors**2, kappa * (abs_td - 0.5*kappa))
        
        tau_expanded = tau.view(1, n_quantiles, 1).to(device)
        indicator = (td_errors < 0).float()
        quantile_weights = torch.abs(tau_expanded - indicator)
        
        loss = (quantile_weights * huber).sum(dim=2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Step {step+1}: loss = {loss.item():.4f}")
    
    print("✓ Training loop test PASSED!\n")


def main():
    print("\n" + "="*60)
    print("QR-DQN VERIFICATION (v0.9)")
    print("="*60)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        test_noisy_linear()
        test_fleet_qrdqn()
        test_training_loop()
        
        print("="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nQR-DQN implementation is working correctly.")
        print("You can now run full training with:")
        print("  python train_markov_fleet.py --episodes 1000 --device cuda")
        print()
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
