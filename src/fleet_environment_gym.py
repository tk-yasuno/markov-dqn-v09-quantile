"""
Gymnasium-compatible Multi-Bridge Fleet Management Environment
Phase 3: Vectorization-ready implementation
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, Any
from dataclasses import dataclass
import sys
from pathlib import Path

# Import existing components
sys.path.insert(0, str(Path(__file__).parent))
from fleet_environment_v05 import (
    FleetConfig, BridgeProfile, BridgeCluster,
    FLEET_ACTIONS, RURAL_STRATEGIES,
    UrbanAgentDQN, RuralAgentDQN
)


class FleetManagementGym(gym.Env):
    """
    Gymnasium-compatible Fleet Management Environment
    
    Manages 100 bridges (20 urban + 80 rural) with dual-agent system.
    Compatible with vectorized environments for parallelization.
    """
    
    metadata = {
        'render_modes': ['human', 'ansi'],
        'render_fps': 1
    }
    
    def __init__(self, config: Optional[FleetConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        
        self.cfg = config if config is not None else FleetConfig()
        self.render_mode = render_mode
        
        # Define action spaces
        # Urban: 20 bridges, each with 5 actions (0-4)
        # Rural: 1 strategy selection (0-7)
        self.action_space = spaces.Dict({
            'urban': spaces.MultiDiscrete([5] * self.cfg.n_urban_bridges),
            'rural': spaces.Discrete(8)
        })
        
        # Define observation spaces
        # Urban: 20 bridges × 4 features + 1 budget = 81D
        # Rural: 9 statistics + 1 budget = 10D
        self.observation_space = spaces.Dict({
            'urban': spaces.Box(
                low=0.0, high=100.0,
                shape=(self.cfg.n_urban_bridges * 4 + 1,),
                dtype=np.float32
            ),
            'rural': spaces.Box(
                low=0.0, high=100.0,
                shape=(10,),
                dtype=np.float32
            )
        })
        
        # Initialize bridge fleet
        self._initialize_fleet()
        
        # Episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        
    def _initialize_fleet(self):
        """Initialize bridge profiles with diverse ages (20-50 years)"""
        np.random.seed(self.cfg.seed)
        
        self.bridge_profiles = {}
        self.bridge_states = {}
        
        # Urban bridges (20)
        for i in range(self.cfg.n_urban_bridges):
            age = np.random.randint(20, 51)
            initial_condition = max(3, 10 - (age - 20) // 5)
            
            profile = BridgeProfile(
                bridge_id=i,
                cluster=BridgeCluster.URBAN,
                deck_area=np.random.uniform(800, 1200),
                traffic_aadt=np.random.randint(15000, 25000),
                age=age,
                target_condition=self.cfg.urban_target_condition,
                maintenance_cost_factor=np.random.uniform(0.9, 1.1),
                closure_impact=np.random.uniform(150, 250),
                initial_age=age
            )
            
            self.bridge_profiles[i] = profile
            # State: [condition, age, years_since_maintenance, deterioration_rate]
            self.bridge_states[i] = np.array([
                initial_condition, age, 0, 0.3
            ], dtype=np.float32)
        
        # Rural bridges (80)
        for i in range(self.cfg.n_urban_bridges, 
                      self.cfg.n_urban_bridges + self.cfg.n_rural_bridges):
            age = np.random.randint(25, 51)
            initial_condition = max(4, 10 - (age - 25) // 6)
            
            profile = BridgeProfile(
                bridge_id=i,
                cluster=BridgeCluster.RURAL,
                deck_area=np.random.uniform(300, 600),
                traffic_aadt=np.random.randint(500, 3000),
                age=age,
                target_condition=self.cfg.rural_target_condition,
                maintenance_cost_factor=np.random.uniform(0.8, 1.0),
                closure_impact=np.random.uniform(30, 60),
                initial_age=age
            )
            
            self.bridge_profiles[i] = profile
            self.bridge_states[i] = np.array([
                initial_condition, age, 0, 0.25
            ], dtype=np.float32)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset environment to initial state
        
        Returns:
            observation (dict): {'urban': np.array, 'rural': np.array}
            info (dict): Additional information
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.cfg.seed = seed
        
        # Reinitialize fleet
        self._initialize_fleet()
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.urban_budget = self.cfg.total_annual_budget * self.cfg.urban_budget_share
        self.rural_budget = self.cfg.total_annual_budget * self.cfg.rural_budget_share
        
        # Get initial observations
        urban_obs = self._get_urban_observation()
        rural_obs = self._get_rural_observation()
        
        observation = {
            'urban': urban_obs,
            'rural': rural_obs
        }
        
        info = {
            'episode': 0,
            'step': 0,
            'urban_bridges': self.cfg.n_urban_bridges,
            'rural_bridges': self.cfg.n_rural_bridges
        }
        
        return observation, info
    
    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one time step
        
        Args:
            action: {'urban': np.array of 20 actions, 'rural': int strategy}
        
        Returns:
            observation (dict): Next state
            reward (float): Step reward
            terminated (bool): Episode ended naturally
            truncated (bool): Episode ended by time limit
            info (dict): Additional information
        """
        urban_actions = action['urban']
        rural_action = int(action['rural'])
        
        # Execute maintenance actions
        urban_cost, urban_reward = self._execute_urban_actions(urban_actions)
        rural_cost, rural_reward = self._execute_rural_strategy(rural_action)
        
        # Age bridges and apply deterioration
        self._age_bridges()
        
        # Calculate total reward
        total_cost = urban_cost + rural_cost
        cost_penalty = -self.cfg.cost_lambda * total_cost
        
        # Municipality reward (cooperative)
        municipality_reward = urban_reward + rural_reward + cost_penalty
        
        # Cooperative bonus
        if urban_reward > 0 and rural_reward > 0:
            municipality_reward *= (1 + self.cfg.cooperative_bonus)
        
        self.episode_reward += municipality_reward
        self.current_step += 1
        
        # Check termination
        terminated = (self.current_step >= self.cfg.horizon_years)
        truncated = False
        
        # Get next observations
        urban_obs = self._get_urban_observation()
        rural_obs = self._get_rural_observation()
        
        observation = {
            'urban': urban_obs,
            'rural': rural_obs
        }
        
        info = {
            'episode': 0,
            'step': self.current_step,
            'urban_cost': urban_cost,
            'rural_cost': rural_cost,
            'total_cost': total_cost,
            'urban_reward': urban_reward,
            'rural_reward': rural_reward,
            'municipality_reward': municipality_reward,
            'urban_spent': urban_cost,
            'rural_spent': rural_cost
        }
        
        return observation, municipality_reward, terminated, truncated, info
    
    def _get_urban_observation(self) -> np.ndarray:
        """Get urban agent observation (81D)"""
        obs = []
        for i in range(self.cfg.n_urban_bridges):
            obs.extend(self.bridge_states[i])
        obs.append(self.urban_budget / 1000.0)  # Normalized budget
        return np.array(obs, dtype=np.float32)
    
    def _get_rural_observation(self) -> np.ndarray:
        """Get rural agent observation (10D)"""
        rural_states = [
            self.bridge_states[i] for i in range(
                self.cfg.n_urban_bridges,
                self.cfg.n_urban_bridges + self.cfg.n_rural_bridges
            )
        ]
        
        conditions = [s[0] for s in rural_states]
        ages = [s[1] for s in rural_states]
        
        obs = [
            np.mean(conditions),
            np.min(conditions),
            np.max(conditions),
            np.std(conditions),
            np.mean(ages),
            np.sum([1 for c in conditions if c < 6]),  # Critical count
            np.sum([1 for c in conditions if c >= 8]),  # Good count
            np.mean([s[2] for s in rural_states]),  # Avg years since maintenance
            np.mean([s[3] for s in rural_states]),  # Avg deterioration rate
            self.rural_budget / 1000.0  # Normalized budget
        ]
        
        return np.array(obs, dtype=np.float32)
    
    def _execute_urban_actions(self, actions: np.ndarray) -> Tuple[float, float]:
        """Execute urban maintenance actions"""
        total_cost = 0.0
        rewards = []
        
        for i, action in enumerate(actions):
            bridge = self.bridge_profiles[i]
            state = self.bridge_states[i]
            
            action_info = FLEET_ACTIONS[int(action)]
            cost = action_info['cost_base'] * bridge.maintenance_cost_factor
            
            # Apply action effects
            if action_info['effect'] == 'light':
                state[0] = min(10, state[0] + 1)
                state[2] = 0
            elif action_info['effect'] == 'medium':
                state[0] = min(10, state[0] + 2)
                state[2] = 0
            elif action_info['effect'] == 'major':
                state[0] = min(10, state[0] + 4)
                state[2] = 0
            elif action_info['effect'] == 'replace':
                state[0] = 10
                state[1] = 0
                state[2] = 0
            
            total_cost += cost
            
            # Reward calculation
            condition = state[0]
            target = bridge.target_condition
            
            if condition >= target:
                reward = (condition - target) * 10
            elif condition >= 6:
                reward = (condition - 6) * 5
            else:
                reward = -self.cfg.urban_critical_penalty * (6 - condition)
            
            rewards.append(reward)
        
        total_reward = sum(rewards)
        return total_cost, total_reward
    
    def _execute_rural_strategy(self, strategy: int) -> Tuple[float, float]:
        """Execute rural maintenance strategy"""
        rural_bridges = range(
            self.cfg.n_urban_bridges,
            self.cfg.n_urban_bridges + self.cfg.n_rural_bridges
        )
        
        total_cost = 0.0
        rewards = []
        
        # Simplified strategy execution
        for bridge_id in rural_bridges:
            state = self.bridge_states[bridge_id]
            bridge = self.bridge_profiles[bridge_id]
            
            # Decide action based on strategy and condition
            action = self._rural_strategy_to_action(strategy, state[0])
            
            if action > 0:
                action_info = FLEET_ACTIONS[action]
                cost = action_info['cost_base'] * bridge.maintenance_cost_factor * 0.8
                total_cost += cost
                
                # Apply effects (simplified)
                if action_info['effect'] != 'none':
                    state[0] = min(10, state[0] + action)
                    state[2] = 0
            
            # Reward
            condition = state[0]
            if condition >= 6:
                reward = (condition - 6) * 3
            else:
                reward = -5 * (6 - condition)
            
            rewards.append(reward)
        
        total_reward = sum(rewards)
        return total_cost, total_reward
    
    def _rural_strategy_to_action(self, strategy: int, condition: float) -> int:
        """Map strategy to action based on condition"""
        if condition < 5:
            return 3  # Rehabilitation
        elif condition < 6:
            return 2  # Major repair
        elif condition < 7:
            return 1 if strategy >= 4 else 0  # Minor or none
        else:
            return 0  # No action
    
    def _age_bridges(self):
        """Age all bridges and apply deterioration"""
        for bridge_id, state in self.bridge_states.items():
            state[1] += 1  # Increment age
            state[2] += 1  # Years since maintenance
            
            # Deterioration
            deterioration = state[3] * (1 + state[2] * 0.1)
            state[0] = max(0, state[0] - deterioration)
    
    def render(self):
        """Render environment state"""
        if self.render_mode == 'ansi':
            output = f"Step: {self.current_step}/{self.cfg.horizon_years}\n"
            output += f"Episode Reward: {self.episode_reward:.2f}\n"
            return output
        elif self.render_mode == 'human':
            print(f"Step: {self.current_step}/{self.cfg.horizon_years}")
            print(f"Episode Reward: {self.episode_reward:.2f}")


# Export components
__all__ = [
    'FleetManagementGym',
    'FleetConfig',
    'UrbanAgentDQN',
    'RuralAgentDQN',
    'FLEET_ACTIONS',
    'RURAL_STRATEGIES'
]
