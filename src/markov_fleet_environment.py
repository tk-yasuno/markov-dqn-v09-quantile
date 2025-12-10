"""
Markov Decision Process Fleet Environment for 100-Bridge Maintenance

Features:
- Urban: 20 bridges (higher traffic, higher importance)
- Rural: 80 bridges (lower traffic, standard maintenance)
- Unified transition matrix across all bridges (municipality-level estimation)
- Vectorized state transitions for efficient parallel processing
- Gymnasium-compatible interface

Based on: dqn_bridge_maintenance.py from base_markov_state
"""

from typing import Optional, Tuple, Dict, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ----- MDP Constants (NBI-derived, Municipality-level) -----

STATE_NAMES = ["Good", "Fair", "Poor"]  # 0, 1, 2
ACTION_NAMES = ["None", "Work31", "Work33", "Work34", "Work35", "Work38"]

# Transition matrices: P[a][s][s'] = Pr(s'|s,a)
# Single unified transition matrix used across ALL 100 bridges
# Represents municipality-level average transition behavior
TRANSITIONS = {
    0: np.array([  # None (do nothing)
        [0.99, 0.01, 0.00],  # from Good
        [0.00, 0.98, 0.02],  # from Fair
        [0.00, 0.00, 1.00],  # from Poor
    ], dtype=np.float32),
    1: np.array([  # Work 31 (major rehabilitation)
        [0.87, 0.05, 0.08],
        [0.02, 0.97, 0.01],
        [0.21, 0.10, 0.68],
    ], dtype=np.float32),
    2: np.array([  # Work 33 (rehabilitation) - same as None in paper
        [0.99, 0.01, 0.00],
        [0.00, 0.98, 0.02],
        [0.00, 0.00, 1.00],
    ], dtype=np.float32),
    3: np.array([  # Work 34 (deck work)
        [0.87, 0.13, 0.00],
        [0.03, 0.97, 0.00],
        [0.00, 0.00, 1.00],
    ], dtype=np.float32),
    4: np.array([  # Work 35 (deck replacement)
        [0.95, 0.05, 0.00],
        [0.02, 0.97, 0.01],
        [0.06, 0.20, 0.74],
    ], dtype=np.float32),
    5: np.array([  # Work 38 (widening)
        [0.90, 0.08, 0.02],
        [0.02, 0.97, 0.01],
        [0.00, 0.17, 0.83],
    ], dtype=np.float32),
}

# Average action costs (USD thousands)
# Same cost structure applied to all bridges
ACTION_COST_KUSD = np.array([
    0.0,      # None
    2279.69,  # Work 31
    1018.62,  # Work 33
    766.76,   # Work 34
    1005.71,  # Work 35
    3126.26,  # Work 38
], dtype=np.float32)

# Health transition reward table: R[from_state][to_state]
# Municipality-level health valuation
HEALTH_REWARD = np.array([
    [3, -1, -3],  # from Good to Good/Fair/Poor
    [2,  0, -2],  # from Fair to Good/Fair/Poor
    [5,  2, -1],  # from Poor to Good/Fair/Poor
], dtype=np.float32)


# ----- Bridge Type Configuration -----

class BridgeType:
    """Bridge type configuration (Urban vs Rural)"""
    URBAN = "urban"
    RURAL = "rural"
    
    # Importance multipliers for reward calculation
    URBAN_IMPORTANCE = 1.5  # Urban bridges have 1.5x health reward
    RURAL_IMPORTANCE = 1.0  # Rural bridges have standard weight


# ----- Markov Fleet Environment -----

class MarkovFleetEnvironment(gym.Env):
    """
    Gymnasium environment for 100-bridge fleet maintenance with Markov transitions.
    
    Fleet Composition:
        - 20 Urban bridges (indices 0-19): Higher importance
        - 80 Rural bridges (indices 20-99): Standard importance
    
    State Space:
        - 100 bridges × 3 health states = 300-dim state vector (one-hot encoded)
        - Or: 100-dim discrete states (0=Good, 1=Fair, 2=Poor)
    
    Action Space:
        - MultiDiscrete([6] * 100): Each bridge can take 6 actions
    
    Transition Dynamics:
        - Single municipality-level transition matrix applied to ALL bridges
        - Stochastic transitions: P(s'|s,a) from TRANSITIONS
    
    Reward:
        - Health improvement reward (weighted by bridge type)
        - Cost penalty (proportional to action cost)
        - Urban bridges have 1.5x health reward weight
    
    Episode:
        - 30 years (30 timesteps per episode)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        n_urban: int = 20,
        n_rural: int = 80,
        horizon_years: int = 30,
        cost_lambda: float = 1e-3,
        use_onehot: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize fleet environment.
        
        Args:
            n_urban: Number of urban bridges
            n_rural: Number of rural bridges
            horizon_years: Episode length in years
            cost_lambda: Cost penalty scaling factor (per USD thousand)
            use_onehot: If True, state is one-hot encoded (300-dim)
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.n_urban = n_urban
        self.n_rural = n_rural
        self.n_bridges = n_urban + n_rural
        self.horizon_years = horizon_years
        self.cost_lambda = cost_lambda
        self.use_onehot = use_onehot
        
        # Bridge type indices
        self.urban_indices = list(range(n_urban))
        self.rural_indices = list(range(n_urban, self.n_bridges))
        
        # Importance weights (for reward calculation)
        self.importance_weights = np.ones(self.n_bridges, dtype=np.float32)
        self.importance_weights[:n_urban] = BridgeType.URBAN_IMPORTANCE
        self.importance_weights[n_urban:] = BridgeType.RURAL_IMPORTANCE
        
        # Observation space
        if use_onehot:
            # One-hot encoded: 100 bridges × 3 states = 300-dim
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(self.n_bridges * 3,), dtype=np.float32
            )
        else:
            # Discrete states: 100-dim array of {0, 1, 2}
            self.observation_space = spaces.Box(
                low=0, high=2, shape=(self.n_bridges,), dtype=np.int32
            )
        
        # Action space: 100 bridges × 6 actions
        self.action_space = spaces.MultiDiscrete([6] * self.n_bridges)
        
        # Episode state
        self.states = None  # Current bridge states [n_bridges]
        self.t = 0
        
        # Random seed
        if seed is not None:
            self.seed(seed)
    
    def seed(self, seed: int):
        """Set random seed"""
        np.random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed
            options: Optional dict with "init_states" to set initial bridge states
        
        Returns:
            observation: Initial state observation
            info: Diagnostic information
        """
        super().reset(seed=seed)
        
        self.t = 0
        
        # Initialize bridge states
        if options and "init_states" in options:
            self.states = np.array(options["init_states"], dtype=np.int32)
        else:
            # Random initialization: mostly Good (70%), some Fair (25%), few Poor (5%)
            self.states = np.random.choice(
                [0, 1, 2],
                size=self.n_bridges,
                p=[0.70, 0.25, 0.05]
            ).astype(np.int32)
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one year of maintenance actions.
        
        Args:
            actions: Array of action indices [n_bridges]
        
        Returns:
            observation: Next state observation
            reward: Total fleet reward (health - cost)
            terminated: True if episode is done
            truncated: False (not used)
            info: Diagnostic information
        """
        actions = np.asarray(actions, dtype=np.int32)
        assert actions.shape == (self.n_bridges,), f"Expected {self.n_bridges} actions"
        
        # Store previous states for reward calculation
        prev_states = self.states.copy()
        
        # Apply Markov transitions for each bridge
        next_states = np.zeros(self.n_bridges, dtype=np.int32)
        health_rewards = np.zeros(self.n_bridges, dtype=np.float32)
        
        for i in range(self.n_bridges):
            s = self.states[i]
            a = actions[i]
            
            # Sample next state from transition matrix
            transition_probs = TRANSITIONS[a][s]
            # Normalize (handle floating point errors)
            transition_probs = transition_probs / transition_probs.sum()
            next_state = np.random.choice([0, 1, 2], p=transition_probs)
            
            # Health reward for this transition
            health_r = HEALTH_REWARD[s, next_state]
            
            next_states[i] = next_state
            health_rewards[i] = health_r
        
        # Update states
        self.states = next_states
        self.t += 1
        
        # Calculate total reward
        # Health reward (weighted by importance)
        weighted_health_reward = np.sum(health_rewards * self.importance_weights)
        
        # Cost penalty
        action_costs = ACTION_COST_KUSD[actions]
        total_cost = np.sum(action_costs)
        cost_penalty = -self.cost_lambda * total_cost
        
        # Total reward
        reward = float(weighted_health_reward + cost_penalty)
        
        # Episode termination
        terminated = self.t >= self.horizon_years
        truncated = False
        
        # Info dict
        info = self._get_info()
        info.update({
            "year": self.t,
            "health_reward": float(weighted_health_reward),
            "cost_penalty": float(cost_penalty),
            "total_cost_kusd": float(total_cost),
            "actions_taken": self._count_actions(actions),
            "state_distribution": self._get_state_distribution(),
        })
        
        obs = self._get_observation()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        if self.use_onehot:
            # One-hot encode each bridge state
            onehot = np.zeros((self.n_bridges, 3), dtype=np.float32)
            onehot[np.arange(self.n_bridges), self.states] = 1
            return onehot.flatten()
        else:
            return self.states.copy()
    
    def _get_info(self) -> Dict:
        """Get diagnostic information"""
        state_dist = self._get_state_distribution()
        
        return {
            "timestep": self.t,
            "n_good": state_dist["Good"],
            "n_fair": state_dist["Fair"],
            "n_poor": state_dist["Poor"],
            "urban_health": self._get_type_health("urban"),
            "rural_health": self._get_type_health("rural"),
        }
    
    def _get_state_distribution(self) -> Dict[str, int]:
        """Get count of bridges in each state"""
        return {
            "Good": int(np.sum(self.states == 0)),
            "Fair": int(np.sum(self.states == 1)),
            "Poor": int(np.sum(self.states == 2)),
        }
    
    def _get_type_health(self, bridge_type: str) -> float:
        """Get average health score for bridge type"""
        if bridge_type == "urban":
            indices = self.urban_indices
        else:
            indices = self.rural_indices
        
        # Health score: Good=2, Fair=1, Poor=0
        health_scores = 2 - self.states[indices]
        return float(np.mean(health_scores))
    
    def _count_actions(self, actions: np.ndarray) -> Dict[str, int]:
        """Count how many times each action was taken"""
        counts = {}
        for i, name in enumerate(ACTION_NAMES):
            counts[name] = int(np.sum(actions == i))
        return counts
    
    def render(self):
        """Render current state (text-based)"""
        if self.states is None:
            return
        
        print(f"\n=== Year {self.t}/{self.horizon_years} ===")
        
        # Urban bridges
        print(f"\nUrban Bridges (0-{self.n_urban-1}):")
        urban_states = self.states[:self.n_urban]
        urban_dist = {
            "Good": np.sum(urban_states == 0),
            "Fair": np.sum(urban_states == 1),
            "Poor": np.sum(urban_states == 2),
        }
        print(f"  Good: {urban_dist['Good']}, Fair: {urban_dist['Fair']}, Poor: {urban_dist['Poor']}")
        
        # Rural bridges
        print(f"\nRural Bridges ({self.n_urban}-{self.n_bridges-1}):")
        rural_states = self.states[self.n_urban:]
        rural_dist = {
            "Good": np.sum(rural_states == 0),
            "Fair": np.sum(rural_states == 1),
            "Poor": np.sum(rural_states == 2),
        }
        print(f"  Good: {rural_dist['Good']}, Fair: {rural_dist['Fair']}, Poor: {rural_dist['Poor']}")
        
        # Total
        print(f"\nTotal Fleet:")
        total_dist = self._get_state_distribution()
        print(f"  Good: {total_dist['Good']}, Fair: {total_dist['Fair']}, Poor: {total_dist['Poor']}")


# ----- Utility Functions -----

def get_transition_matrix(action: int) -> np.ndarray:
    """Get transition matrix for action"""
    return TRANSITIONS[action].copy()


def get_action_cost(action: int) -> float:
    """Get average cost for action (USD thousands)"""
    return float(ACTION_COST_KUSD[action])


def get_health_reward(from_state: int, to_state: int) -> float:
    """Get health reward for state transition"""
    return float(HEALTH_REWARD[from_state, to_state])


def print_transition_matrix_info():
    """Print transition matrix information"""
    print("=== Markov Transition Matrices (Municipality-level) ===\n")
    
    for action_idx, action_name in enumerate(ACTION_NAMES):
        print(f"Action {action_idx}: {action_name}")
        print(f"Cost: ${ACTION_COST_KUSD[action_idx]:.2f}k")
        print(f"Transition Matrix P[s][s']:")
        trans = TRANSITIONS[action_idx]
        print(f"         Good   Fair   Poor")
        for s, state_name in enumerate(STATE_NAMES):
            print(f"  {state_name:4s}  {trans[s, 0]:.2f}  {trans[s, 1]:.2f}  {trans[s, 2]:.2f}")
        print()


# ----- Test Code -----

if __name__ == "__main__":
    print("Testing MarkovFleetEnvironment...\n")
    
    # Print transition matrices
    print_transition_matrix_info()
    
    # Create environment
    env = MarkovFleetEnvironment(n_urban=20, n_rural=80, seed=42)
    
    print(f"Environment created:")
    print(f"  Total bridges: {env.n_bridges}")
    print(f"  Urban: {env.n_urban}")
    print(f"  Rural: {env.n_rural}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Test reset
    obs, info = env.reset(seed=42)
    print(f"\nInitial state:")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Good: {info['n_good']}, Fair: {info['n_fair']}, Poor: {info['n_poor']}")
    
    # Test step with random actions
    actions = np.zeros(env.n_bridges, dtype=np.int32)  # Do nothing
    obs, reward, terminated, truncated, info = env.step(actions)
    
    print(f"\nAfter step (all 'None' actions):")
    print(f"  Reward: {reward:.2f}")
    print(f"  Good: {info['n_good']}, Fair: {info['n_fair']}, Poor: {info['n_poor']}")
    print(f"  Terminated: {terminated}")
    
    print("\n✓ MarkovFleetEnvironment test passed!")
