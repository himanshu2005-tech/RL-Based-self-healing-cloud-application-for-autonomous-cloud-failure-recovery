import numpy as np
import pickle

class QLearningAgent:
    """
    Tabular Q-Learning Agent.
    Discretizes the continuous state space into bins.
    """
    def __init__(self, action_space_size=5, num_bins=5, learning_rate=0.1, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 bins=None, feature_idx=None):
        self.action_space_size = action_space_size
        self.num_bins = num_bins
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # State boundaries for discretization. `bins` (one array of interior edges per
        # feature) and `feature_idx` (which observation entries to use) override the
        # v1 defaults below.
        self.feature_idx = feature_idx
        self.bins = bins if bins is not None else [
            np.linspace(0.0, 1.0, num_bins - 1),   # CPU
            np.linspace(0.0, 1.0, num_bins - 1),   # RAM
            np.linspace(0.0, 10.0, num_bins - 1),  # Response Time
            np.linspace(0.0, 1.0, num_bins - 1),   # Error Rate
            np.linspace(0.0, 1.0, num_bins - 1)    # Request Load
        ]
        
        # Q-table shape: one axis per feature (len(edges) + 1 bins each), then actions
        self.q_table = np.zeros([len(b) + 1 for b in self.bins] + [action_space_size])
        
    def _discretize_state(self, state):
        if self.feature_idx is not None:
            state = [state[i] for i in self.feature_idx]
        discretized = []
        for i, val in enumerate(state):
            idx = np.digitize(val, self.bins[i])
            discretized.append(idx)
        return tuple(discretized)

    def act(self, state, evaluate=False):
        state_idx = self._discretize_state(state)
        
        if not evaluate and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space_size)
        
        return np.argmax(self.q_table[state_idx])

    def update(self, state, action, reward, next_state, terminated=False):
        state_idx = self._discretize_state(state)
        next_state_idx = self._discretize_state(next_state)
        
        best_next_action = np.argmax(self.q_table[next_state_idx])
        bootstrap = 0.0 if terminated else self.gamma * self.q_table[next_state_idx][best_next_action]
        td_target = reward + bootstrap
        td_error = td_target - self.q_table[state_idx][action]
        
        self.q_table[state_idx][action] += self.lr * td_error
        
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon
            }, f)
            
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.epsilon = data['epsilon']
