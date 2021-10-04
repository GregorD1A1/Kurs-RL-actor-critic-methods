import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer():
    def __init__(self, max_len, input_shape, n_actions):
        self.buf_len = max_len
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.buf_len, *input_shape))
        self.new_state_memory = np.zeros((self.buf_len, *input_shape))
        self.action_memory = np.zeros((self.buf_len, n_actions))
        self.reward_memory = np.zeros(self.buf_len)
        self.terminal_memory = np.zeros(self.buf_len, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.buf_len
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.buf_len)
        batch_indeces = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch_indeces]
        new_states = self.new_state_memory[batch_indeces]
        actions = self.action_memory[batch_indeces]
        rewards = self.reward_memory[batch_indeces]
        dones = self.terminal_memory[batch_indeces]

        return states, actions, rewards, new_states, dones


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, name, chkpt_dir = 'checkpointy',
        fc1_dims=400, fc2_dims=300):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        # czy wiersz poniżej jest w ogóle potrzebny?
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        # niby powinno się rozwalić w przypadku 2D środowiska
        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        q1_action_value = self.fc1(T.cat([state, action], dim=1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)
        q1 = self.q1(q1_action_value)
        return q1

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alfa, input_dims, fc1_dims, fc2_dims, n_actions, name,
            chkpt_dir = 'checkpointy'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        # czy wiersz poniżej jest w ogóle potrzebny?
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ddpg')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alfa)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        prob = T.tanh(self.mu(prob))    # jeśli akcja > +-1, pomnożyć razy max_akcję
        return prob

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
