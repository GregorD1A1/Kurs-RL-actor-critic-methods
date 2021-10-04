import numpy as np


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
