import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print('device: ', self.device)
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PolicyGradientAgent():
    def __init__(self, lr, input_dims, gamma=0.99, n_actions=4):
        self.gamma = gamma
        self.lr = lr
        self.reward_memory = []
        self.action_memory = []

        self.policy = PolicyNetwork(lr=self.lr, input_dims=input_dims,
            n_actions=n_actions)

    # prześledzić wiersz po wierszu co się dzieje by zrozumieć jak działa
    def wybor_akcji(self, observation):
        # obserwację bierzemy w dodatkowe [], ponieważ tensorflow potrzebuje
        # dodatkowy wymiar tensora - rozmiar paczki
        state = T.Tensor([observation]).to(self.policy.device)
        # normalnie sieć wypluwa jakieś prawdopodobieństwa dla każdego neuronu
        # wyjściowego. Funkcja softmax mnoży je tak, by summa wszystkich
        # wynosiła 1.
        prawdopodobienstwa = F.softmax(self.policy.forward(state))
        # chyba przypisuje numer każdemu z możliwych prawdopodobienstw ?
        prawdopodobienstwa_akcji = \
            T.distributions.Categorical(prawdopodobienstwa)
        action = prawdopodobienstwa_akcji.sample()
        log_prawdopodobienstwa = prawdopodobienstwa_akcji.log_prob(action)
        #dadajemy do pamięci
        self.action_memory.append(log_prawdopodobienstwa)
        # item jest chyba po to, by z tensora zrobić coś, co akceptuje gym
        return action.item()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.policy.optimizer.zero_grad()
        # G_t = R_t+1 + gamma * R_t+2 + gamma**2 * R_t+3 ...     lub:
        # G_t = sum from k=0 to k=T {gamma**k + R_t+1+k}
        # z innej beczki możemy olać to float64
        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1    # gamma**k
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        G = T.tensor(G, dtype=T.float).to(self.policy.device)

        loss = 0
        for g, logprob in zip (G, self.action_memory):
            loss += -g * logprob
        loss.backward()
        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []
