import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)
        self.v = nn.Linear(fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print('device: ', self.device)
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)
        return (pi, v)


class ActorCriticAgent():
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions,
            gamma=0.99):
        self.gamma = gamma
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.acnetwork = ActorCriticNetwork(lr=lr, input_dims=input_dims,
            n_actions=n_actions, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.ln_prob = None

    def choose_action(self, observation):
        # obserwację bierzemy w dodatkowe [], ponieważ tensorflow potrzebuje
        # dodatkowy wymiar tensora - rozmiar paczki
        state = T.tensor([observation], dtype=T.float).to(self.acnetwork.device)
        # normalnie sieć wypluwa jakieś prawdopodobieństwa dla każdego neuronu
        # wyjściowego. Funkcja softmax mnoży je tak, by summa wszystkich
        # wynosiła 1.
        prawdopodobienstwa, _ = self.acnetwork.forward(state)
        prawdopodobienstwa = F.softmax(prawdopodobienstwa, dim=1)
        # chyba przypisuje numer każdemu z możliwych prawdopodobienstw ?
        prawdopodobienstwa_akcji = \
            T.distributions.Categorical(prawdopodobienstwa)
        action = prawdopodobienstwa_akcji.sample()
        ln_prawdopodobienstwa = prawdopodobienstwa_akcji.log_prob(action)
        self.ln_prob = ln_prawdopodobienstwa
        # item jest chyba po to, by z tensora zrobić coś, co akceptuje gym
        return action.item()

    def learn(self, state, reward, state_, done):
        # zerujemy gradienty w optimizerze
        self.acnetwork.optimizer.zero_grad()
        # przeganiamy do postaci tensoró
        state = T.tensor([state], dtype=T.float).to(self.acnetwork.device)
        state_ = T.tensor([state_], dtype=T.float).to(self.acnetwork.device)
        reward = T.tensor([reward], dtype=T.float).to(self.acnetwork.device)
        _, critic_value = self.acnetwork.forward(state)
        _, critic_value_ = self.acnetwork.forward(state_)

        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value
        actor_loss = -self.ln_prob*delta
        critic_loss = delta**2
        (actor_loss + critic_loss).backward()
        self.acnetwork.optimizer.step()
