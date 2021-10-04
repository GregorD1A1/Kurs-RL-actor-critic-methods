import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork


class Agent():
    def __init__(self, alfa, beta, tau, env, input_dims, env_name, n_actions=2, fc1_dims=256,
            fc2_dims=256, gamma=0.99, max_buff_size=1000000, batch_size=100,
            reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_len=max_buff_size, input_shape=input_dims,
            n_actions=n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions



        self.actor = ActorNetwork(alfa=alfa, input_dims=input_dims,
            fc1_dims=fc1_dims, fc2_dims=fc2_dims, n_actions=n_actions,
            name=env_name+'_actor', max_action=env.action_space.high)
        self.critic1 = CriticNetwork(beta=beta, input_dims=input_dims,
            fc1_dims=fc1_dims, fc2_dims=fc2_dims, n_actions=n_actions,
            name=env_name+'_critic1')
        self.critic2 = CriticNetwork(beta=beta, input_dims=input_dims,
            fc1_dims=fc1_dims, fc2_dims=fc2_dims, n_actions=n_actions,
            name=env_name+'_critic2')
        self.value = ValueNetwork(beta=beta, input_dims=input_dims,
            fc1_dims=fc1_dims, fc2_dims=fc2_dims,
            name=env_name+'_value')
        self.target_value = ValueNetwork(beta=beta, input_dims=input_dims,
            fc1_dims=fc1_dims, fc2_dims=fc2_dims,
            name=env_name+'_target_value')

        self.scale = reward_scale

        # przenosimy parametry z sieci głównych na targetowe
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        print('zapisuje czekpointy...')
        self.actor.save_checkpoint()
        self.critic1.save_checkpoint()
        self.critic2.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()

    def load_models(self):
        print('ładuję czekpointy...')
        self.actor.load_checkpoint()
        self.critic1.load_checkpoint()
        self.critic2.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()


    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, state_, done = \
            self.memory.sample_buffer(self.batch_size)

        state = T.tensor(state, dtype=T.float).to(self.critic1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic1.device)
        state_ = T.tensor(state_, dtype=T.float).to(self.critic1.device)
        reward = T.tensor(reward, dtype=T.float).to(self.critic1.device)
        done = T.tensor(done).to(self.critic1.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic1.forward(state, actions)
        q2_new_policy = self.critic2.forward(state, actions)

        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        # zerowanie gradientu
        self.value.optimizer.zero_grad()

        value_target = critic_value - log_probs
        value_loss = 0.5 * (F.mse_loss(value, value_target))
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # a teraz to wszystko z reparametryzacją
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic1.forward(state, actions)
        q2_new_policy = self.critic2.forward(state, actions)

        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        # updatujemy aktora
        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic1.forward(state, action).view(-1)
        q2_old_policy = self.critic2.forward(state, action).view(-1)
        critic1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic1_loss + critic2_loss
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.update_network_parameters()


    def update_network_parameters(self, tau=None):
        if tau is None: tau = self.tau
        value_params = self.value.named_parameters()
        target_value_params = self.target_value.named_parameters()

        value_state_dict = dict(value_params)
        target_value_state_dict = dict(target_value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)
