import torch as T
import torch.nn.functional as F
import numpy as np
from networks import  ReplayBuffer, ActorNetwork, CriticNetwork


class Agent():
    # beta to zdaje się learning rate dla krityka
    def __init__(self, alfa, beta, tau, env, input_dims, n_actions=2, fc1_dims=400,
            fc2_dims=300, gamma=0.99, max_buff_size=1000000, batch_size=100,
            warmup=1000, update_actor_interval=2, action_noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alfa = alfa
        self.beta = beta
        self.action_noise = action_noise
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.learn_step_counter = 0
        self.time_step = 0
        self.n_actions = n_actions
        self.update_actor_interval = update_actor_interval
        self.action_noise = action_noise
        self.warmup = warmup

        self.memory = ReplayBuffer(max_len=max_buff_size, input_shape=input_dims,
            n_actions=n_actions)

        self.actor = ActorNetwork(alfa=alfa, input_dims=input_dims,
            fc1_dims=fc1_dims, fc2_dims=fc2_dims, n_actions=n_actions,
            name='actor')
        self.critic1 = CriticNetwork(beta=beta, input_dims=input_dims,
            fc1_dims=fc1_dims, fc2_dims=fc2_dims, n_actions=n_actions,
            name='critic1')
        self.critic2 = CriticNetwork(beta=beta, input_dims=input_dims,
            fc1_dims=fc1_dims, fc2_dims=fc2_dims, n_actions=n_actions,
            name='critic2')
        self.target_actor = ActorNetwork(alfa=alfa, input_dims=input_dims,
            fc1_dims=fc1_dims, fc2_dims=fc2_dims, n_actions=n_actions,
            name='targrt_actor')
        self.target_critic1 = CriticNetwork(beta=beta, input_dims=input_dims,
            fc1_dims=fc1_dims, fc2_dims=fc2_dims, n_actions=n_actions,
            name='target_critic1')
        self.target_critic2 = CriticNetwork(beta=beta, input_dims=input_dims,
            fc1_dims=fc1_dims, fc2_dims=fc2_dims, n_actions=n_actions,
            name='target_critic2')

        # przenosimy parametry z sieci głównych na targetowe
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale=self.action_noise, size=(self.n_actions,))).to(self.actor.device)
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
        # dodajemy szum
        mu_prime = mu + T.tensor(np.random.normal(scale=self.action_noise, size=(self.n_actions,)),
                dtype=T.float).to(self.actor.device)
        # i przycinamy zaszumiony wynik do granic action spejsu
        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step += 1
        # detachujemy by dostać czystą akcję a nie tensor
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        print('zapisuje czekpointy...')
        self.actor.save_checkpoint()
        self.critic1.save_checkpoint()
        self.critic2.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic1.save_checkpoint()
        self.target_critic2.save_checkpoint()

    def load_models(self):
        print('łąduję czekpointy...')
        self.actor.load_checkpoint()
        self.critic1.load_checkpoint()
        self.critic2.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic1.load_checkpoint()
        self.target_critic2.load_checkpoint()


    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, done = \
            self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.critic1.device)
        actions = T.tensor(actions, dtype=T.float).to(self.critic1.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.critic1.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.critic1.device)
        done = T.tensor(done).to(self.critic1.device)

        target_actions = self.target_actor.forward(states_)
        # dodajemy szum
        target_actions = target_actions + T.clamp(T.tensor(
                np.random.normal(scale=0.2)), -0.5, 0.5)
        # obcinamy znów, by wartość nie przekroczyła granic środowiska
        # może się rozwalic jeśli absolutne wartości górnej i dolnej granicy akcji nie są równe
        target_actions = T.clamp(target_actions, self.min_action[0],
                self.max_action[0])

        q1_ = self.target_critic1.forward(states_, target_actions)
        q2_ = self.target_critic2.forward(states_, target_actions)
        q1 = self.critic1.forward(states, actions)
        q2 = self.critic2.forward(states, actions)

        q1_[done] = 0.0
        q2_[done] = 0.0


        # robi płaski, poziomy tensor z pionowego
        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_)

        ygreki_target = rewards + self.gamma*critic_value_
        ygreki_target = ygreki_target.view(self.batch_size, 1)
        # zerowanie gradientów
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()

        q1_loss = F.mse_loss(ygreki_target, q1)
        q2_loss = F.mse_loss(ygreki_target, q2)

        # sumujemy straty, bo zosobna pytorch nie pozwala je backpropagować
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()

        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.learn_step_counter += 1

        # przechodzimy do updatu tylko co któryś krok
        if self.learn_step_counter % self.update_actor_interval != 0:
            return

        # updatujemy aktora
        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic1.forward(states, self.actor.forward(states))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau == None: tau = self.tau
        actor_params = self.actor.named_parameters()
        critic1_params = self.critic1.named_parameters()
        critic2_params = self.critic2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic1_params = self.target_critic1.named_parameters()
        target_critic2_params = self.target_critic2.named_parameters()

        actor_state_dict = dict(actor_params)
        critic1_state_dict = dict(critic1_params)
        critic2_state_dict = dict(critic2_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic1_state_dict = dict(target_critic1_params)
        target_critic2_state_dict = dict(target_critic2_params)

        for name in critic1_state_dict:
            critic1_state_dict[name] = tau*critic1_state_dict[name].clone() + \
                (1-tau)*target_critic1_state_dict[name].clone()

        for name in critic2_state_dict:
            critic2_state_dict[name] = tau*critic2_state_dict[name].clone() + \
                (1-tau)*target_critic2_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic1.load_state_dict(critic1_state_dict)
        self.target_critic2.load_state_dict(critic2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
