import gym
import numpy as np
from landerAC import ActorCriticAgent
from utils import plot_learning_curve


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = ActorCriticAgent(gamma=0.99, lr = 5e-6, input_dims=[8],
        n_actions = 4, fc1_dims=2048, fc2_dims=1536)
    n_games = 3001
    fname = 'ACTOR_CRITIC' + 'LunarLander-v2' + 'lr=' + str(agent.lr) + \
        str(n_games) + 'games' + 'fc1_dims' + str(agent.fc1_dims) + \
        'fc2_dims:' + str(agent.fc2_dims)
    plik_wyjsciowy = 'plots/' + fname + '.png'

    scores = []
    for epizode_number in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.learn(observation, reward, observation_, done)
            observation = observation_
            if epizode_number % 100 == 0:
                env.render()
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('epizod', epizode_number, ', punkty: ', score,
            ', punkty uśrednione: ', avg_score)

x = [i+1 for i in range(len(scores))]
plot_learning_curve(scores, x, plik_wyjsciowy)
