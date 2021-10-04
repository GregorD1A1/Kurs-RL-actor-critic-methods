import gym
import matplotlib.pyplot as plt
import numpy as np
from landerGrad import PolicyGradientAgent


def plot_learning_curve(scores, x, plik_wyjsciowy):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Średnia z poprzednich 100 punktów')
    plt.savefig(plik_wyjsciowy)

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    n_games = 3001
    agent = PolicyGradientAgent(gamma=0.99, lr = 0.0005, input_dims=[8],
        n_actions = 4)
    fname = 'REINFORCE' + 'LunarLander-v2' + 'lr=' + str(agent.lr) + \
        str(n_games) + 'games'
    plik_wyjsciowy = 'plots/' + fname + '.png'

    scores = []
    for epizode_number in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.wybor_akcji(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_rewards(reward)
            observation = observation_
            if epizode_number % 100 == 0:
                env.render()
        agent.learn()
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('epizod', epizode_number, ', punkty: ', score,
            ', punkty uśrednione: ', avg_score)

x = [i+1 for i in range(len(scores))]
plot_learning_curve(scores, x, plik_wyjsciowy)
