import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve


if __name__ == '__main__':
    env = gym.make('BipedalWalker-v3')
    agent = Agent(alfa=0.001, beta=0.001, tau =0.005, batch_size=100,
        input_dims=env.observation_space.shape, fc1_dims=400, fc2_dims=300,
        n_actions=env.action_space.shape[0], env=env)
    n_games = 100
    score_history = []
    agent.load_models()

    for epizode_number in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            observation = observation_
            env.render()

        score_history.append(score)
        avg_score = np.mean(score_history[-10:])
        print('epizod:', epizode_number, 'punkty:', score,
            'punkty uśrednione:', avg_score)
