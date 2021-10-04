import pybullet_envs
import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    #env_name = 'InvertedPendulumBulletEnv-v0'
    #env_name = 'BipedalWalkerHardcore-v3'
    #env_name = 'MinitaurBulletEnv-v0'
    env_name = 'AntBulletEnv-v0'
    env = gym.make(env_name)
    agent = Agent(alfa=0.0003, beta=0.0003, tau =0.005, batch_size=256,
        input_dims=env.observation_space.shape, fc1_dims=256, fc2_dims=256,
        n_actions=env.action_space.shape[0], env=env, reward_scale=2,
        env_name=env_name)
    n_games = 251
    fname = 'SAC' + env_name
    plik_wyjsciowy = 'plots/' + fname + '.png'

    best_score = env.reward_range[0]
    score_history = []
    test_mode = False
    if test_mode:
        agent.load_models()
        env.render(mode='human')

    steps = 0
    agent.load_models()
    for epizode_number in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            if not test_mode:
                agent.learn()
            score += reward
            observation = observation_
            #env.render()

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not test_mode:
                agent.save_models()


        print('epizod:', epizode_number, 'punkty:', score,
            'punkty uśrednione:', avg_score)
    if not test_mode:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(score_history, x, plik_wyjsciowy)
