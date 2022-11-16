# Name: Megan Le
# email: mle25@wisc.edu
# Class/Semester: CS 540 Spring 2021
# Instructor: Sharon Li
import gym
import random
import numpy as np
import time
from collections import deque
import pickle

from collections import defaultdict

EPISODES = 20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v0")
    env.seed(1)
    env.action_space.np_random.seed(1)

    Q_table = np.zeros((env.observation_space.n, env.action_space.n))
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0  # initialize the episode reward
        state = env.reset()  # reset the state before taking a step
        done = False  # initialize done

        while not done:
            action = None
            if np.random.uniform(0, 1) < EPSILON:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_table[state, :])

            new_state, reward, done, info = env.step(action)  # take a step in the state with the action

            # r+γmaxa′∈AQ(s′,a′)
            target = reward + DISCOUNT_FACTOR * np.max(Q_table[new_state, :])
            # Q(s,a)+α(target−Q(s,a))
            Q_table[state, action] = Q_table[state, action] + LEARNING_RATE * (target - Q_table[state, action])

            episode_reward += reward  # append the current reward to the total episode reward
            state = new_state  # set new state to be the current state

        episode_reward_record.append(episode_reward)  # append the episode's reward to the episode reward record
        EPSILON *= EPSILON_DECAY  # change epsilon according to epsilon decay

        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record)) / 100))
            print("EPSILON: " + str(EPSILON))

    ####DO NOT MODIFY######
    model_file = open('Q_TABLE.pkl', 'wb')
    pickle.dump([Q_table, EPSILON], model_file)
    #######################
