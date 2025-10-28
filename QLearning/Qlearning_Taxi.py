import numpy as np
import gymnasium as gym
import random

env = gym.make('Taxi-v3')                   # initialize environment
alpha = 0.9                                 # learning rate: how important of new information and old information
gamma = 0.99                                # discount factor: how important of the value in future/ long term
epsilon = 1                                 # the ratio between exploration and exploitation
epsilon_decay = 0.9995                      # the ratio for reducing epsilon
min_epsilon = 0.01
num_episodes = 10000
max_steps = 100                             # max step for each episode

# Initialize q table equal to zero
q_table = np.zeros((env.observation_space.n, env.action_space.n))

def take_action(state):
    if random.uniform(0,1) < epsilon:                      # if random value < epsion then exploration
        action = env.action_space.sample()                       # randomly choose action
    else:
        action = np.argmax(q_table[state,:])                    # choose the best action base on q value (exploitation)

    return action

for episode in range(num_episodes):
    state, _ = env.reset()                                       #randomly set the initial state

    for step in range(max_steps):
       # env.render()
        action = take_action(state)                             # choose action

        # use the action
        next_state, reward, done, truncated, info = env.step(action)

        max_q_future = np.max(q_table[next_state,:])                    # the optimal future value

        # update q table using Bellman equation
        q_table[state,action] = (1-alpha)*q_table[state,action] + alpha*(reward + gamma*max_q_future)

        state = next_state

        #update epsilon
        epsilon = max(epsilon_decay * epsilon, min_epsilon)

        if done or truncated:
            break
    #print("Trainning Episode {} with reward {}".format(episode, reward))


env.close()
# Deploy the model after trainning

env = gym.make('Taxi-v3', render_mode='human')

for episode in range(6):
    state, _ = env.reset()

    for step in range(max_steps):
        env.render()
        action = np.argmax(q_table[state,:])
        next_state, reward, done, truncated, info = env.step(action)

        state = next_state

        if done or truncated:
            env.render()
            print("Episode {} finished with reward = {} ".format(episode,reward))
            break



