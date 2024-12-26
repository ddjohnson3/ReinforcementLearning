import gymnasium as gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict


EPISODES =  30000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0

if __name__ == "__main__":
    env_name = "CliffWalking-v0"
    env = gym.envs.make(env_name)
    env.reset(seed=1)
    #starts with a pessimistic estimate of zero reward for each state.
    Q_table = defaultdict(default_Q_value)
    
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()[0]
        
        #epsilon greedy for action
        if random.random() < EPSILON:
            #random
            action = env.action_space.sample()
        else:
            #take best q value
            q_vals = {}
            for a in range(env.action_space.n):
                q_vals[a] = Q_table[(obs, a)]
            #max q val
            action = max(q_vals, key=q_vals.get)
        
        while (not done):
            #take action and get information about the next state
            next_state,reward,terminated,truncated,info = env.step(action)
            done = terminated or truncated

            #next_action COULD BE RANDOM FOR EXPLORATION OR COULD BE GREEEDY
            if random.random() < EPSILON:
                next_action = env.action_space.sample()
            else:
                
                next_q_vals = {}
                for a in range(env.action_space.n):
                    next_q_vals[a] = Q_table[(next_state, a)]
                next_action = max(next_q_vals, key=next_q_vals.get)
            
            #backpropagate one step and compute new q 
            new_q = Q_table[(obs,action)] + LEARNING_RATE * (reward + DISCOUNT_FACTOR * Q_table[(next_state, next_action)] - Q_table[(obs, action)])
            Q_table[(obs,action)] = new_q

            #update state and action
            obs = next_state
            action = next_action

            episode_reward += reward # update episode reward
        
        EPSILON = max(EPSILON * EPSILON_DECAY, 0.01)
        episode_reward_record.append(episode_reward)
            

        # record the reward for this episode
        episode_reward_record.append(episode_reward) 
     
        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    
    model_file = open(f'Q_TABLE_SARSA.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()
