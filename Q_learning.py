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

    #default dict automatically creates a key if doesn't exsist
    Q_table = defaultdict(default_Q_value) # starts with estimate zero reward for each state
    
    #double ended Queue
    episode_reward_record = deque(maxlen=100)

    #each episode is a full interation to the goal or end state
    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()[0]

        
        while (not done):
            rand = random.random()
            if rand < EPSILON:
                action = env.action_space.sample()
            else:
                q_values = {}
                for action in range(env.action_space.n):
                    q_values[action] = Q_table[(obs,action)]
                best_action = max(q_values, key=q_values.get)
                action = best_action
    
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_state_q_values = []
            for next_action in range(env.action_space.n):
                next_state_q_values.append(Q_table[(next_state,next_action)])
            best_next_q = max(next_state_q_values)
            new_q = Q_table[(obs,action)] + LEARNING_RATE * (reward + DISCOUNT_FACTOR * best_next_q - Q_table[(obs,action)])
            Q_table[(obs,action)] = new_q
            obs = next_state
            episode_reward += reward
        
        EPSILON = max(EPSILON * EPSILON_DECAY, 0.01)            
            

        episode_reward_record.append(episode_reward) 
     
        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    
    model_file = open(f'Q_TABLE_QLearning.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()
