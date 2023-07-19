#!/usr/bin/env python
# coding: utf-8

# In[22]:


# gym==0.17.2
# numpy==1.18.0

import gym
import numpy as np
from tqdm import trange

class QLearningAgent(object):
    def __init__(self):
        n_episodes = input("Insert number of episodes to run:")
        self.n_episodes = int(n_episodes)
        
        self.Q = None
        self.alpha = None
        self.gamma = None
        self.env = gym.make("Taxi-v3").env

    def solve(self):
        """Implement the agent"""
        
        #-- Initialize variables
        self.epsilon = 1
        self.gamma = 0.9
        self.alpha = 0.2
        
        #-- Initialize Q-table
        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        
        #-- Loop through episodes
        for episode in range(self.n_episodes):
            self.state = self.env.reset()
            
            while True:
                self.action = self.greedy_action(self.state)
                next_state,next_reward,is_done,_ = self.env.step(self.action)
                
                #-- Optimal action for next state that yields maximum expected reward
                next_action = np.argmax(self.Q[next_state, :])
                
                #-- Update Q-table
                self.update(next_state, next_reward, next_action)
                
                self.state = next_state
                
                if is_done:
                    break
        
        #-- Update epsilon-greedy to prevent being stuck in local optimum. Introduces randomness in action selection.
        self.update_epsilon()
        
    def greedy_action(self,state):
        """return agent action"""
        if np.random.random() < self.epsilon:
            return np.random.randint(6)
        else:
            return np.argmax(self.Q[state, :])
    
    def update_epsilon(self):
        """update epsilon-greedy"""
        if self.epsilon > 0.1:
            self.epsilon = self.epsilon * (1 - 0.1)
           
    def update(self,next_state, reward, next_action):
        """New Q-value[state,action] = Current Q-value[state,action] + Learning Rate * (Reward[state,action] + Discount Rate * Maximum Expected Value of Next (state,action) pair - Current Q-value[state,action])"""
        self.Q[self.state, self.action] = self.Q[self.state, self.action] + self.alpha * (reward + self.gamma * self.Q[next_state, next_action] - self.Q[self.state, self.action])

    def Q_table_val(self, state, action):
        """return the optimal value for State-Action pair in the Q Table"""
        return self.Q[state][action]
    
    def Q_table(self):
        """return the Q Table"""
        return self.Q

