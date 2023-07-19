#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# numpy==1.18.0
# gym==0.17.2


import gym
import numpy as np
from math import sqrt
from gym.envs import toy_text


class FrozenLakeAgent(object):
    def __init__(self):
        pass

    
    def amap_to_gym(self, amap):
        """Maps the `amap` string to a gym env"""
        amap = np.asarray(amap, dtype='c')
        side = int(sqrt(amap.shape[0]))
        amap = amap.reshape((side, side))
        return gym.make('FrozenLake-v0', desc=amap).unwrapped
 

    def solve(self, amap, gamma, alpha, epsilon, n_episodes, seed):
        """Implement the agent"""
        
        #-- Initialize variables
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.n_episodes = n_episodes
        
        self.env = self.amap_to_gym(amap)
        np.random.seed(seed)
        self.env.seed(seed)
        
        #-- Initialize Q-table
        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        
        #-- Loop through episodes
        for episode in range(self.n_episodes):
            self.state = self.env.reset()
            self.action = self.greedy_action(self.state)

            while True:
                next_state,next_reward,is_done,_ = self.env.step(self.action)
                next_action = self.greedy_action(next_state)
                
                self.update(next_state, next_reward, next_action)
                
                self.state = next_state
                self.action = next_action
                
                if is_done:
                    break
                    
        self.Q = np.argmax(self.Q, axis=1)
        policy = self.return_directions()
        return policy

   
    def return_directions(self):
        """return agent Frozen Lake directions"""
        policy_movement = []
        for move in self.Q:
            if move == 0:
                policy_movement.append('<')
            elif move == 1:
                policy_movement.append('v')
            elif move == 2:
                policy_movement.append('>')
            else:
                policy_movement.append('^')
        return ''.join(map(str, policy_movement))
    
        
    def greedy_action(self,state):
        """return agent action"""
        if np.random.random() < self.epsilon:
            return np.random.randint(4)
        else:
            return np.argmax(self.Q[state, :])
        
        
    def update(self,next_state, reward, next_action):
        """New Q-value[state,action] = Current Q-value[state,action] + Learning Rate * (Reward[state,action] + Discount Rate * Future Reward - Current Q-value[state,action])"""
        self.Q[self.state, self.action] = self.Q[self.state, self.action] + self.alpha * (reward + self.gamma * self.Q[next_state, next_action] - self.Q[self.state, self.action])

