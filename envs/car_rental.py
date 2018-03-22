#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Jack's Car Rental environment

* States - (#cars, #cars) tuples representing nr cars in loc 1 and 2  
    * There are a max of 20 cars in each location
* Actions - the net number of cars moved $\in$ {-5, ..., 5} (+ve from 1 to 2)
* Environment - For locations 1 and 2 respectively:
    * requests are sampled from a poisson with means 3 and 4
    * returns are sampled from a poission with means 3 and 2
* Rewards - +10 for every car sucessfully loaned, -2 for every car moved
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MAX_NR_CARS = 20
ACTION_SPACE = range(-5, 6)
MAX_EPOCHS = 10

class CarRentalEnv():
    def __init__(self,
                 action_space=ACTION_SPACE,
                 max_nr_cars=MAX_NR_CARS,
                 rent_reward=10,
                 move_reward=-2,
                 initial_state=(10, 10),
                 env_request_lam=(3, 4),
                 env_return_lam=(3, 2)):
        self.action_space = action_space
        self.max_nr_cars = max_nr_cars
        self.rent_reward = rent_reward
        self.move_reward = move_reward
        self.state = np.array(initial_state)
        self.env_request_lam = env_request_lam
        self.env_return_lam = env_return_lam
        self.step_nr = 0
        columns = ['Open State',
                   'Requests',
                   'Rented', 
                   'Returns',
                   'Close State',
                   'Action',
                   'Movement',
                   'Reward']
        self.history = pd.DataFrame(columns=columns)
        self.history.loc[self.step_nr, 'Open State'] = self.state
        
    
    def step(self, action):
        """
        1. Sample nr cars requested at each location
        2. Calculate reward for sucessful hire
        3. Sample nr cars returned
        4. Perform action
            * If a movement is requested that is larger than the number of cars
            available, only the number available are moved. Otherwise, moving
            a large number of cars from one depo to another would be a way
            of creating 5 new cars from nowhere each round. The agent is still
            penalised for what it tried to move.
        """
        assert action in self.action_space, \
            "Your requested action [{}] is not in the action space [{}]".\
                format(action, self.action_space)
        self.step_nr += 1
        self.history.loc[self.step_nr, 'Open State'] = self.state
        requests = np.array(
                [np.random.poisson(lam=lam) for lam in self.env_request_lam])
        new_state = np.maximum(self.state - requests, np.zeros_like(requests))
        rented = self.state - new_state
        returns = [np.random.poisson(lam=lam) for lam in self.env_return_lam]
        new_state = np.minimum(new_state+returns, 
                               np.ones_like(returns)*self.max_nr_cars)
        self.history.loc[self.step_nr, 'Close State'] = new_state
        if action >= 0:  # +ve action sends cars from 1 --> 2
            action_adj = min(action, new_state[0])
        else: # -ve action sends cars from 2 --> 1
            action_adj = max(action, -new_state[1])
        movement = [-action_adj, action_adj]
        new_state = np.minimum(new_state+movement, 
                               np.ones_like(returns)*self.max_nr_cars)
        self.state = new_state
        reward = np.sum(rented)*self.rent_reward + abs(action)*self.move_reward
        self.history.loc[self.step_nr, 'Requests'] = requests
        self.history.loc[self.step_nr, 'Rented'] = rented
        self.history.loc[self.step_nr, 'Returns'] = returns
        self.history.loc[self.step_nr, 'Action'] = action
        self.history.loc[self.step_nr, 'Movement'] = movement
        self.history.loc[self.step_nr, 'Reward'] = reward
        # OpenAI gym terminology
        observation = self.state
        done = False
        info = None
        return observation, reward, done, info
    
    def render(self, full_history=True):
        if full_history:
            print(self.history)
        else:
            print(self.history.loc[self.step_nr])

    
if __name__ == '__main__':
    policy = np.zeros((MAX_NR_CARS, MAX_NR_CARS))
    value = np.zeros((MAX_NR_CARS, MAX_NR_CARS))
    env = CarRentalEnv()
    env.render()
    for ii in range(MAX_EPOCHS):
        action = np.random.choice(env.action_space)
        env.step(action)
        env.render()
        