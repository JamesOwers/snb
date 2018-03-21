#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Code to reproduce Figure 4.2 - Jack's Car Rental problem solved by greedy
Policy iteration. 

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
import logging

MAX_NR_CARS = 20
ACTION_SPACE = range(-5, 6)
MAX_EPOCHS = 5

class CarRentalEnv():
    def __init__(self,
                 action_space=ACTION_SPACE,
                 max_nr_cars=MAX_NR_CARS,
                 rent_reward=10,
                 move_reward=-2,
                 initial_state=(10, 10),
                 env_request_lam=(3, 4),
                 env_return_lam=(3, 2),
                 verbose=0):
        logger = logging.getLogger("Jack's Car Rental Env")
        ch = logging.StreamHandler()
        logger.handlers = []
        logger.addHandler(ch)
        logger.setLevel(logging.WARNING)
        if verbose > 0:
            logger.setLevel(logging.INFO)
            if verbose > 1:
                logger.setLevel(logging.DEBUG)
        self.logger = logger
        self.action_space = action_space
        self.max_nr_cars = max_nr_cars
        self.rent_reward = rent_reward
        self.move_reward = move_reward
        self.state = np.array(initial_state)
        self.env_request_lam = env_request_lam
        self.env_return_lam = env_return_lam
        self.logger.info("Initialised Jack's Car Rental Env with params:\n{}".
                         format(vars(self)))
    
    def step(self, action):
        """
        1. Sample nr cars requested at each location
        2. Calculate reward for sucessful hire
        3. Sample nr cars returned
        4. Perform action
        """
        reward = 0
        assert action in self.action_space, \
            "Your requested action [{}] is not in the action space [{}]".\
                format(action, self.action_space)
        requests = np.array(
                [np.random.poisson(lam=lam) for lam in self.env_request_lam])
        new_state = np.maximum(self.state - requests, np.zeros_like(requests))
        rented = self.state - new_state
        self.logger.info("Requests: {}".format(requests))
        self.logger.info("Fulfilled rentals: {}".format(rented))
        self.logger.info("New state: {}".format(new_state))
        returns = [np.random.poisson(lam=lam) for lam in self.env_return_lam]
        new_state = np.minimum(new_state+returns, 
                               np.ones_like(returns)*self.max_nr_cars)
        self.logger.info("Returns: {}".format(returns))
        self.logger.info("New state: {}".format(new_state))
        movement = [-action, action]  # +ve action sends cars from 1 --> 2
        new_state = np.minimum(new_state+movement, 
                               np.ones_like(returns)*self.max_nr_cars)
        self.logger.info("Movement: {}".format(movement))
        self.logger.info("New state: {}".format(new_state))
        self.state = new_state
        reward = np.sum(rented)*self.rent_reward + abs(action)*self.move_reward
        # OpenAI gym terminology
        observation = self.state
        done = False
        info = None
        return observation, reward, done, info
    
    def render(self):
        return self.state
        
env = CarRentalEnv(verbose=2)
    
policy = np.zeros((MAX_NR_CARS, MAX_NR_CARS))
value = np.zeros((MAX_NR_CARS, MAX_NR_CARS))

#for ii in range(MAX_EPOCHS):
    