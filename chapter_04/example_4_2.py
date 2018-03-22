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

The order of events is:
1. s = cars at both depots at close of business
2. a = number of cars to move (+ve moves from 1 --> 2)
3. s' = cars at both depots at close of buiness *tomorrow*
4. r = +10 for each car sucessfully rented -2 for each car moved
    * caveat: need to check nr cars available to rent, i.e. nr cars
    successfully rented = min(cars_avail, cars_requested)
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

MAX_NR_CARS = 20
ACTION_SPACE = range(-5, 6)
MAX_EPOCHS = 10
RENT_REWARD = 10
MOVE_REWARD = -2

policy = np.zeros((MAX_NR_CARS, MAX_NR_CARS))
value = np.zeros((MAX_NR_CARS, MAX_NR_CARS))

fig, ax = plt.subplots(1,1)
mu = 10
x = np.arange(poisson.ppf(0.01, mu),
              poisson.ppf(0.99, mu))
ax.plot(x, poisson.pmf(x, mu), 'bo', ms=8, label='poisson pmf')
ax.vlines(x, 0, poisson.pmf(x, mu), colors='b', lw=5, alpha=0.5)

# Evaluate

# Improve