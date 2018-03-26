#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Code to reproduce Figure 4.2 - Jack's Car Rental problem solved by greedy
Policy iteration. 

* States - (#cars, #cars) tuples representing nr cars in loc 1 and 2  
    * There are a max of 20 cars in each location
* Actions - the net number of cars moved in {-5, ..., 5} (+ve from 1 to 2)
* Environment - For locations 1 and 2 respectively:
    * requests are sampled from a poisson with means 3 and 4
    * returns are sampled from a poission with means 3 and 2
        * N.B. THEY ARE NOT ABLE TO BE HIRED TILL THE NEXT DAY
* Rewards - +10 for every car sucessfully loaned, -2 for every car moved

The order of events is:
1. s = cars at both depots at close of business
2. a = number of cars to move (+ve moves from 1 --> 2)
3. s' = cars at both depots at close of buiness *tomorrow*
4. r = +10 for each car sucessfully rented -2 for each car moved
    * caveat: need to check nr cars available to rent, i.e. nr cars
    successfully rented = min(cars_avail, cars_requested)
    
Design decisions:
* For construction of p, chose to assign zero probability of sending cars from
  a to b if it would take b over the 20 car limit (could have chosen to allow
  this, deduct the cost, but keep the car limit i.e. presume 'wasted' cars sent
  to the regional depot)
"""

import os
#import matplotlib.pyplot as plt
import numpy as np
import pickle
#import pandas as pd
from scipy.stats import poisson
from collections import defaultdict
from itertools import product

PROJECT_HOME = '/home/james/git/snb'
assert os.path.exists(PROJECT_HOME), 'Set PROJECT_HOME (=[{}]) in this file'.\
                                     format(PROJECT_HOME)
MAX_NR_CARS = 5
ACTION_SPACE = lambda: range(-5, 6)
STATE_SPACE = lambda: product(range(MAX_NR_CARS+1), range(MAX_NR_CARS+1))
MAX_EPOCHS = 10
RENT_REWARD = 10
MOVE_REWARD = -2

policy = np.zeros((MAX_NR_CARS, MAX_NR_CARS))
value = np.zeros((MAX_NR_CARS, MAX_NR_CARS))

poisson_pmf_dict = dict()
poisson_cdf_dict = dict()

def make_env_probs():
    # Getting poisson probabilities is a big overhead - create a lookup table
    
    def quick_poisson_pmf(n, lam):
            global poisson_pmf_dict
            key = (n, lam)
            if key not in poisson_pmf_dict:
                poisson_pmf_dict[key] = poisson.pmf(n, lam)
            return poisson_pmf_dict[key]    
    
    def quick_poisson_cdf(n, lam):
            global poisson_cdf_dict
            key = (n, lam)
            if key not in poisson_cdf_dict:
                poisson_cdf_dict[key] = poisson.cdf(n, lam)
            return poisson_cdf_dict[key] 
        
    REQUEST_LAM = (3, 4)
    RETURN_LAM = (3, 2)

    # construct the transition probabilities of the environment
    prob = defaultdict(int)
    # ss = state_before = (nr_cars depot 1, nr_cars depot 2)
    for ss_0 in STATE_SPACE():
        print("p(s', r|s={}, a=...)".format(ss_0))
        for aa in ACTION_SPACE():
            if aa > 0:  # aa +ve means 0 --> 1
                if any([# sending more cars than avail in 1
                        ss_0[0]-aa < 0,
                        # receiving more cars than space in 2
                        ss_0[1]+aa > MAX_NR_CARS]):
                    continue
            else:  # aa -ve means 1 --> 0
                if any([# receiving more cars than space in 1
                        ss_0[0]-aa > MAX_NR_CARS,
                        # sending more cars than avail in 2
                        ss_0[1]+aa < 0]):
                    continue
            ss_action = (ss_0[0]-aa, ss_0[1]+aa)
            for rented0, rented1 in product(range(ss_action[0]+1), 
                            range(ss_action[1]+1)):
                ss_rent = ss_action
                lam0, lam1 = REQUEST_LAM
                if rented0 == ss_rent[0]:  # i.e. requests for all the cars (or more)
                    rent0_prob = 1 - quick_poisson_cdf(rented0 - 1, lam0)  # remaining mass
                else:
                    rent0_prob = quick_poisson_pmf(rented0, lam0)
                if rented1 == ss_rent[1]:  # i.e. requests for all the cars (or more)
                    rent1_prob = 1 - quick_poisson_cdf(rented1 - 1, lam1)  # remaining mass
                else:
                    rent1_prob = quick_poisson_pmf(rented1, lam1)
                rent_prob = rent0_prob * rent1_prob
                ss_rent = (ss_rent[0]-rented0, ss_rent[1]-rented1)
                reward = (rented0+rented1)*RENT_REWARD + abs(aa)*MOVE_REWARD
                for returned0, returned1 in product(range(MAX_NR_CARS-ss_rent[0]+1), 
                                    range(MAX_NR_CARS-ss_rent[1]+1)):
                    ss_retn = ss_rent
                    lam0, lam1 = RETURN_LAM
                    if returned0 == MAX_NR_CARS-ss_retn[0]:
                        retn0_prob = 1 - quick_poisson_cdf(returned0 - 1, lam0)
                    else:
                        retn0_prob = quick_poisson_pmf(returned0, lam0)
                    if returned1 == MAX_NR_CARS-ss_retn[1]:
                        retn1_prob = 1 - quick_poisson_cdf(returned1 - 1, lam1)
                    else:
                        retn1_prob = quick_poisson_pmf(returned1, lam1)
                    retn_prob = retn0_prob * retn1_prob
                    ss_1 = (ss_retn[0]+returned0, ss_retn[1]+returned1)
                    prob[(ss_1, reward, ss_0, aa)] += rent_prob * retn_prob

    with open('{}/chapter_04/prob.pkl'.format(PROJECT_HOME), 'wb') as f:
        pickle.dump(prob, f)

if not os.path.exists('{}/chapter_04/prob.pkl'.format(PROJECT_HOME)):
    make_env_probs()

CHECK_PROBS = True
if CHECK_PROBS:
    import pandas as pd
    prob = pickle.load(open('{}/chapter_04/prob.pkl'.format(PROJECT_HOME), 'rb'))
    prob = pd.Series(prob)
    prob.name = "p(s', r|s, a)"
    prob.index.names = ["s'", 'r', 's', 'a']
#    prob.sort_index(level=["s", 'a', "s'", 'r'])
#    prob.groupby(level=['s', 'a']).sum().plot('hist')
    grp_sum = prob.groupby(level=['s', 'a']).sum().sort_index(level=['s', 'a'])
    tol = 1e-9
    assert(grp_sum[grp_sum < 1.-tol].shape[0] == 0)
#prob_df = pd.DataFrame(prob)


#fig, ax = plt.subplots(1,1)
#mu = 10
#x = np.arange(poisson.ppf(0.01, mu),
#              poisson.ppf(0.99, mu))
#ax.plot(x, poisson.pmf(x, mu), 'bo', ms=8, label='poisson pmf')
#ax.vlines(x, 0, poisson.pmf(x, mu), colors='b', lw=5, alpha=0.5)

# Evaluate

# Improve
