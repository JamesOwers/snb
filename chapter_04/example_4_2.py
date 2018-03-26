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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
from scipy.stats import poisson
from collections import defaultdict
from itertools import product

PROJECT_HOME = '/home/james/git/snb'
FIGURE_OUT = '{}/chapter_04'.format(PROJECT_HOME)
assert os.path.exists(PROJECT_HOME), 'Set PROJECT_HOME (=[{}]) in this file'.\
                                     format(PROJECT_HOME)
MAX_NR_CARS = 20
ACTION_SPACE = lambda: range(-5, 6)
STATE_SPACE = lambda: product(range(MAX_NR_CARS+1), range(MAX_NR_CARS+1))
MAX_EPOCHS = 10
RENT_REWARD = 10
MOVE_REWARD = -2
GAMMA = 0.9
CHECK_PROBS = True
REQUEST_LAM = (3, 4)
RETURN_LAM = (3, 2)

poisson_pmf_dict = dict()
poisson_cdf_dict = dict()

def make_env_probs():
    # TODO: Probably better to work in log space for small probabilities
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

    with open('{}/chapter_04/prob_{}.pkl'.format(PROJECT_HOME,
              MAX_NR_CARS), 'wb') as f:
        pickle.dump(prob, f)
    
    return prob


def evaluate_policy(policy, value, prob, discount, tol=1e-9):
    """
    Performs an iterative policy evaluation. Stops when the maximum value 
    difference for a given iteration is less than tol.
    
    Arguments
    ---------
    policy : numpy array returning action for a given state tuple
    value : numpy array returning the value of a state given a state tuple
    prob : pandas series object with index s', r, s, a returning p(s',r|s,a)
    discount : the discount factor to apply when getting the expected update
    """
    states = prob.index.levels[prob.index.names.index('s')]
    max_diff = np.inf
    while max_diff > tol:
        max_diff = 0
        init_value = value.copy()
        for s in states:
            a = policy[s]
            probs_gvn_sa = prob[:, :, s, a].reset_index()
            df = probs_gvn_sa
            idx = [ii for ii in zip(*df["s'"].values)]
            rtn = (df["p(s', r|s, a)"] * (df['r'] + 
                                          discount*value[idx[0], idx[1]]))
            value[s] = np.sum(rtn)
            diff = abs(init_value[s] - value[s])
            max_diff = max(diff, max_diff)
    return value


def improve_policy(policy, value, prob, discount):
    policy_stable = True
    states = prob.index.levels[prob.index.names.index('s')]
    old_policy = policy.copy()
    for s in states:
        old_action = old_policy[s]
        probs_gvn_s = prob[:, :, s, :].reset_index()
        df = probs_gvn_s
        action_value = dict()
        for a, df_a in df.groupby('a'):
            idx = [ii for ii in zip(*df_a["s'"].values)]
            rtn = (df_a["p(s', r|s, a)"] * (df_a['r'] + 
                                          discount*value[idx[0], idx[1]]))
            action_value[a] = rtn.sum()
        # N.B. idxmin returns the first occurance, the sort_index() ensures
        # that ties are broken by selecting the minimum action (i.e. most -ve)
        policy[s] = pd.Series(action_value).sort_index().idxmax()
        if policy_stable:
            policy_stable = policy[s] == old_action
    return policy, policy_stable


def plot_iteration(policy, value):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    sns.heatmap(value, annot=True, fmt='.0f', ax=ax[0]).invert_yaxis()
    ax[0].set_ylabel('Depot 1')
    ax[0].set_xlabel('Depot 2')
    ax[0].set_title('Value')
    action_space = list(ACTION_SPACE())
    sns.heatmap(policy, annot=True, fmt='.0f', ax=ax[1], 
                vmin=action_space[0], vmax=action_space[-1]).invert_yaxis()
    ax[1].set_title('Policy')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    

if __name__ == '__main__':
    print("Running Jack's cars experiment with max_cars = {}".format(
            MAX_NR_CARS))
    if not os.path.exists('{}/chapter_04/prob_{}.pkl'.format(PROJECT_HOME,
                          MAX_NR_CARS)):
        print('Making environment transition probabilities')
        prob = make_env_probs()
    else:
        print('Loading environment transition probabilities from file')
        prob = pickle.load(open('{}/chapter_04/prob_{}.pkl'.
                                format(PROJECT_HOME, MAX_NR_CARS), 'rb'))
        
    # Make Panda-y - very useful for iterating over states as index levels
    # are stored. See evaluate_policy and improve_policy for examples
    prob = pd.Series(prob)
    prob.name = "p(s', r|s, a)"
    prob.index.names = ["s'", 'r', 's', 'a']
    
    if CHECK_PROBS:
        print('Checking probabilities sum to 1')
#        prob = pickle.load(open('{}/chapter_04/prob_20.pkl'.format(PROJECT_HOME), 'rb'))
    #    prob.sort_index(level=["s", 'a', "s'", 'r'])
    #    prob.groupby(level=['s', 'a']).sum().plot('hist')
        grp_sum = prob.groupby(level=['s', 'a']).sum().sort_index(level=['s', 'a'])
        tol = 1e-12
        assert(grp_sum[grp_sum < 1.-tol].shape[0] == 0), "...They don't"
        print('Test passed')
    #    pd.DataFrame(prob).query('s == tuple((0, 4))')
    
    policy = np.zeros((MAX_NR_CARS+1, MAX_NR_CARS+1), dtype=int)
    value = np.zeros((MAX_NR_CARS+1, MAX_NR_CARS+1), dtype=float)
    ii = 0
    plot_iteration(policy, value)
    plt.suptitle('Iteration {}'.format(ii))
    plt.savefig('{}/{}__iteration_{}.pdf'.format(
                FIGURE_OUT, MAX_NR_CARS, ii), dpi=300)
    policy_stable = False
    while not policy_stable:
        ii += 1
        print('Iteration {}'.format(ii))
        print('Evaluating Policy')
        value = evaluate_policy(policy, value, prob, GAMMA)
        print('Improving Policy')
        policy, policy_stable = improve_policy(policy, value, prob, GAMMA)
        plot_iteration(policy, value)
        plt.suptitle('Iteration {}'.format(ii))
        plt.savefig('{}/{}__iteration_{}.pdf'.format(
                FIGURE_OUT, MAX_NR_CARS, ii), dpi=300)
    print('Policy stable')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    