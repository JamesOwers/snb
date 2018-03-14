#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Was intrigued if I could solve the system of linear equations directly
as is stated as the method for this gridworld example. States are numbered
as in np.arange(5**2).reshape(5, 5):
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24]])
"""

import numpy as np


A = np.zeros((25, 25))
b = np.zeros(25)

neighbour_dict = {}
for ii in range(25):
    neighbours = []
    file = (ii % 5)
    if file - 1 >= 0:
        neighbours += [ii-1]
    if file + 1 <= 4:
        neighbours += [ii+1]
    rank = ii // 5
    if rank - 1 >= 0:
        neighbours+= [ii-5]
    if rank + 1 <= 4:
        neighbours+= [ii+5]
    neighbour_dict[ii] = neighbours

# Corrections
neighbour_dict[1] = [21]
neighbour_dict[3] = [13]

# Centres
# v = 1/4{9/10*(v_north + v_east + v_south + v_west)}
# 40/9*v - [v_neighbours] = 0
coef = 40/9  # coefficient
val = 0  # value
for ii in [6, 7, 8, 11, 12, 13, 16, 17, 18]:
    A[ii, ii] = coef
    for nn in neighbour_dict[ii]:
        A[ii, nn] = -1
    b[ii] = val

# Corners
# v = 1/4{9/10*(v_1 + v_2 + 2*v) - 2}
# 22/9*v - v_1 - v_2 = -20/9
c_coef = 22/9  # corner coefficient
c_val = -20/9  # corner value
for ii in [0, 4, 20, 24]:
    A[ii, ii] = c_coef
    for nn in neighbour_dict[ii]:
        A[ii, nn] = -1
    b[ii] = c_val

# Edges
# v = 1/4{9/10*(v_1 + v_2 + v_3 + v) - 1}
e_coef = 31/9  # edge coefficient
e_val = -10/9  # edge value
for ii in [2, 9, 14, 19, 5, 10, 15, 21, 22, 23]:
    A[ii, ii] = e_coef
    for nn in neighbour_dict[ii]:
        A[ii, nn] = -1
    b[ii] = e_val

# Specials
A[1, 1] = 10/9
A[1, neighbour_dict[1]] = -1
b[1] = 100/9

A[3, 3] = 10/9
A[3, neighbour_dict[3]] = -1
b[3] = 50/9

sltn = np.linalg.solve(A,b).reshape(5, 5)
#print(sltn)
print(np.round(sltn, 1))