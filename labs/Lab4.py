#!/usr/bin/ipython3

'''
This program explores the one-dimensional heat diffusion equation 
for Lab 3, modeling the ground temperature dynamics over time and 
seasonally for Kangerlussuaq, Greenland.

To reproduce the values and plots in my report, please simply run the script.
In the terminal, use the commands "ipython" and then "run Lab3.py". 
'''

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand

plt.style.use('seaborn-v0_8-colorblind')
plt.ion()
plt.close('all')

def forest_fire(isize=3, jsize=3, nstep=4, spread_chance=1.0):
    '''
    This function models a forest fire...

    Parameters
    ----------
    isize, jsize:   int, defaults = 3
                    Set the size of the forest in x and y direction, repesectively
    nstep:          int, default = 4
                    Set the number of steps to advance solution
    spread_chance:  float, default = 1.0
                    Threshold
    '''

    # Create a forest grid with each box representing a tree
    forest = np.zeros((nstep, isize, jsize)) + 2

    # Create the initial fire NEEDS TO BE UPDATED FOR LAB
    forest[0, isize//2, jsize//2] = 3

    # Iterate through entire forest, identify fires, and spread fire as needed
    for k in range(nstep-1):
        for i in range(isize):
            for j in range(jsize):

                # Check if a spot is on fire
                if forest[k, i, j] == 3: 

                    # Spread fire in each direction, based on the random chance and only spread to forest

                    if (spread_chance > rand()) & (forest[k, i-1, j] == 2) & (i>0):
                        forest[k+1, i-1, j] = 3 # spread North

                    if (spread_chance > rand()) & (forest[k, i-1, j] == 2) & (i<0):
                        forest[k+1, i+1, j] = 3 # spread South

                    if (spread_chance > rand()) & (forest[k, i-1, j] == 2) & (j>0):
                        forest[k+1, i, j-1] = 3 # spread West
                    
                    if (spread_chance > rand()rand_num) & (forest[k, i-1, j] == 2) & (j<0):
                        forest[k+1, i, j+1] = 3 # spread East

    return forest


forest_state = forest_fire()
print(forest_state)
# grid = np.meshgrid(3, 3)
# plt.contourf(grid, np.squeeze(forest_state, axis=0))
                

'''
Does not capture real physics of what happens in a forest fire

Average may be okay sometimes, but other factors have much larger effect
'''



