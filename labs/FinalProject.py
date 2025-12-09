#!/usr/bin/ipython3

'''
To reproduce similar values and plots to the ones in my report, please simply run
the script. In the terminal, use the commands "ipython" and then "run Lab4.py". 
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numpy.random import rand

plt.style.use('seaborn-v0_8-colorblind')
plt.ion()
plt.close('all')

def calc_pwind(w, dhat):

    # Convert w and dhat vectors into arrays
    w = np.asarray(w, dtype=float)
    dhat = np.asarray(dhat, dtype=float)

    dot = np.dot(w, dhat)

    return max( min(dot/10, 1), 0 )

# Define the function to model a forest fire
def forest_fire(isize=3, jsize=3, nstep=4, p_base=0.1, p_bare=0.0, p_ignite=0.0, wind=[10, 10]):
    '''
    This function models a forest fire on a 2D grid. Each ell can be bare/burnt (state 1),
    unburned forest (state 2), or actively burning (state 3).

    Parameters
    ----------
    isize, jsize:   int, defaults = 3
                    Set the size of the forest in x and y direction, respectively
    nstep:          int, default = 4
                    Set the number of steps to advance solution
    spread_chance:  float, default = 1.0
                    Set the probability that fire can spread in any direction, from 0 to 1 (or 0 to 100 %)
    bare_chance:    float, default = 0.0
                    Set the probability that a cell is naturally bare to begin with
    ignite_chance:  float, default = 0.0
                    Set the probability that a cell will catch fire at the start of the simulation.
                    If 0, the center cell is set on fire instead.

    Returns
    -------
    forest: array
            Forest state at each time step.
    '''

    # Create a forest grid with each box representing a tree
    forest = np.zeros((nstep, isize, jsize)) + 2

    # Create the initial fire 
    if p_ignite > 0:
        ignited = (np.random.rand(isize, jsize) < p_ignite) & (forest[0] == 2)
        forest[0, ignited] = 3
    else:
        # Just the center of forest catches fire
        forest[0, isize//2, jsize//2] = 3

    # Create initial bare forest cells
    isbare = np.random.rand(isize, jsize) # Create an array of randomly generated numbers to represent which cells begin bare
    isbare = isbare < p_bare # Turn it into an array of True/False values
    forest[0, isbare] = 1 # Change forest cells to bare if the corresponding isbare boolean is true

    # Create direction arrays
    wind = np.array(wind, dtype=float)
    north = np.array([0,1])
    south = np.array([0,-1])
    east = np.array([1,0])
    west = np.array([-1,0])

    # Iterate through entire forest, identify fires, and spread fire as needed
    for k in range(nstep-1):

        # Assume the next time step is the same as the current
        forest[k+1, :, :] = forest[k, :, :]

        for i in range(isize):
            for j in range(jsize):

                # Check if a spot is on fire
                if forest[k, i, j] == 3: 

                    # Spread fire in each direction, based on the random chance and only spread to forest
                    
                    p_wind = calc_pwind(wind, north)
                    p_spread = 1 - (1-p_base) * (1-p_wind)
                    if (i>0) and (forest[k, i-1, j] == 2) and (p_spread > rand()):
                        forest[k+1, i-1, j] = 3 # spread North to South

                    p_wind = calc_pwind(wind, south)
                    p_spread = 1 - (1-p_base) * (1-p_wind)
                    if (i<isize-1) and (forest[k, i+1, j] == 2) and (p_spread > rand()):
                        forest[k+1, i+1, j] = 3 # spread South to North

                    p_wind = calc_pwind(wind, west)
                    p_spread = 1 - (1-p_base) * (1-p_wind)
                    if (j>0) and (forest[k, i, j-1] == 2) and (p_spread > rand()):
                        forest[k+1, i, j-1] = 3 # spread East to West
                    
                    p_wind = calc_pwind(wind, east)
                    p_spread = 1 - (1-p_base) * (1-p_wind)
                    if (j<jsize-1) and (forest[k, i, j+1] == 2) and (p_spread > rand()):
                        forest[k+1, i, j+1] = 3 # spread West to East

                    # Change current burning trees to burnt in the next time step
                    forest[k+1, i, j] = 1

    return forest

def plot_fire(forest, nsteps=4, width=12, height=4):
    '''
    Given a forest fire of size (ntime, nx, ny), plot the evolution of 
    the fire over time. Creates a row of subplots showing the forest state
    at each time step.
    
    Parameters
    ----------
    forest: array
            3D array of forest states, returned by forest_fire function
    nsteps: int, default = 4
            Number of time steps to plot.
    width, height: float, defaults 12, 4
                Figure size.

    Returns
    -------
    fig, ax: Figure and Axes
             Matplotlib figure and array of axes with the plots.
    '''

    # Generate our custom segmented color map for this project.
    # We can specify colors by names and then create a colormap that only uses
    # those names. We have 3 fundamental states, so we want only 3 colors.
    forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])

    # Create the figure and plot the forest state for each time step
    fig, ax = plt.subplots(1, nsteps, figsize=(width, height), sharey=True)
    for k in range(nsteps):
        ax[k].pcolor(forest[k], cmap=forest_cmap, vmin=1, vmax=3)
        #ax[k].invert_yaxis() # Fixes y-axis so North is up and South is down
        ax[k].set_title(f'Forest State (Step {k})')
        ax[k].set_xlabel('X (km)')
    
    # Other formatting
    ax[0].set_ylabel('Y (km)') # Set shared y-axis label
    fig.tight_layout(rect=[0, 0, 1, 0.90]) # tight layout with room for title
    fig.suptitle('Forest Fire Progression over Time', weight='bold', y=0.97) # Set heading title

    # Create a legend
    ax[0].plot([], [], lw=3, color='tan', label='Bare/Burnt')
    ax[0].plot([], [], lw=3, color='darkgreen', label='Forest')
    ax[0].plot([], [], lw=3, color='crimson', label='Burning')
    fig.legend(ncol=3)

    return fig, ax

# Test calc_pwind
print(calc_pwind([15, 5], [1, 0]))  # wind mostly east, direction east; dot = 15; p_wind=1
print(calc_pwind([15, 5], [0, 1]))  # wind mostly east, direction north; dot = 5; p_wind=0.5
print(calc_pwind([15, 5], [-1, 0])) # wind east, direction west; dot = -15; p_wind=0

# Test forest fire
forest_state = forest_fire(isize=3, jsize=3, nstep=4, p_base=0.1, p_bare=0.0, p_ignite=0.0, wind=[10, 5])
plot_fire(forest=forest_state, nsteps=4, width=12.5, height=4)