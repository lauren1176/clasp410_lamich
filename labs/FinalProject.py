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

# Define the function to calculate the fire spread probability based on wind 
def calc_pwind(w, dhat, linear=True):

    # Convert w and dhat vectors into arrays
    w = np.asarray(w, dtype=float)
    dhat = np.asarray(dhat, dtype=float)

    # Calculate dot product
    s = np.dot(w, dhat)

    # Initialize result variable
    result = 0

    # Get p_wind value for linear case
    if linear:
        result = max( min(s/10.0, 1.0), 0.0 )

    # Get p_wind value for nonlinear case
    else:
        # Wind speed boundaries
        s1 = 0.447
        s2 = 5.365
        s3 = 13.859
        s4 = 20.565

        # Piecewise function using if statements
        if s <= 0.0:
            result = 0.0

        elif s <= s1:
            result = 0.1 * s / s1

        elif s <= s2:
            result = 0.1 + (0.3 - 0.1) * (s - s1) / (s2 - s1)

        elif s <= s3:
            result = 0.3 + (0.6 - 0.3) * (s - s2) / (s3 - s2)

        elif s <= s4:
            result = 0.6 + (0.9 - 0.6) * (s - s3) / (s4 - s3)

        else:
            result = 1.0

    return result

# Define the function to model a forest fire
def forest_fire(isize=3, jsize=3, nstep=4, p_base=1, p_bare=0.0, p_ignite=0.0, wind=[0, 0]):
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

                    # Spread fire in each direction, based on random chance and only spread to forest
                    
                    p_wind = calc_pwind(wind, north) # Calculate the wind spread probability
                    p_spread = 1 - (1-p_base) * (1-p_wind)
                    if (i<isize-1) and (forest[k, i+1, j] == 2) and (p_spread > rand()):
                        forest[k+1, i+1, j] = 3 # spread North (visually up)

                    p_wind = calc_pwind(wind, south)
                    p_spread = 1 - (1-p_base) * (1-p_wind)
                    if (i>0) and (forest[k, i-1, j] == 2) and (p_spread > rand()):
                        forest[k+1, i-1, j] = 3 # spread South (visually down)

                    p_wind = calc_pwind(wind, east)
                    p_spread = 1 - (1-p_base) * (1-p_wind)
                    if (j<jsize-1) and (forest[k, i, j+1] == 2) and (p_spread > rand()):
                        forest[k+1, i, j+1] = 3 # spread East

                    p_wind = calc_pwind(wind, west)
                    p_spread = 1 - (1-p_base) * (1-p_wind)
                    if (j>0) and (forest[k, i, j-1] == 2) and (p_spread > rand()):
                        forest[k+1, i, j-1] = 3 # spread West
                    
                    # Change current burning trees to burnt in the next time step
                    forest[k+1, i, j] = 1

    return forest

# Define a plotting function for a basic forest fire progression
def plot_fire(forest, nsteps=4, width=12, height=4):
    '''
    Given a forest fire of size (ntime, nx, ny), plot the evolution of 
    the fire over time. Creates a row of subplots showing the forest state
    at each time step.
    
    Parameters
    ----------
    forest: array
            3D array of forest states, returned by forest_fire function
    nsteps: int; default = 4
            Number of time steps to plot
    width, height:  float; defaults = 12, 4
                    Figure size

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

def run_fire_sims(nruns=10, **fire_kwargs):
    '''
    Run forest fire simulations and average the final burn state.

    Parameters
    ----------
    nruns:  int
            Number of simulations to run
    kwargs: dict
            Keyword arguments to pass to the forest_fire() function

    Returns
    -------
    burn_prob:  2D array
                Probability that each cell is burned at the final step
    '''

    # Run one simulation to get grid size
    test_run = forest_fire(**fire_kwargs)
    final_shape = test_run[-1].shape

    # Initialize variable to count number of burned cells
    burn_count = np.zeros(final_shape)

    for i in range(nruns):
        # Call the forest_fire function for 
        forest = forest_fire(**fire_kwargs)
        final_state = forest[-1]

        # Count burned cells in final state
        burn_count += (final_state == 1)

    # Convert counts to probabilities
    burn_prob = burn_count / nruns

    return burn_prob

### Question 1 ###

# Plot 3 x 3 grid 
forest_state = forest_fire(p_base=1)
plot_fire(forest=forest_state, nsteps=4, width=12.5, height=4)

# Plot 5 x 3 grid 
forest_state2 = forest_fire(isize=3, jsize=5, p_base=1)
plot_fire(forest=forest_state2, nsteps=4, width=16.5, height=3.5)


### Question 2 ###

# calc_pwind test cases
print('Test Cases for p_wind')
print(f'[10, 0] in north direction: {calc_pwind([10, 0], [0, 1])}') # wind strongly east, direction north; dot = 0; p_wind=0
print(f'[5, 0] in east direction: {calc_pwind([5, 0], [1, 0])}')  # wind weakly east, direction east; dot = 5; p_wind = 0.5
print(f'[5, 0] in west direction: {calc_pwind([5, 0], [-1, 0])}') # wind weakly east, direction west; dot = -5; p_wind = 0
print(f'[10, -10] in south direction: {calc_pwind([10, -10], [0, -1])}') # wind strongly east & south, direction south; dot = 10; p_wind = 1


### Question 3 ###

# Test forest fire case 1 #

# Define the parameters that will be passed into forest_fire for this test case
fire_params = dict(
    isize=11,
    jsize=11,
    nstep=10,
    p_base=0.2,
    p_bare=0.0,
    p_ignite=0.0,
    wind=[10, 0]
)
# Run the fire simulations and get the probability that a cell is burned
nruns = 10
burn_prob = run_fire_sims(nruns, **fire_params)

# Create the figure
fig, ax = plt.subplots(1, 1, figsize=(7, 7))

# Plot the final forest state's probability that a cell is burned
plot = ax.pcolor(burn_prob, cmap='afmhot', vmin=0, vmax=1)

# Plot format
ax.set_title(f'Linear Model: Burn Probability over {nruns} Runs for Eastward Wind [10, 0]')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
cbar = plt.colorbar(plot, orientation='horizontal')
cbar.set_label('Burn Probability')

# Test forest fire case 2 #

# Define the parameters that will be passed into forest_fire for this test case
fire_params_2 = dict(
    isize=11,
    jsize=11,
    nstep=10,
    p_base=0.2,
    p_bare=0.0,
    p_ignite=0.0,
    wind=[0, 10]
)
# Run the fire simulations and get the probability that a cell is burned
burn_prob_2 = run_fire_sims(nruns, **fire_params_2)

# Create the figure
fig, ax = plt.subplots(1, 1, figsize=(7, 7))

# Plot the final forest state's probability that a cell is burned
plot = ax.pcolor(burn_prob_2, cmap='afmhot', vmin=0, vmax=1)

# Plot format
ax.set_title(f'Linear Model: Burn Probability over {nruns} Runs for Northward Wind [0, 10]')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
cbar = plt.colorbar(plot, orientation='horizontal')
cbar.set_label('Burn Probability')

# Test forest fire case 3 #

# Define the parameters that will be passed into forest_fire for this test case
fire_params_3 = dict(
    isize=11,
    jsize=11,
    nstep=15,
    p_base=0.2,
    p_bare=0.0,
    p_ignite=0.0,
    wind=[10, 10]
)
# Run the fire simulations and get the probability that a cell is burned
burn_prob_3 = run_fire_sims(nruns, **fire_params_3)

# Create the figure
fig, ax = plt.subplots(1, 1, figsize=(7, 7))

# Plot the final forest state's probability that a cell is burned
plot = ax.pcolor(burn_prob_3, cmap='afmhot', vmin=0, vmax=1)

# Plot format
ax.set_title(f'Linear Model: Burn Probability over {nruns} Runs for Northeast Wind [10, 10]')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
cbar = plt.colorbar(plot, orientation='horizontal')
cbar.set_label('Burn Probability')

# Test forest fire case 4 #

# Define the parameters that will be passed into forest_fire for this test case
fire_params_4 = dict(
    isize=11,
    jsize=11,
    nstep=15,
    p_base=0.2,
    p_bare=0.0,
    p_ignite=0.0,
    wind=[-10, -10]
)
# Run the fire simulations and get the probability that a cell is burned
burn_prob_4 = run_fire_sims(nruns, **fire_params_4)

# Create the figure
fig, ax = plt.subplots(1, 1, figsize=(7, 7))

# Plot the final forest state's probability that a cell is burned
plot = ax.pcolor(burn_prob_4, cmap='afmhot', vmin=0, vmax=1)

# Plot format
ax.set_title(f'Linear Model: Burn Probability over {nruns} Runs for Southwest Wind [-10, -10]')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
cbar = plt.colorbar(plot, orientation='horizontal')
cbar.set_label('Burn Probability')


### Question 4 ###

# Test wind speed case 1 #

# Define the parameters that will be passed into forest_fire for this test case
fire_params = dict(
    isize=11,
    jsize=11,
    nstep=10,
    p_base=0.2,
    p_bare=0.0,
    p_ignite=0.0,
    wind=[0.2, 0]
)
# Run the fire simulations and get the probability that a cell is burned
nruns = 10
burn_prob = run_fire_sims(nruns, **fire_params)

# Create the figure
fig, ax = plt.subplots(1, 1, figsize=(7, 7))

# Plot the final forest state's probability that a cell is burned
plot = ax.pcolor(burn_prob, cmap='afmhot', vmin=0, vmax=1)

# Plot format
ax.set_title(f'Nonlinear Model: Burn Probability over {nruns} Runs for Very Light Wind [0.2, 0]')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
cbar = plt.colorbar(plot, orientation='horizontal')
cbar.set_label('Burn Probability')

# Test wind speed case 2 #

# Define the parameters that will be passed into forest_fire for this test case
fire_params = dict(
    isize=11,
    jsize=11,
    nstep=10,
    p_base=0.2,
    p_bare=0.0,
    p_ignite=0.0,
    wind=[3, 0]
)
# Run the fire simulations and get the probability that a cell is burned
nruns = 10
burn_prob = run_fire_sims(nruns, **fire_params)

# Create the figure
fig, ax = plt.subplots(1, 1, figsize=(7, 7))

# Plot the final forest state's probability that a cell is burned
plot = ax.pcolor(burn_prob, cmap='afmhot', vmin=0, vmax=1)

# Plot format
ax.set_title(f'Nonlinear Model: Burn Probability over {nruns} Runs for Light Breeze [3, 0]')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
cbar = plt.colorbar(plot, orientation='horizontal')
cbar.set_label('Burn Probability')

# Test wind speed case 3 #

# Define the parameters that will be passed into forest_fire for this test case
fire_params = dict(
    isize=11,
    jsize=11,
    nstep=10,
    p_base=0.2,
    p_bare=0.0,
    p_ignite=0.0,
    wind=[8, 0]
)
# Run the fire simulations and get the probability that a cell is burned
nruns = 10
burn_prob = run_fire_sims(nruns, **fire_params)

# Create the figure
fig, ax = plt.subplots(1, 1, figsize=(7, 7))

# Plot the final forest state's probability that a cell is burned
plot = ax.pcolor(burn_prob, cmap='afmhot', vmin=0, vmax=1)

# Plot format
ax.set_title(f'Nonlinear Model: Burn Probability over {nruns} Runs for Strong Breeze [8, 0]')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
cbar = plt.colorbar(plot, orientation='horizontal')
cbar.set_label('Burn Probability')

# Test wind speed case 4 #

# Define the parameters that will be passed into forest_fire for this test case
fire_params = dict(
    isize=11,
    jsize=11,
    nstep=10,
    p_base=0.2,
    p_bare=0.0,
    p_ignite=0.0,
    wind=[16, 0]
)
# Run the fire simulations and get the probability that a cell is burned
nruns = 10
burn_prob = run_fire_sims(nruns, **fire_params)

# Create the figure
fig, ax = plt.subplots(1, 1, figsize=(7, 7))

# Plot the final forest state's probability that a cell is burned
plot = ax.pcolor(burn_prob, cmap='afmhot', vmin=0, vmax=1)

# Plot format
ax.set_title(f'Nonlinear Model: Burn Probability over {nruns} Runs for Light Gale [16, 0]')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
cbar = plt.colorbar(plot, orientation='horizontal')
cbar.set_label('Burn Probability')

# Test wind speed case 5 #

# Define the parameters that will be passed into forest_fire for this test case
fire_params = dict(
    isize=11,
    jsize=11,
    nstep=10,
    p_base=0.2,
    p_bare=0.0,
    p_ignite=0.0,
    wind=[22, 0]
)
# Run the fire simulations and get the probability that a cell is burned
nruns = 10
burn_prob = run_fire_sims(nruns, **fire_params)

# Create the figure
fig, ax = plt.subplots(1, 1, figsize=(7, 7))

# Plot the final forest state's probability that a cell is burned
plot = ax.pcolor(burn_prob, cmap='afmhot', vmin=0, vmax=1)

# Plot format
ax.set_title(f'Nonlinear Model: Burn Probability over {nruns} Runs for Strong Gale [22, 0]')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
cbar = plt.colorbar(plot, orientation='horizontal')
cbar.set_label('Burn Probability')