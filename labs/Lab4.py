#!/usr/bin/ipython3

'''
For use in Lab 4, this program explores a forest fire spread model and applies it
to a disease transmission scenario, through use of pseudo-stochatic random chance
parameters, which include spread, bare (infected), and ignition (inital infection)
probabilities.

To reproduce the values and plots in my report, please simply run the script.
In the terminal, use the commands "ipython" and then "run Lab4.py". 
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numpy.random import rand

plt.style.use('seaborn-v0_8-colorblind')
plt.ion()
plt.close('all')

# Define the function to model a forest fire
def forest_fire(isize=3, jsize=3, nstep=4, spread_chance=1.0, bare_chance=0.0, ignite_chance=0.0):
    '''
    This function models a forest fire...

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
                    Set the probability that a cell will catch fire at the start of the simulation
    
    '''

    # Create a forest grid with each box representing a tree
    forest = np.zeros((nstep, isize, jsize)) + 2

    # Create the initial fire 
    if ignite_chance > 0:
        ignited = (np.random.rand(isize, jsize) < ignite_chance) & (forest[0] == 2)
        forest[0, ignited] = 3
    else:
        # Just center of forest catches fire
        forest[0, isize//2, jsize//2] = 3

    # Create initial bare forest cells
    isbare = np.random.rand(isize, jsize) # Create an array of randomly generated numbers to represent which cells begin bare
    isbare = isbare < bare_chance # Turn it into an array of True/False values
    forest[0, isbare] = 1 # Change forest cells to bare if the corresponding isbare boolean is true

    # Iterate through entire forest, identify fires, and spread fire as needed
    for k in range(nstep-1):

        # Assume the next time step is the same as the current
        forest[k+1, :, :] = forest[k, :, :]

        for i in range(isize):
            for j in range(jsize):

                # Check if a spot is on fire
                if forest[k, i, j] == 3: 

                    # Spread fire in each direction, based on the random chance and only spread to forest

                    if (i>0) and (forest[k, i-1, j] == 2) and (spread_chance > rand()):
                        forest[k+1, i-1, j] = 3 # spread North

                    if (i<isize-1) and (forest[k, i+1, j] == 2) and (spread_chance > rand()):
                        forest[k+1, i+1, j] = 3 # spread South

                    if (j>0) and (forest[k, i, j-1] == 2) and (spread_chance > rand()):
                        forest[k+1, i, j-1] = 3 # spread West
                    
                    if (j<jsize-1) and (forest[k, i, j+1] == 2) and (spread_chance > rand()):
                        forest[k+1, i, j+1] = 3 # spread East

                    # Change current burning trees to burnt in the next time step
                    forest[k+1, i, j] = 1

    return forest

def plot_fire(forest, nsteps=4, width=12, height=4):
    '''
    Given a forest of size (ntime, nx, ny)
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

def fire_summary(forest):
    '''
    Calculates various measures of a forest fire simulation.

    Parameters
    ----------

    Returns
    -------
    A dictionary of 
    '''

    # Get forest dimensions
    n, i, j = forest.shape

    # Record states per time step
    N_burn_t = np.zeros(n)
    N_bare_t = np.zeros(n)
    N_forest_t = np.zeros(n)
    for k in range(n):
        N_burn_t[k] = (forest[k] == 3).sum()
        N_bare_t[k] = (forest[k] == 1).sum()
        N_forest_t[k] = (forest[k] == 2).sum()

    # Calculate the final number of bare, forested, and burning cells
    N_bare = int(N_bare_t[-1])
    N_forest = int(N_forest_t[-1])
    N_burn = int(N_burn_t[-1])

    # Calculate final burned fraction (cells burned / area)
    burn_frac = N_bare / (i * j)

    # Calculate the number of steps needed for all cells to be bare or forested
    N_burn_n = np.zeros(n)
    steps_to_all_burn = n # default to the number of steps if the fire doesn't go out
    for k in range(n):
        N_burn_n[k] = (forest[k] == 3).sum()

        if (N_burn_n[k] == 0):
            steps_to_all_burn = k
            break

    # Caluclate the max number of cells on fire
    peak_fire = int(N_burn_t.max())

    return {
        'N_bare': N_bare,
        'N_forest': N_forest,
        'N_burn': N_burn,
        'burn_frac': burn_frac,
        'steps_to_all_burn': steps_to_all_burn,
        'peak_fire': peak_fire
    }


### Question 1 ###

# Plot 3 x 3 grid 
forest_state = forest_fire(isize=3, jsize=3, nstep=4, spread_chance=1.0, bare_chance=0.0, ignite_chance=0.0)
plot_fire(forest=forest_state, nsteps=4, width=12.5, height=4)

# Plot 5 x 3 grid 
forest_state2 = forest_fire(isize=3, jsize=5, nstep=4, spread_chance=1.0, bare_chance=0.0, ignite_chance=0.0)
plot_fire(forest=forest_state2, nsteps=4, width=16.5, height=3.5)


### Question 2 ###

# Define constants
nx, ny = 6, 6       # Number of cells in X and Y direction.
prob_spread = 1.0   # Chance to spread to adjacent cells.
prob_bare = 0.0     # Chance of cell to start as bare patch.
prob_ignite = 0.2   # Chance of cell to start on fire.
nstep = 10          # Number of steps
runs = 3            # Number of simulations to run

# Define arrays of spread and bare probabilties to test
prob_spread_arr = np.arange(0, 1.1, 0.1)
prob_bare_arr = np.arange(0, 1.1, 0.1)

# VARYING PROB_SPREAD #

# Set up figure to plot metrics
fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharex=True)

# Initialize arrays
all_burn_fracs = []
all_burn_times = []
all_peaks = []

# Run simulations varying prob_spread
for r in range(runs):

    spread_vary_summary = []

    for p in prob_spread_arr:
        forest_state = forest_fire(isize=nx, jsize=ny, nstep=nstep, spread_chance=p, bare_chance=prob_bare, ignite_chance=prob_ignite)
        #fig, ax = plot_fire(forest=forest_state, nsteps=4, width=12.5, height=4)
        #fig.suptitle(f'Forest Fire Progression over Time (spread probabilty = {p:.1f})', weight='bold', y=0.97) # Set heading title
        results = fire_summary(forest_state)
        summary = {'spread_chance': float(p)}
        summary.update(results)
        spread_vary_summary.append(summary)

    # Combine dictionary variables into arrays for printing and plotting
    x = np.array([m['spread_chance'] for m in spread_vary_summary])
    burn_fracs = np.array([m['burn_frac'] for m in spread_vary_summary])
    burn_times = np.array([m['steps_to_all_burn'] for m in spread_vary_summary])
    peaks = np.array([m['peak_fire'] for m in spread_vary_summary])

    # Append current run to the master arrays of metrics
    all_burn_fracs.append(burn_fracs)
    all_burn_times.append(burn_times)
    all_peaks.append(peaks)

    # Plot this current run on each subplot
    ax[0].plot(x, burn_fracs, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax[1].plot(x, burn_times, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax[2].plot(x, peaks, marker='o', alpha=0.7, label=f'Run {r+1}')

    # Print run summary
    print(f'Spread Variation Results for Run {r+1}:')
    print(f'Mean burned fraction: {burn_fracs.mean():.3f}')
    print(f'Median burned fraction: {np.median(burn_fracs):.3f}')
    print(f'Min burn_frac: {burn_fracs.min():.3f}, Max burn_frac: {burn_fracs.max():.3f}')
    print(f'Median time of burn: {np.median(burn_times):.2f}')
    print(f'Median peak cells on fire: {np.median(peaks):.2f}\n')

# Plot format
ax[0].set_title('Final Burned Fraction')
ax[1].set_title('Steps to Fire Extinction')
ax[2].set_title('Peak Number of Cells on Fire')
ax[0].set_xlabel('Spread Chance')
ax[0].set_ylabel('Fraction (number of burned cells/total area)')
ax[1].set_xlabel('Spread Chance')
ax[1].set_ylabel('Number of Steps')
ax[2].set_xlabel('Spread Chance')
ax[2].set_ylabel('Number of Cells')
fig.suptitle('Impact of Spread Probability on Forest Fires', y=0.95, weight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95]) # tight layout with room for title

# Create one legend
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', ncol=runs)


# VARYING PROB_BARE #

# Set up figure to plot metrics
fig2, ax2 = plt.subplots(1, 3, figsize=(14, 4), sharex=True)

# Initialize arrays
all_burn_fracs_b = []
all_burn_times_b = []
all_peaks_b = []

# Run simulations varying prob_bare
for r in range(runs):

    bare_vary_summary = []

    for p in prob_bare_arr:
        forest_state = forest_fire(isize=nx, jsize=ny, nstep=nstep, spread_chance=prob_spread, bare_chance=p, ignite_chance=prob_ignite)
        #fig, ax = plot_fire(forest=forest_state, nsteps=4, width=12.5, height=4)
        #fig.suptitle(f'Forest Fire Progression over Time (bare probabilty = {p:.1f})', weight='bold', y=0.97) # Set heading title
        results = fire_summary(forest_state)
        summary = {'bare_chance': float(p)}
        summary.update(results)
        bare_vary_summary.append(summary)

    # Combine dictionary variables into arrays for printing and plotting
    x = np.array([m['bare_chance'] for m in bare_vary_summary])
    burn_fracs = np.array([m['burn_frac'] for m in bare_vary_summary])
    burn_times = np.array([m['steps_to_all_burn'] for m in bare_vary_summary])
    peaks = np.array([m['peak_fire'] for m in bare_vary_summary])

    # Append current run to the master arrays of metrics
    all_burn_fracs_b.append(burn_fracs)
    all_burn_times_b.append(burn_times)
    all_peaks_b.append(peaks)

    # Plot this current run on each subplot
    ax2[0].plot(x, burn_fracs, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax2[1].plot(x, burn_times, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax2[2].plot(x, peaks, marker='o', alpha=0.7, label=f'Run {r+1}')

    # Print run summary
    print(f'Bare Variation Results for Run {r+1}:')
    print(f'Mean burned fraction: {burn_fracs.mean():.3f}')
    print(f'Median burned fraction: {np.median(burn_fracs):.3f}')
    print(f'Min burn_frac: {burn_fracs.min():.3f}, Max burn_frac: {burn_fracs.max():.3f}')
    print(f'Median time of burn: {np.median(burn_times):.2f}')
    print(f'Median peak cells on fire: {np.median(peaks):.2f}\n')

# Plot format
ax2[0].set_title('Final Burned Fraction')
ax2[1].set_title('Steps to Fire Extinction')
ax2[2].set_title('Peak Number of Cells on Fire')
ax2[0].set_xlabel('Chance to be Bare')
ax2[0].set_ylabel('Fraction (number of burned cells/total area)')
ax2[1].set_xlabel('Chance to be Bare')
ax2[1].set_ylabel('Number of Steps')
ax2[2].set_xlabel('Chance to be Bare')
ax2[2].set_ylabel('Number of Cells')
fig2.suptitle('Impact of Bare Cell Probability on Forest Fires', y=0.95, weight='bold')
fig2.tight_layout(rect=[0, 0, 1, 0.95]) # tight layout with room for title

# Create one legend
handles2, labels2 = ax2[0].get_legend_handles_labels()
fig2.legend(handles2, labels2, loc='upper right', ncol=runs)


### Question 3 ###

# Create an updated version of the forest fire model to apply to disease spread
def disease_spread(isize=3, jsize=3, nstep=4, spread_chance=1.0, bare_chance=0.0, ignite_chance=0.0):
    '''
    This function models a forest fire...

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
                    Set the probability that a cell will catch fire at the start of the simulation
    
    '''

    # Create a forest grid with each box representing a tree
    forest = np.zeros((nstep, isize, jsize)) + 2

    # Create the initial fire 
    if ignite_chance > 0:
        ignited = (np.random.rand(isize, jsize) < ignite_chance) & (forest[0] == 2)
        forest[0, ignited] = 3
    else:
        # Just center of forest catches fire
        forest[0, isize//2, jsize//2] = 3

    # Create initial bare forest cells
    isbare = np.random.rand(isize, jsize) # Create an array of randomly generated numbers to represent which cells begin bare
    isbare = isbare < bare_chance # Turn it into an array of True/False values
    forest[0, isbare] = 1 # Change forest cells to bare if the corresponding isbare boolean is true

    # Iterate through entire forest, identify fires, and spread fire as needed
    for k in range(nstep-1):

        # Assume the next time step is the same as the current
        forest[k+1, :, :] = forest[k, :, :]

        for i in range(isize):
            for j in range(jsize):

                # Check if a spot is on fire
                if forest[k, i, j] == 3: 

                    # Spread fire in each direction, based on the random chance and only spread to forest

                    if (i>0) and (forest[k, i-1, j] == 2) and (spread_chance > rand()):
                        forest[k+1, i-1, j] = 3 # spread North

                    if (i<isize-1) and (forest[k, i+1, j] == 2) and (spread_chance > rand()):
                        forest[k+1, i+1, j] = 3 # spread South

                    if (j>0) and (forest[k, i, j-1] == 2) and (spread_chance > rand()):
                        forest[k+1, i, j-1] = 3 # spread West
                    
                    if (j<jsize-1) and (forest[k, i, j+1] == 2) and (spread_chance > rand()):
                        forest[k+1, i, j+1] = 3 # spread East

                    # Change current burning trees to burnt in the next time step
                    forest[k+1, i, j] = 1

    return forest


'''
Discussion:

Does not capture real physics of what happens in a forest fire

Average may be okay sometimes, but other factors have much larger effect
'''



