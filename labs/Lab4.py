#!/usr/bin/ipython3

'''
For use in Lab 4, this program explores a forest fire spread model and applies it
to a disease spread scenario, through use of pseudostochastic random chance
parameters, which include spread, bare (infected), and ignition (inital infection)
probabilities.

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

# Define the function to model a forest fire
def forest_fire(isize=3, jsize=3, nstep=4, spread_chance=1.0, bare_chance=0.0, ignite_chance=0.0):
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
    Calculates various measures of a forest fire simulation using 
    the counts of bare, forest, and burning cells over time.

    Parameters
    ----------
    forest: array
            3D array of forest states, returned by forest_fire function

    Returns
    -------
    summary: dict
             Dictionary of...
                'N_bare': final number of bare/burnt cells
                'N_forest': final number of forest cells
                'N_burn': final number of burning cells
                'burn_frac': final burned fraction (N_bare / total cells)
                'steps_to_all_burn': time step when no cells are burning
                'peak_fire': maximum number of burning cells at a time
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
fig, ax = plt.subplots(2, 3, figsize=(14, 8), sharex=True)

# Initialize arrays
all_burn_fracs = []
all_burn_times = []
all_peaks = []
all_final_burn = []
all_final_bare = []
all_final_forest = []

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
    num_burn = np.array([m['N_burn'] for m in spread_vary_summary])
    num_bare = np.array([m['N_bare'] for m in spread_vary_summary])
    num_forest = np.array([m['N_forest'] for m in spread_vary_summary])

    # Append current run to the master arrays of metrics
    all_burn_fracs.append(burn_fracs)
    all_burn_times.append(burn_times)
    all_peaks.append(peaks)
    all_final_burn.append(num_burn)
    all_final_bare.append(num_bare)
    all_final_forest.append(num_forest)

    # Plot this current run on each subplot
    ax[0,0].plot(x, burn_fracs, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax[0,1].plot(x, burn_times, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax[0,2].plot(x, peaks, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax[1,0].plot(x, num_burn, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax[1,1].plot(x, num_bare, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax[1,2].plot(x, num_forest, marker='o', alpha=0.7, label=f'Run {r+1}')

    # Print run summary
    print(f'Spread Variation Results for Run {r+1}:')
    print(f'Mean burned fraction: {burn_fracs.mean():.3f}')
    print(f'Median burned fraction: {np.median(burn_fracs):.3f}')
    print(f'Min burn_frac: {burn_fracs.min():.3f}, Max burn_frac: {burn_fracs.max():.3f}')
    print(f'Median time of burn: {np.median(burn_times):.2f}')
    print(f'Median peak cells on fire: {np.median(peaks):.2f}\n')

# Plot format
ax[0,0].set_title('Final Burned Fraction')
ax[0,1].set_title('Steps to Fire Extinction')
ax[0,2].set_title('Peak Number of Cells on Fire')
ax[1,0].set_title('Final Number of Burning Cells')
ax[1,1].set_title('Final Number of Bare Cells')
ax[1,2].set_title('Final Number of Forested Cells')
ax[0,0].set_ylabel('Fraction (number of burned cells/total area)')
ax[0,1].set_ylabel('Number of Steps')
ax[0,2].set_ylabel('Number of Cells')
ax[1,0].set_xlabel('Spread Chance')
ax[1,0].set_ylabel('Number of Cells')
ax[1,1].set_xlabel('Spread Chance')
ax[1,1].set_ylabel('Number of Cells')
ax[1,2].set_xlabel('Spread Chance')
ax[1,2].set_ylabel('Number of Cells')
fig.suptitle('Impact of Spread Probability on Forest Fires', y=0.95, weight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95]) # tight layout with room for title

# Create one legend
handles, labels = ax[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', ncol=runs)


# VARYING PROB_BARE #

# Set up figure to plot metrics
fig2, ax2 = plt.subplots(2, 3, figsize=(14, 8), sharex=True)

# Initialize arrays
all_burn_fracs_b = []
all_burn_times_b = []
all_peaks_b = []
all_final_burn_b = []
all_final_bare_b = []
all_final_forest_b = []

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
    num_burn = np.array([m['N_burn'] for m in bare_vary_summary])
    num_bare = np.array([m['N_bare'] for m in bare_vary_summary])
    num_forest = np.array([m['N_forest'] for m in bare_vary_summary])

    # Append current run to the master arrays of metrics
    all_burn_fracs_b.append(burn_fracs)
    all_burn_times_b.append(burn_times)
    all_peaks_b.append(peaks)
    all_final_burn_b.append(num_burn)
    all_final_bare_b.append(num_bare)
    all_final_forest_b.append(num_forest)

    # Plot this current run on each subplot
    ax2[0,0].plot(x, burn_fracs, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax2[0,1].plot(x, burn_times, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax2[0,2].plot(x, peaks, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax2[1,0].plot(x, num_burn, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax2[1,1].plot(x, num_bare, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax2[1,2].plot(x, num_forest, marker='o', alpha=0.7, label=f'Run {r+1}')


    # Print run summary
    print(f'Initial Bare Variation Results for Run {r+1}:')
    print(f'Mean burned fraction: {burn_fracs.mean():.3f}')
    print(f'Median burned fraction: {np.median(burn_fracs):.3f}')
    print(f'Min burn_frac: {burn_fracs.min():.3f}, Max burn_frac: {burn_fracs.max():.3f}')
    print(f'Median time of burn: {np.median(burn_times):.2f}')
    print(f'Median peak cells on fire: {np.median(peaks):.2f}\n')

# Plot format
ax2[0,0].set_title('Final Burned Fraction')
ax2[0,1].set_title('Steps to Fire Extinction')
ax2[0,2].set_title('Peak Number of Cells on Fire')
ax2[1,0].set_title('Final Number of Burning Cells')
ax2[1,1].set_title('Final Number of Bare Cells')
ax2[1,2].set_title('Final Number of Forested Cells')
ax2[0,0].set_ylabel('Fraction (number of burned cells/total area)')
ax2[0,1].set_ylabel('Number of Steps')
ax2[0,2].set_ylabel('Number of Cells')
ax2[1,0].set_xlabel('Bare Chance')
ax2[1,0].set_ylabel('Number of Cells')
ax2[1,1].set_xlabel('Bare Chance')
ax2[1,1].set_ylabel('Number of Cells')
ax2[1,2].set_xlabel('Bare Chance')
ax2[1,2].set_ylabel('Number of Cells')
fig2.suptitle('Impact of Initial Bare Probability on Forest Fires', y=0.95, weight='bold')
fig2.tight_layout(rect=[0, 0, 1, 0.95]) # tight layout with room for title

# Create one legend
handles2, labels2 = ax2[0,0].get_legend_handles_labels()
fig2.legend(handles2, labels2, loc='upper right', ncol=runs)


### Question 3 ###

# Create an updated version of the forest fire model to apply to disease spread
def disease_spread(isize=3, jsize=3, nstep=4, spread_chance=1.0, immune_chance=0.0, infect_chance=0.0, fatal_chance=0.5):
    '''
    This function models the spread of an infectious disease on a 2D grid. Each cell can be
    dead (state 0), immune (state 1), healthy (state 2), or actively infected (state 3).

    Parameters
    ----------
    isize, jsize:   int, defaults = 3
                    Set the size of the population in x and y direction, respectively
    nstep:          int, default = 4
                    Set the number of steps to advance solution
    spread_chance:  float, default = 1.0
                    Set the probability that infection can spread in any direction, from 0 to 1 (0 to 100%)
    immune_chance:  float, default = 0.0
                    Set the probability that a cell (person) is naturally immune to begin with
    infect_chance:  float, default = 0.0
                    Set the probability that a cell (person) will be infected at the start of the simulation.
                    If 0, the center cell is set as infected instead.
    fatal_chance:   float, default = 0.5
                    Set the probability that an infected cell (person) dies instead of recovering and becoming immune.

    Returns
    -------
    population: array
                Population state at each time step.
    '''

    # Create a population grid with each box representing a tree
    population = np.zeros((nstep, isize, jsize)) + 2

    # Create the initial fire 
    if infect_chance > 0:
        infected = (np.random.rand(isize, jsize) < infect_chance) & (population[0] == 2)
        population[0, infected] = 3
    else:
        # Just center of population catches fire
        population[0, isize//2, jsize//2] = 3

    # Create initial immune population cells
    isimmune = np.random.rand(isize, jsize) # Create an array of randomly generated numbers to represent which cells begin immune
    isimmune = isimmune < immune_chance # Turn it into an array of True/False values
    population[0, isimmune] = 1 # Change population cells to immune if the corresponding isimmune boolean is true

    # Iterate through entire population, identify infected people, and spread disease as needed
    for k in range(nstep-1):

        # Assume the next time step is the same as the current
        population[k+1, :, :] = population[k, :, :]

        for i in range(isize):
            for j in range(jsize):

                # Check if a spot is on fire
                if population[k, i, j] == 3: 

                    # Spread disease in each direction, based on the random chance and only spread to non-immune and alive population

                    if (i>0) and (population[k, i-1, j] == 2) and (spread_chance > rand()):
                        population[k+1, i-1, j] = 3 # spread Up

                    if (i<isize-1) and (population[k, i+1, j] == 2) and (spread_chance > rand()):
                        population[k+1, i+1, j] = 3 # spread Down

                    if (j>0) and (population[k, i, j-1] == 2) and (spread_chance > rand()):
                        population[k+1, i, j-1] = 3 # spread Left
                    
                    if (j<jsize-1) and (population[k, i, j+1] == 2) and (spread_chance > rand()):
                        population[k+1, i, j+1] = 3 # spread Right

                    # Change current infected person to immune OR DEAD in the next time step
                    if fatal_chance > rand():
                        population[k+1, i, j] = 0 # Person is dead because random number was within mortality threshold
                    else:
                        population[k+1, i, j] = 1

    return population

def plot_disease(population, nsteps=4, width=12, height=4):
    '''
    Given a disease spread simulation of size (ntime, nx, ny), plot the evolution of 
    the population over time. Creates a row of subplots showing the population state
    at each time step.

    Parameters
    ----------
    population: array
                3D array of population states, returned by disease_spread function
    nsteps:     int, default = 4
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
    # those names. We have 4 fundamental states, so we want only 4 colors.
    population_cmap = ListedColormap(['dimgray', 'royalblue', 'gold', 'crimson'])

    # Create the figure and plot the population state for each time step
    fig, ax = plt.subplots(1, nsteps, figsize=(width, height), sharey=True)
    for k in range(nsteps):
        ax[k].pcolor(population[k], cmap=population_cmap, vmin=0, vmax=3)
        ax[k].set_title(f'Population State (Step {k})')
        ax[k].set_xlabel('X (person)')
    
    # Other formatting
    ax[0].set_ylabel('Y (person)') # Set shared y-axis label
    fig.tight_layout(rect=[0, 0, 1, 0.87]) # tight layout with room for title
    fig.suptitle('Disease Spread Progression over Time', weight='bold', y=0.97) # Set heading title

    # Create a legend
    ax[0].plot([], [], lw=3, color='dimgray', label='Dead')
    ax[0].plot([], [], lw=3, color='royalblue', label='Immune')
    ax[0].plot([], [], lw=3, color='gold', label='Healthy')
    ax[0].plot([], [], lw=3, color='crimson', label='Infected')
    fig.legend(ncol=2)

    return fig, ax

def disease_summary(population):
    '''
    Calculates various measures of a disease spread simulation using 
    the counts of healthy, immune, infected, and dead cells over time.

    Parameters
    ----------
    population: array
                3D array of population states, returned by disease_spread function

    Returns
    -------
    summary: dict
             Dictionary of...
                'N_immune': final number of immune cells
                'N_healthy': final number of healthy cells
                'N_infect': final number of infected cells
                'N_dead': final number of dead cells
                'dead_frac': final dead fraction (N_dead / total cells)
                'immune_frac': final immune fraction (N_immune / total cells)
                'disease_extinction': time step when no infected cells remain
                'peak_infection': maximum number of infected cells at a time
    '''

    # Get population dimensions
    n, i, j = population.shape

    # Record states per time step
    N_infect_t = np.zeros(n)
    N_immune_t = np.zeros(n)
    N_healthy_t = np.zeros(n)
    N_dead_t = np.zeros(n)
    for k in range(n):
        N_infect_t[k] = (population[k] == 3).sum()
        N_immune_t[k] = (population[k] == 1).sum()
        N_healthy_t[k] = (population[k] == 2).sum()
        N_dead_t[k] = (population[k] == 0).sum()

    # Calculate the final number of healthy, immune, infected, and dead cells
    N_healthy = int(N_healthy_t[-1])
    N_immune = int(N_immune_t[-1])
    N_infect = int(N_infect_t[-1])
    N_dead = int(N_dead_t[-1])

    # Calculate final dead fraction (cells dead / area)
    dead_frac = N_dead / (i * j)

    # Calculate final immune fraction (cells immune / area)
    immune_frac = N_immune / (i * j)

    # Calculate the number of steps needed for no disease to remain
    N_infect_n = np.zeros(n)
    disease_extinction = n # default to the number of steps if the disease doesn't end
    for k in range(n):
        N_infect_n[k] = (population[k] == 3).sum()

        if (N_infect_n[k] == 0):
            disease_extinction = k
            break

    # Caluclate the max number of people infected
    peak_infection = int(N_infect_t.max())

    return {
        'N_immune': N_immune,
        'N_healthy': N_healthy,
        'N_infect': N_infect,
        'N_dead': N_dead,
        'dead_frac': dead_frac,
        'immune_frac': immune_frac,
        'disease_extinction': disease_extinction,
        'peak_infection': peak_infection
    }


# Plot 3 x 3 population-disease spread grid 
population_state = disease_spread(isize=3, jsize=3, nstep=4, spread_chance=0.8, immune_chance=0.1, infect_chance=0.0, fatal_chance=0.3)
plot_disease(population=population_state, nsteps=4, width=12.5, height=4)

# Define constants
nx, ny = 6, 6       # Number of cells in X and Y direction.
prob_spread = 0.8   # Chance to spread to adjacent cells.
prob_immune = 0.0   # Chance of cell to start as immune.
prob_infect = 0.2   # Chance of cell to start infected.
prob_fatal = 0.3    # Chance of cell to die from being infected.
nstep = 10          # Number of steps
runs = 3            # Number of simulations to run

# Define arrays of mortality and immunity probabilties to test
prob_mort_arr = np.arange(0, 1.1, 0.1)
prob_immu_arr = np.arange(0, 1.1, 0.1)

# VARYING MORTALITY #

# Set up figure to plot metrics
fig, ax = plt.subplots(2, 4, figsize=(18, 8), sharex=True)

# Initialize arrays
all_immune_fracs = []
all_dead_fracs = []
all_disease_ext = []
all_peaks = []
all_final_infect = []
all_final_immune = []
all_final_healthy = []
all_final_dead = []

# Run simulations varying prob_mortality
for r in range(runs):

    mort_vary_summary = []

    for p in prob_mort_arr:
        population_state = disease_spread(isize=nx, jsize=ny, nstep=nstep, spread_chance=prob_spread, immune_chance=prob_immune, infect_chance=prob_infect, fatal_chance=p)
        results = disease_summary(population_state)
        summary = {'mort_chance': float(p)}
        summary.update(results)
        mort_vary_summary.append(summary)

    # Combine dictionary variables into arrays for printing and plotting
    x = np.array([m['mort_chance'] for m in mort_vary_summary])
    immune_fracs = np.array([m['immune_frac'] for m in mort_vary_summary])
    dead_fracs = np.array([m['dead_frac'] for m in mort_vary_summary])
    disease_ext = np.array([m['disease_extinction'] for m in mort_vary_summary])
    peaks = np.array([m['peak_infection'] for m in mort_vary_summary])
    num_infect = np.array([m['N_infect'] for m in mort_vary_summary])
    num_immune = np.array([m['N_immune'] for m in mort_vary_summary])
    num_healthy = np.array([m['N_healthy'] for m in mort_vary_summary])
    num_dead = np.array([m['N_dead'] for m in mort_vary_summary])

    # Append current run to the master arrays of metrics
    all_immune_fracs.append(immune_fracs)
    all_dead_fracs.append(dead_fracs)
    all_disease_ext.append(disease_ext)
    all_peaks.append(peaks)
    all_final_infect.append(num_infect)
    all_final_immune.append(num_immune)
    all_final_healthy.append(num_healthy)
    all_final_dead.append(num_dead)

    # Plot this current run on each subplot
    ax[0,0].plot(x, immune_fracs, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax[0,1].plot(x, dead_fracs, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax[0,2].plot(x, disease_ext, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax[0,3].plot(x, peaks, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax[1,0].plot(x, num_immune, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax[1,1].plot(x, num_dead, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax[1,2].plot(x, num_healthy, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax[1,3].plot(x, num_infect, marker='o', alpha=0.7, label=f'Run {r+1}')

    # Print run summary
    print(f'Mortality Variation Results for Run {r+1}:')
    print(f'Median immune fraction: {np.median(immune_fracs):.3f}')
    print(f'Median dead fraction: {np.median(dead_fracs):.3f}')
    print(f'Min immune fraction: {immune_fracs.min():.3f}, Max immune fraction: {immune_fracs.max():.3f}')
    print(f'Min dead fraction: {dead_fracs.min():.3f}, Max dead fraction: {dead_fracs.max():.3f}')
    print(f'Median time to disease extinction: {np.median(disease_ext):.2f}')
    print(f'Median peak cells infected: {np.median(peaks):.2f}\n')

# Plot format
ax[0,0].set_title('Final Immune Fraction')
ax[0,1].set_title('Final Dead Fraction')
ax[0,2].set_title('Steps to Disease Extinction')
ax[0,3].set_title('Peak Number of Infected Cells')
ax[1,0].set_title('Final Number of Immune Cells')
ax[1,1].set_title('Final Number of Dead Cells')
ax[1,2].set_title('Final Number of Healthy Cells')
ax[1,3].set_title('Final Number of Infected Cells')
ax[0,0].set_ylabel('Fraction (number of immune cells/total area)')
ax[0,1].set_ylabel('Fraction (number of dead cells/total area)')
ax[0,2].set_ylabel('Number of Steps')
ax[0,3].set_ylabel('Number of Cells')
ax[1,0].set_xlabel('Mortality Chance')
ax[1,0].set_ylabel('Number of Cells')
ax[1,1].set_xlabel('Mortality Chance')
ax[1,1].set_ylabel('Number of Cells')
ax[1,2].set_xlabel('Mortality Chance')
ax[1,2].set_ylabel('Number of Cells')
ax[1,3].set_xlabel('Mortality Chance')
ax[1,3].set_ylabel('Number of Cells')
fig.suptitle('Impact of Mortality Probability on Disease Spread within a Population', y=0.95, weight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95]) # tight layout with room for title

# Create one legend
handles, labels = ax[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', ncol=runs)


# VARYING INITIAL IMMUNITY #

# Set up figure to plot metrics
fig, ax = plt.subplots(2, 4, figsize=(18, 8), sharex=True)

# Initialize arrays
all_immune_fracs = []
all_dead_fracs = []
all_disease_ext = []
all_peaks = []
all_final_infect = []
all_final_immune = []
all_final_healthy = []
all_final_dead = []

# Run simulations varying initial immunity
for r in range(runs):

    imm_vary_summary = []

    for p in prob_mort_arr:
        population_state = disease_spread(isize=nx, jsize=ny, nstep=nstep, spread_chance=prob_spread, immune_chance=p, infect_chance=prob_infect, fatal_chance=prob_fatal)
        results = disease_summary(population_state)
        summary = {'immune_chance': float(p)}
        summary.update(results)
        imm_vary_summary.append(summary)

    # Combine dictionary variables into arrays for printing and plotting
    x = np.array([m['immune_chance'] for m in imm_vary_summary])
    immune_fracs = np.array([m['immune_frac'] for m in imm_vary_summary])
    dead_fracs = np.array([m['dead_frac'] for m in imm_vary_summary])
    disease_ext = np.array([m['disease_extinction'] for m in imm_vary_summary])
    peaks = np.array([m['peak_infection'] for m in imm_vary_summary])
    num_infect = np.array([m['N_infect'] for m in imm_vary_summary])
    num_immune = np.array([m['N_immune'] for m in imm_vary_summary])
    num_healthy = np.array([m['N_healthy'] for m in imm_vary_summary])
    num_dead = np.array([m['N_dead'] for m in imm_vary_summary])

    # Append current run to the master arrays of metrics
    all_immune_fracs.append(immune_fracs)
    all_dead_fracs.append(dead_fracs)
    all_disease_ext.append(disease_ext)
    all_peaks.append(peaks)
    all_final_infect.append(num_infect)
    all_final_immune.append(num_immune)
    all_final_healthy.append(num_healthy)
    all_final_dead.append(num_dead)

    # Plot this current run on each subplot
    ax[0,0].plot(x, immune_fracs, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax[0,1].plot(x, dead_fracs, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax[0,2].plot(x, disease_ext, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax[0,3].plot(x, peaks, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax[1,0].plot(x, num_immune, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax[1,1].plot(x, num_dead, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax[1,2].plot(x, num_healthy, marker='o', alpha=0.7, label=f'Run {r+1}')
    ax[1,3].plot(x, num_infect, marker='o', alpha=0.7, label=f'Run {r+1}')

    # Print run summary
    print(f'Initial Immunity Variation Results for Run {r+1}:')
    print(f'Median immune fraction: {np.median(immune_fracs):.3f}')
    print(f'Median dead fraction: {np.median(dead_fracs):.3f}')
    print(f'Min immune fraction: {immune_fracs.min():.3f}, Max immune fraction: {immune_fracs.max():.3f}')
    print(f'Min dead fraction: {dead_fracs.min():.3f}, Max dead fraction: {dead_fracs.max():.3f}')
    print(f'Median time to disease extinction: {np.median(disease_ext):.2f}')
    print(f'Median peak cells infected: {np.median(peaks):.2f}\n')

# Plot format
ax[0,0].set_title('Final Immune Fraction')
ax[0,1].set_title('Final Dead Fraction')
ax[0,2].set_title('Steps to Disease Extinction')
ax[0,3].set_title('Peak Number of Infected Cells')
ax[1,0].set_title('Final Number of Immune Cells')
ax[1,1].set_title('Final Number of Dead Cells')
ax[1,2].set_title('Final Number of Healthy Cells')
ax[1,3].set_title('Final Number of Infected Cells')
ax[0,0].set_ylabel('Fraction (number of immune cells/total area)')
ax[0,1].set_ylabel('Fraction (number of dead cells/total area)')
ax[0,2].set_ylabel('Number of Steps')
ax[0,3].set_ylabel('Number of Cells')
ax[1,0].set_xlabel('Immunity Chance')
ax[1,0].set_ylabel('Number of Cells')
ax[1,1].set_xlabel('Immunity Chance')
ax[1,1].set_ylabel('Number of Cells')
ax[1,2].set_xlabel('Immunity Chance')
ax[1,2].set_ylabel('Number of Cells')
ax[1,3].set_xlabel('Immunity Chance')
ax[1,3].set_ylabel('Number of Cells')
fig.suptitle('Impact of Initial Immune Probability on Disease Spread within a Population', y=0.95, weight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95]) # tight layout with room for title

# Create one legend
handles, labels = ax[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', ncol=runs)