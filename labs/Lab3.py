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

plt.style.use('seaborn-v0_8-colorblind')
plt.ion()
plt.close('all')

# Function developed in class
# Define a function to solve the heat equation
def solve_heat(xstop=1, tstop=0.2, dx=0.2, dt=0.02, c2=1, U_init=None, lowerbound=0, upperbound=0):
    '''
    Solves the heat equation via a forward-difference method. Takes time and space steps 
    and endpoints, initial conditions, and boundary conditions. Returns 1D arrays of space
    and time, and a 2D array of temperatures, effectively modeling how heat diffuses over 
    time through a column.

    Parameters
    ----------
    xstop:  float
            The maximum space coordinate (representing depth in m), default = 1.
        
    tstop:  float
            The maximum time coordinate (in seconds or days), default = 0.2.

    dx:     float
            Spatial step size (m), default = 0.2.

    dt:     float
            Time step size (s or days), default = 0.02. 

    c2:     float
            c^2, the square of the diffusion coefficient (m^2/s or m^2/day), 
            representing thermal diffusivity. The default = 1. 
    
    U_init: Numpy array or None
            The inital temperature profile. If None, the example problem's 
            profile 4x - 4x^2 is used. The default = None. 

    lowerbound: float, none, or callable
                The lower boundary at x = 0. 
                    If a float, a Dirichlet or constant temperature is used. 
                    If none, a Neumann or zero gradient is used. 
                    If callable, a function returning temperature is used. 
                The default = 0.

    upperbound: float, none, or callable
                The upper boundary at x = xstop. 
                    If a float, a Dirichlet or constant temperature is used. 
                    If none, a Neumann or zero gradient is used. 
                    If callable, a function returning temperature is used. 
                The default = 0.

    Returns
    -------
    x, t:   1D Numpy arrays
            Space and time values, respectively.

    U:      Numpy array
            The solution of the heat equation, size is nSpace x nTime.
    '''

    # Check stability criteria
    dt_max = dx**2 / (2*c2)
    if dt > dt_max:
        print('Values used in this model:')
        print(f'\tdt = {dt}\n\tc2 = {c2}\n\tdx = {dx}')
        raise ValueError(f'DANGER: dt={dt} > dt_max={dt_max}')

    # Get grid sizes (needs plus 1 to include 0 as well)
    N = int(tstop / dt) + 1
    M = int(xstop / dx) + 1

    # Set up space and time grid
    t = np.linspace(0, tstop, N)
    x = np.linspace(0, xstop, M)

    # Create the solution matrix
    U = np.zeros([M, N])

    # Set initial conditions
    if U_init is None:
        U[:, 0] = 4*x - 4*x**2 # default initial condition based on example problem
    else:
        U[:, 0] = U_init

    # Get the "r" coefficient
    r = c2 * (dt/dx**2)

    # Solve the equation
    for j in range(N-1): # Time loop, starting at 0
        U[1:M-1, j+1] = (1-2*r) * U[1:M-1, j] + r*(U[2:M, j] + U[:M-2, j])

        # Apply Neumann or Dirichlet boundary conditions
        # Set Neumann boundary conditions if upper/lower bounds are none (not constants)
        if lowerbound is None:
            U[0, j+1] = U[1, j+1] # Neumann: j+1 is the current time
        elif callable(lowerbound):
            U[0, j+1] = lowerbound(t[j+1]) # function must take time and returns one value
        else:
            U[0, j+1] = lowerbound # Dirichlet

        if upperbound is None:
            U[-1, j+1] = U[-2, j+1]
        elif callable(upperbound):
            U[-1, j+1] = upperbound(t[j+1]) 
        else:
            U[-1, j+1] = upperbound

    # Return the solution to the caller
    return t, x, U

# Define Kangerlussuaq average temperatures
t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4, 10.7, 8.5, 3.1, -6.0, -12.0, -16.9])

# Function provided in Lab 3 pdf
# Define a function to create a continuous model of Kangerlussuaq temperatures
def temp_kanger(t, warming=0):
    '''
    Uses an array of average surface temperatures from Kangerlussuaq, 
    Greenland to produce a time series array of seasonal temperatures 
    over the course of a given number of days. Can shift temperatures 
    to simulate global warming conditions.

    Parameters
    ----------
    t:          Numpy array
                Array of times (days), no default.

    warming:    float
                Degrees Celsius to shift temperature by, default = 0.

    Returns
    -------
    Time series of temperature for Kangerlussuaq, Greenland.
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()

    return (t_amp*np.sin(np.pi/180 * t - np.pi/2) + t_kanger.mean()) + warming

# Function developed in class
# Define a function to plot the 2D heat equation solution
def plot_heatsolve(t, x, U, title=None, **kwargs):
    '''
    Plots and formats the 2D solution for the solve_heat function.
    Extra kwargs are handed to pcolor.

    Paramters
    ---------
    t, x:   1D Numpy arrays
            Space and time values, respectively.

    U:      Numpy array
            The solution of the heat equation, size is nSpace x nTime.

    title:  str, default is None
            Title for the figure.

    **kwargs:   dict
                Additional keyword arguments to be passed into the pcolor function.

    Returns
    -------
    fig, ax:    Matplotlib figure & axes objects
                The figure and axes of the plot

    cbar:       Matplotlib color bar object
                The color bar on the final plot
    '''

    # Check the kwargs for any defaults
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'hot' # Set default cmap to hot if no other cmap used

    # Create and configure the figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Add contours to the axes
    contour = ax.pcolor(t/365.25, x, U, **kwargs)
    cbar = plt.colorbar(contour)

    # Other format and labels
    cbar.set_label(r'Temperature ($^{\circ}C$)')
    ax.set_xlabel('Time ($years$)')
    ax.set_ylabel('Depth ($m$)')
    ax.invert_yaxis()
    ax.set_title(title)

    # Improve layout
    fig.tight_layout()

    return fig, ax, cbar


### Question 1 ###

# Solution to example heat problem (10.3 from Fink/Matthews)
sol10p3 = [[0.000000, 0.640000, 0.960000, 0.960000, 0.640000, 0.000000],
           [0.000000, 0.480000, 0.800000, 0.800000, 0.480000, 0.000000],
           [0.000000, 0.400000, 0.640000, 0.640000, 0.400000, 0.000000],
           [0.000000, 0.320000, 0.520000, 0.520000, 0.320000, 0.000000],
           [0.000000, 0.260000, 0.420000, 0.420000, 0.260000, 0.000000],
           [0.000000, 0.210000, 0.340000, 0.340000, 0.210000, 0.000000],
           [0.000000, 0.170000, 0.275000, 0.275000, 0.170000, 0.000000],
           [0.000000, 0.137500, 0.222500, 0.222500, 0.137500, 0.000000],
           [0.000000, 0.111250, 0.180000, 0.180000, 0.111250, 0.000000],
           [0.000000, 0.090000, 0.145625, 0.145625, 0.090000, 0.000000],
           [0.000000, 0.072812, 0.117813, 0.117813, 0.072812, 0.000000]]

# Convert this solution to an array and transpose it to get correct ordering
sol10p3 = np.array(sol10p3).transpose()

# Solve example problem with the solve_heat function 
# The function's defaults align with the example problem
t, x, U = solve_heat()

# Print example solution and my solution to validate results
print(f'Example solution from solve_heat function:\n{np.array_str(U, precision=3)}\n')
print(f'Provided example solution:\n{np.array_str(sol10p3, precision=3)}\n')

# Compare the solution values in U and sol10p3 to see if they match
if np.allclose(U, sol10p3, atol=1e-5): # atol is the allowed tolerance to account for rounding errors
    print('All values match: heat_solve function validated.\n')
else:
    print('Mismatching values found.\n')


### Question 2 ###

# Define parameters (in days instead of seconds)
c2 = 0.25 * 0.0864 # m^2/day (instead of 8640 / 1000**2, just put 0.0864.)
dx = 1.0                    # m
dt = 10.0                   # day 
xstop = 100                 # m
tstop = 365*100             # days (100 yrs)

# Solve and plot the heat equation based on the conditions defined above
t, x, U = solve_heat(xstop=xstop, tstop=tstop+1, dx=dx, dt=dt, c2=c2, U_init=0, lowerbound=temp_kanger, upperbound=5)
plot_heatsolve(t, x, U, title='Kangerlussuaq Ground Temperature Profile over Time', cmap='RdBu_r', vmin=-15, vmax=15)

# Create temperature vs depth / seasonal temperature profiles

# Set indexing for the final year of results
loc = int(-365/dt) # final 365 days of the result 
# Extract the min values over the final year
winter = U[:, loc:].min(axis=1)
# Extract the min values over the final year
summer = U[:, loc:].max(axis=1)

# Plot the summer and winter temperature profiles
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
ax2.plot(winter, x, color='dodgerblue', label='Winter')
ax2.plot(summer, x, color='tomato', linestyle='--', label='Summer')

# Plot format
ax2.set_xlabel('Temperature ($^{\circ}C$)')
ax2.set_xlim(-8, 8)
ax2.set_ylabel('Depth ($m$)')
ax2.invert_yaxis()
ax2.set_title('Kangerlussuaq Seasonal Ground Temperature Profile')
ax2.grid(True)
ax2.legend()
fig2.tight_layout()

# Determine when steady state is reached

# Define step, index of middle layer, and year when steady state is reached
steps_yr = int(365/dt)  # time steps per year
steady_yr = None        # empty 
yr_threshold = 5        # number of years that need to be unchanging
z_min, z_max = 10, 90   # top and bottom depths for the gradient to be calculated for
depth_idx = np.where((x >= z_min) & (x <= z_max))[0] # get indices for the gradient

# Loop through each year
for yr in range(yr_threshold + 1, int(t[-1]/365)):
    # Get current and previous steps
    start = (yr-1)*steps_yr
    end = yr*steps_yr
    prev_start = (yr-1-yr_threshold)*steps_yr
    prev_end = (yr-yr_threshold)*steps_yr

    # Compute gradient with depth for current and previous year
    grad_now = np.diff(U[depth_idx, start:end], axis=0)
    grad_prev = np.diff(U[depth_idx, prev_start:prev_end], axis=0)
    
    # Max change in gradient between consecutive years
    max_grad_change = np.max(np.abs(grad_now - grad_prev))
    
    # Get steady state year based on certain tolerance
    if max_grad_change < 0.01:
        steady_yr = yr
        break

# Print results
print('For current temperatures:')
if steady_yr is not None:
    print(f'Steady state is reached after ~ {steady_yr} years.')
else:
    print(f'Steady state not reached within this timeframe.')

# Determine the depths of the active and permafrost layers (if steady state reached)

if steady_yr is not None:
    # Get the temperature profile of the year steady state is reached
    steady_step = steady_yr * steps_yr
    start = steady_step - steps_yr
    end = steady_step
    T_yr = U[:, start:end]

    # Calculate summer as the max temps and winter as the min temps for the steady state year
    T_summer = T_yr.max(axis=1)
    T_winter = T_yr.min(axis=1)

    active_idx = np.where((T_summer > 0) & (T_winter < 0))[0] # Indices where ground melts and refreezes
    active_bot = x[active_idx[-1]] # Last depth ground melts and refreezes (bottom of active layer and top of permafrost)
    pfrst_bot = x[T_summer <= 0][-1] # Last depth where ground is frozen

    # Print the active and permafrost layer depths
    print(f'Active layer depth: {active_bot:.2f} m')
    print(f'Permafrost depth: {pfrst_bot:.2f} m')


### Question 3 ###

# Define parameters
c2 = 0.25 * 0.0864 # m^2/day
dx = 1.0                    # m
dt = 10.0                   # day 
xstop = 100                 # m
tstop = 365*100             # days (50 yrs)

# Define temperature shifts
shifts = [0.5, 1, 3]

# Create the figure
fig3, ax3 = plt.subplots(len(shifts), 2, figsize=(15, 12), sharex='col', sharey='row')

# Loop through the temperature shifts
for i, shift in enumerate(shifts):

    # Function for shifted temperature 
    def temp_shift(t):
        return temp_kanger(t, warming=shift)

    # Solve the heat equation for the current warming shift
    t, x, U = solve_heat(xstop=xstop, tstop=tstop+1, dx=dx, dt=dt, c2=c2, U_init=0, lowerbound=temp_shift, upperbound=5)

    # Plot the temperature field

    # Plot contours and get colorbar
    contour = ax3[i][0].pcolor(t/365, x, U, cmap='RdBu_r', vmin=-15, vmax=15)
    cbar = plt.colorbar(contour)

    # Additional labels
    cbar.set_label(r'Temperature ($^{\circ}C$)')
    ax3[i][0].set_xlabel('Time ($years$)')
    ax3[i][0].set_ylabel('Depth ($m$)')
    ax3[i][0].invert_yaxis()
    ax3[i][0].set_title(f'Ground Temperature vs Time with {shift} C Warming')

    # Create temperature vs depth / seasonal temperature profiles

    # Set indexing for the final year of results
    loc = int(-365/dt) # final 365 days of the result 
    # Extract the min values over the final year
    winter = U[:, loc:].min(axis=1)
    # Extract the min values over the final year
    summer = U[:, loc:].max(axis=1)
    
    # Plot profiles
    ax3[i][1].plot(winter, x, color='dodgerblue', label='Winter')
    ax3[i][1].plot(summer, x, color='tomato', linestyle='--', label='Summer')

    # Plot format
    ax3[i][1].set_xlabel('Temperature ($^{\circ}C$)')
    ax3[i][1].set_xlim(-8, 8)
    ax3[i][1].set_ylabel('Depth ($m$)')
    ax3[i][1].set_title(f'Seasonal Temperature Profiles at Steady State with {shift} C Warming')
    ax3[i][1].grid(True)
    ax3[i][1].legend()

    # Find the steady state year and layer depths

    # Define step, index of middle layer, and year when steady state is reached
    steps_yr = int(365/dt)  # time steps per year
    steady_yr = None        # empty 
    yr_threshold = 5        # number of years that need to be unchanging
    z_min, z_max = 10, 90   # top and bottom depths for the gradient to be calculated for
    depth_idx = np.where((x >= z_min) & (x <= z_max))[0] # get indices for the gradient

    # Loop through each year
    for yr in range(yr_threshold + 1, int(t[-1]/365)):
        # Get the temperature profile of the year steady state is reached
        start = (yr-1)*steps_yr
        end = yr*steps_yr
        prev_start = (yr-1-yr_threshold)*steps_yr
        prev_end = (yr-yr_threshold)*steps_yr

        # Compute gradient with depth for current and previous year
        grad_now = np.diff(U[depth_idx, start:end], axis=0)
        grad_prev = np.diff(U[depth_idx, prev_start:prev_end], axis=0)
        
        # Max change in gradient between consecutive years
        max_grad_change = np.max(np.abs(grad_now - grad_prev))
        
        # Get steady state year based on certain tolerance
        if max_grad_change < 0.01:
            steady_yr = yr
            break

    # Print results
    print(f'\nFor a global warming temperature shift of {shift} C:')
    if steady_yr is not None:
        print(f'Steady state is reached after ~ {steady_yr} years.')
    else:
        print(f'Steady state not reached within this timeframe.')

    if steady_yr is not None:
        # Get the temperature profile of the year steady state is reached
        steady_step = steady_yr * steps_yr
        start = steady_step - steps_yr
        end = steady_step
        T_yr = U[:, start:end]

        # Calculate summer as the max temps and winter as the min temps for the steady state year
        T_summer = T_yr.max(axis=1)
        T_winter = T_yr.min(axis=1)

        active_idx = np.where((T_summer > 0) & (T_winter < 0))[0] # Indices where ground melts and refreezes
        active_bot = x[active_idx[-1]] # Last depth ground melts and refreezes (bottom of active layer and top of permafrost)
        pfrst_bot = x[T_summer <= 0][-1] # Last depth where ground is frozen

        # Print the active and permafrost layer depths
        print(f'Active layer depth: {active_bot:.2f} m')
        print(f'Permafrost depth: {pfrst_bot:.2f} m')

# Add super title
fig3.suptitle('Kangerlussuaq Ground Temperature Profiles', weight='bold', fontsize=16)

# Adjust spacing
plt.subplots_adjust(hspace=0.25)  
