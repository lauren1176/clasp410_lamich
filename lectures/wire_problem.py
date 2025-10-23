#!/usr/bin/ipython3

'''
Tools and methods for completing Lab 3.
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-colorblind')
plt.ion()
plt.close('all')

# Solution to problem 10.3 from fink/matthews
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

# Convert to an array and transpose it to get correct ordering
sol10p3 = np.array(sol10p3).transpose()

# Define a function to solve the heat equation
def solve_heat(xstop=1, tstop=0.2, dx=0.2, dt=0.02, c2=1, lowerbound=0, upperbound=0):
    '''
    A function for solving the heat equation

    Parameters
    ----------
    c2: float
        c^2, the square of the diffusion coefficient

    Returns
    -------
    x, t:   1D Numpy arrays
            Space and time values, respectively

    U:      Numpy array
            The solution of the heat equation, size is nSpace x nTime
    '''

    # Check stability criteria
    dt_max = dx**2 / (2*c2) / dt
    if dt > dt_max:
        print('Values used in this simulation:')
        print(f'\tdt = {dt}\n\tc2 = {c2}\n\tdx = {dx}')
        raise ValueError(f'DANGER: dt={dt} > dt_max={dt_max}')

    # Get grid sizes (plus 1 to include 0 as well)
    N = int(tstop / dt) + 1
    M = int(xstop / dx) + 1

    # Set up space and time grid
    t = np.linspace(0, tstop, N)
    x = np.linspace(0, xstop, M)

    # Create the solution matrix
    # Set initial conditions
    U = np.zeros([M, N])
    U[:, 0] = 4*x - 4*x**2

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

def plot_heatsolve(t, x, U, title=None, **kwargs):
    '''
    Plot the 2D solution for the `solve_heat` function.

    Extra kwargs are handed to pcolor.

    Paramters
    ---------
    t, x:   1D Numpy arrays
            Space and time values, respectively

    U:      Numpy array
            The solution of the heat equation, size is nSpace x nTime

    title:  str, default is None
            Title for the figure

    Returns
    -------
    fig, ax:    Matplotlib figure & axes objects
                The figure and axes of the plot.

    cbar:       Matplotlib color bar object
                The color bar on the final plot
    '''

    # Check our kwargs for defaults
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'hot' # Set default cmap to hot

    # Create and configure figure & axes
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Add contour to our axes
    contour = ax.pcolor(t, x, U, **kwargs)
    cbar = plt.colorbar(contour)

    # Additional labels
    cbar.set_label(r'Temperature ($^{\circ}C$)')
    ax.set_xlabel('Time ($s$)')
    ax.set_ylabel('Position ($m$)')
    ax.set_title(title)

    # Improve layout
    fig.tight_layout()

    return fig, ax, cbar

# t, x, U = solve_heat()
# plot_heatsolve(t, x, U, title='Toy Problem: Hot Wire!')
# plot_heatsolve(t, x, sol10p3, title='Toy Problem: Reference Solution')

# Dirichlet vs Neumann for HW 
t_D, x_D, U_D = solve_heat(dx=0.02, dt=0.0002)
t_N, x_N, U_N = solve_heat(dx=0.02, dt=0.0002, upperbound=None, lowerbound=None)
# Create and configure figure & axes
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
contour_D = ax[0].pcolor(t_D, x_D, U_D, cmap='hot')
cbar_D = plt.colorbar(contour_D)
cbar_D.set_label(r'Temperature ($^{\circ}C$)')
contour_N = ax[1].pcolor(t_N, x_N, U_N, cmap='hot')
cbar_N = plt.colorbar(contour_N)
cbar_N.set_label(r'Temperature ($^{\circ}C$)')
ax[0].set_title('Dirichlet Boundary Conditions')
ax[0].set_xlabel('Time ($s$)')
ax[0].set_ylabel('Position ($m$)')
ax[1].set_title('Neumann Boundary Conditions')
ax[1].set_xlabel('Time ($s$)')
ax[1].set_ylabel('Position ($m$)')
fig.suptitle('Dirichlet vs Neumann Boundary Conditions', weight='bold')
fig.tight_layout()

'''
Additional Notes:
- Solver is first order in time, so error should grow or shrink with a time step of 1
- log(delta t) vs log(err) slope is 1 because first order in time
- log(delta x) vs log(err) slope is x because second order accurate in space

Dirichlet boundary conditions in lab, where the upperbound is temperature that changes 
with time. As we solve, the temperature on the surface changes as a function of time.
U[-1, j+1] = upperbound_function
'''