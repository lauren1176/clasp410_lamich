#!/usr/bin/ipython3

#!/usr/bin/env python3

'''
Tools and methods for completing Lab 3 which is the best lab.
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
plt.ion()
plt.close('all')

def solve_heat(xstop=1, tstop=0.2, dx=0.2, dt=0.02, c2=1):
    '''
    A function for solving the heat equation

    Parameters
    ----------
    Fill this out don't forget. :P
    c2 : float
        c^2, the square of the diffusion coefficient.

    Returns
    -------
    x, t : 1D Numpy arrays
        Space and time values, respectively.
    U : Numpy array
        The solution of the heat equation, size is nSpace x nTime
    '''

    dt_max = dt**2 / (2*c2)
    if dt > dt_max:
        raise ValueError(f'DANGER: dt={dt} > dt_max={dt_max}')

    # Get grid sizes (plus 1 to include 0 as well)
    N = int(tstop / dt) + 1
    M = int(xstop / dx) + 1

    # Set up space and time grid:
    t = np.linspace(0, tstop, N)
    x = np.linspace(0, xstop, M)

    # Create solution matrix; set initial conditions
    U = np.zeros([M, N])
    U[:, 0] = 4*x - 4*x**2

    # Get our "r" coeff:
    r = c2 * (dt/dx**2)

    # Solve our equation!
    for j in range(N-1):
        U[1:M-1, j+1] = (1-2*r) * U[1:M-1, j] + r*(U[2:M, j] + U[:M-2, j])

    # Return our pretty solution to the caller:
    return t, x, U


def plot_heatsolve(t, x, U, title=None, **kwargs):
    '''
    Plot the 2D solution for the `solve_heat` function.

    Extra kwargs handed to pcolor.

    Paramters
    ---------
    t, x : 1D Numpy arrays
        Space and time values, respectively.
    U : Numpy array
        The solution of the heat equation, size is nSpace x nTime
    title : str, default is None
        Set title of figure.

    Returns
    -------
    fig, ax : Matplotlib figure & axes objects
        The figure and axes of the plot.

    cbar : Matplotlib color bar object
        The color bar on the final plot
    '''

    # Check our kwargs for defaults:
    # Set default cmap to hot
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'hot'

    # Create and configure figure & axes:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Add contour to our axes:
    contour = ax.pcolor(t, x, U, **kwargs)
    cbar = plt.colorbar(contour)

    # Add labels to stuff!
    cbar.set_label(r'Temperature ($^{\circ}C$)')
    ax.set_xlabel('Time ($s$)')
    ax.set_ylabel('Position ($m$)')
    ax.set_title(title)

    fig.tight_layout()

    return fig, ax, cbar

t, x, U = solve_heat()
plot_heatsolve(t, x, U, title='Toy Problem: Hot Wire!')

# Solution to problem 10.3 from fink/matthews as a nested list:
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
# Convert to an array and transpose it to get correct ordering:
sol10p3 = np.array(sol10p3).transpose()

plot_heatsolve(t, x, sol10p3, title='Toy Problem: Reference Solution')

# Solver is first order in time, so error should grow or shrink with a time step of 1
# log(delta t) vs log(err) slope is 1 because first order in time
# log(delta x) vs log(err) slope is x because second order accurate in space