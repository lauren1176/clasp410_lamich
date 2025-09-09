#!/usr/bin/ipython3

# Imports and style
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-colorblind')

plt.ion()

def solve_temp(t, T_init=90., T_env=20.0, k=1/300.):
    '''cd 
    For a given scalar or array of times, `t`, return the 
    analytic solution for Newton's law of cooling.

    Parameters
    ----------
    t : Numpy array
        Array of times, in seconds, for which solution will be provided.

    Other Parameters
    ----------------
    k       :   float
                Heat transfer coefficient, defaults to 1/300. s^-1
    T_env   :   float
                Ambient environment temperature, defaults to 20°C.
    T_init  :   float
                Initial temperature of cooling object/mass, defaults to °C

    Returns
    -------
    t_coffee :  numpy array
                An array of temperatures corresponding to `t`.
    '''

    t_coffee = T_env + (T_init - T_env) * np.exp(-k*t)

    return t_coffee

def time_to_temp(T_final, T_init=90., T_env=20.0, k=1/300.):
    '''
    Given an initial temperature, `T_init`, an ambient temperature, `T_env`,
    and a cooling rate, return the time required to reach a target temperature,
    `T_target`.

    Parameters
    ----------
    T_final :   float
                Target temperature, in °C
    k       :   float
                Heat transfer coefficient, defaults to 1/300. s^-1
    T_env   :   float
                Ambient environment temperature, defaults to 20°C.
    T_init  :   float
                Initial temperature of cooling object/mass, defaults to °C

    Returns
    -------
    t : float
        The time, in seconds, to cool to target T_final
    '''

    t = (-1/k) * np.log((T_final - T_env)/(T_init - T_env))

    return t

def verify_code():
    '''
    Verify that our implementation is correct.
    '''

    t_real = 60. * 10.76
    k = np.log(95/110) / -120
    t_code = time_to_temp(120, T_init=180, T_env=70, k=k)

    print("Target solution is: ", t_real)
    print("Numerical solution is: ", t_code)
    print("Difference is: ", t_real - t_code)


# Solve the problem using the functions declared above

# First, do it quantitatively
t_1 = time_to_temp(65)              # Add cream at T = 65 to get to 60 degreespl
t_2 = time_to_temp(60, T_init=85)   # Add cream immediately
t_c = time_to_temp(60)              # Control case, no cream

print(f"Time to drinkable coffee:\n\tControl case = {t_c:.2f} s\n\tLate cream = {t_1:.2f} s\n\tImmediate cream = {t_2:.2f} s")

# Create time series of temperatures 
t = np.arange(0, 600., 0.5)
temp1 = solve_temp(t) # also the same as control
temp2 = solve_temp(t, T_init=85.)

# Create our figure and plot
fig, ax = plt.subplots(1, 1)
ax.plot(t, temp1, label=f'Add Cream Later (T = {t_1:.2f}s)')
ax.plot(t, temp2, label=f'Add Cream Now (T = {t_2:.2f}s)')

# Plot format
ax.legend()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Temperature (°C)')
ax.set_title('When to add Cream for Quick Coffee Cooling', weight='bold')
ax.grid()

