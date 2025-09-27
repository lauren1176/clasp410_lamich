#!/usr/bin/ipython3
'''
This program explores a competetion models of population growth
for Lab 2 using Ordinary Differential Equations.

To reproduce the values and plots in my report, please simply run the script.
'''

# Imports and style
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-colorblind')
plt.ion()
plt.close('all')

# Define function that, given the current solutions (N1 and N2), 
# returns their time derivatives for the competition system.
# This is based on the lab2 pdf
def dNdt_comp(t, N, a=1, b=2, c=1, d=3):
    '''
    This function calculates the Lotka-Volterra competition equations for
    two species. Given normalized populations, `N1` and `N2`, as well as the
    four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the caller.
    
    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this function.

    Parameters
    ----------
    t:  float
        The current time (not used here).

    N:  two-element list
        The current value of N1 and N2 as a list (e.g., [N1, N2]).

    a, b, c, d: float, defaults=1, 2, 1, 3
                The value of the Lotka-Volterra coefficients.

    Returns
    -------
    dN1dt, dN2dt: floats
                  The time derivatives of `N1` and `N2`.
    '''

    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1dt = a*N[0]*(1-N[0]) - b*N[0]*N[1]
    dN2dt = c*N[1]*(1-N[1]) - d*N[1]*N[0]

    return dN1dt, dN2dt

# Define function that, given the current solutions (N1 and N2), 
# returns their time derivatives based on the predator-prey system.
def dNdt_pp(t, N, a=1, b=2, c=1, d=3):
    '''
    This function calculates the Lotka-Volterra predator-prey equations for
    two species. Given normalized populations, `N1` and `N2`, as well as the
    four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the caller.
    
    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this function.

    Parameters
    ----------
    t:  float
        The current time (not used here).

    N:  two-element list
        The current value of N1 and N2 as a list (e.g., [N1, N2]).

    a, b, c, d: float, defaults=1, 2, 1, 3
                The value of the Lotka-Volterra coefficients.

    Returns
    -------
    dN1dt, dN2dt: floats
                  The time derivatives of `N1` and `N2`.
    '''

    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1dt = a*N[0] - b*N[0]*N[1]
    dN2dt = -c*N[1] + d*N[0]*N[1]

    return dN1dt, dN2dt

# Define a function to solve one of the LV equations via a Euler method
# This solution is based on the lecture example
def solve_euler(dfx, dt=10, N1_i=0.5, N2_i=0.5, t_f=100.0, a=1, b=2, c=1, d=3):
    '''
    Solve ordinary differential equations using Euler's method.  
    Extra kwargs are passed to the dfx function.

    Parameters
    ----------
    dfx:    function
            Function representing the time derivative of our ODE system.
            It should take arguments (t, N, **kwargs), where N is [N1, N2],
            and return (dN1dt, dN2dt)

    N1_i, N2_i: float
                Initial conditions for N1 and N2 for the ODE.

    t_f:    float (default 300)
            The final time for our solver in seconds.

    dt:     float (default 10)
            Time step in seconds.

    Returns
    -------
    t:  numpy array
        time in seconds over the entire solution

    N1, N2: numpy arrays
            the solutions as functions of time
    '''

    # Configure the problem, creating a time array and initializing the functions
    time = np.arange(0, t_f, dt)
    N1 = np.zeros(time.size)
    N2 = np.zeros(time.size)
    N1[0] = N1_i
    N2[0] = N2_i

    # Solve the Euler approximations for N1 and N2
    for i in range(1, time.size):
        dN1dt, dN2dt = dfx(time[i-1], [N1[i-1], N2[i-1]], a=a, b=b, c=c, d=d)
        N1[i] = N1[i-1] + dt * dN1dt 
        N2[i] = N2[i-1] + dt * dN2dt

    return time, N1, N2

# Define a function to solve a one of the LV equations via DOP853
# This solution is based on the lab2 pdf example
# a=1, b=2, c=1, d=3
def solve_rk8(dfx, dt=10, N1_i=0.5, N2_i=0.5, t_f=100.0, a=1, b=2, c=1, d=3):
    '''
    Solve the Lotka-Volterra competition and predator/prey equations using
    Scipy's ODE class and the adaptive step 8th order solver.

    Parameters
    ----------
    func:   function
            A python function that takes `time`, [`N1`, `N2`] as inputs and
            returns the time derivative of N1 and N2.

    N1_init, N2_init: float
                      Initial conditions for `N1` and `N2`, ranging from (0,1]

    dT: float (default = 10)
        Largest timestep allowed in years.

    t_final: float (default = 100)
             Integrate until this value is reached, in years.

    a, b, c, d: float (default = 1, 2, 1, 3)
                Lotka-Volterra coefficient values

    Returns
    -------
    time:   Numpy array
            Time elapsed in years.

    N1, N2: Numpy arrays
            Normalized population density solutions.
    '''
    from scipy.integrate import solve_ivp

    # Configure the initial value problem solver
    result = solve_ivp(dfx, [0, t_f], [N1_i, N2_i], args=(a, b, c, d), method='DOP853', max_step=dt)

    # Perform the integration
    time, N1, N2 = result.t, result.y[0, :], result.y[1, :]

    # Return values to caller
    return time, N1, N2

# dNdt_comp(t, N, a=1, b=2, c=1, d=3)
# solve_euler(dfx, dt=10, N1_i=0.5, N2_i=0.5, t_f=100.0, **kwargs)
# solve_rk8(dfx, dt=10, N1_i=0.5, N2_i=0.5, t_f=100.0, **kwargs)
# dN1dt, dN2dt = dfx(time[i], [N1[i-1], N2[i-1]], a=a, b=b, c=c, d=d)

# dt_comp = 1*365*24*60*60
# dt_pp = 0.05*365*24*60*60
dt_comp = 1  # year
dt_pp = 0.05 # year

dN1_comp, dN2_comp = dNdt_comp(1, [0.3, 0.6], a=1, b=2, c=1, d=3)
dN1_pp, dN2_pp = dNdt_pp(1, [0.3, 0.6], a=1, b=2, c=1, d=3)

# Run competition model
t_e_comp, N1_e_comp, N2_e_comp = solve_euler(dNdt_comp, dt=dt_comp, N1_i=0.3, N2_i=0.3, t_f=20, a=1, b=2, c=1, d=3)
t_rk_comp, N1_rk_comp, N2_rk_comp = solve_rk8(dNdt_comp, dt=dt_comp, N1_i=0.5, N2_i=0.5, t_f=20, a=1, b=2, c=1, d=3)

# Plot
fig, ax = plt.subplots(1, 1)
ax.plot(t_e_comp, N1_e_comp, c='blue', linestyle='-', label='Euler N1')
ax.plot(t_e_comp, N2_e_comp, c='blue', linestyle='--', label='Euler N2')
ax.plot(t_rk_comp, N1_rk_comp, c='red', linestyle='-', label='RK8 N1')
ax.plot(t_rk_comp, N2_rk_comp, c='red', linestyle='--', label='RK8 N2')
ax.set_title('Competition Model')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Population')
ax.legend()
