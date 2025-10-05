#!/usr/bin/ipython3
'''
This program explores competition models of population growth
for Lab 2 using Ordinary Differential Equations, solved via
Euler and RK8 methods.

To reproduce the values and plots in my report, please simply run the script.
In the terminal, use the commands "ipython" and then "run Lab2.py". 

To specifically get Figure 1 in the report, change the dt_comp = 0.01 to 
dt_comp = 1 and change the dt_pp = 0.001 to dt_pp = 0.05 at the top of the 
### Question #1 ### section.
'''

# Imports and style
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use('seaborn-v0_8-colorblind')
plt.ion()
# Option to close all open plots when running
# plt.close('all')

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

    a, b, c, d: float (defaults = 1, 2, 1, 3)
                The value of the Lotka-Volterra coefficients.

    Returns
    -------
    dN1dt, dN2dt: floats
                  The time derivatives of `N1` and `N2`.
    '''

    # N is a two-element list such that N1=N[0] and N2=N[1]
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

    a, b, c, d: float (defaults = 1, 2, 1, 3)
                The value of the Lotka-Volterra coefficients.

    Returns
    -------
    dN1dt, dN2dt: floats
                  The time derivatives of `N1` and `N2`.
    '''

    # N is a two-element list such that N1=N[0] and N2=N[1]
    dN1dt = a*N[0] - b*N[0]*N[1]
    dN2dt = -c*N[1] + d*N[0]*N[1]

    return dN1dt, dN2dt

# Define a function to solve one of the LV equations via a Euler method
# This solution is based on the lecture example
def solve_euler(dfx, dt=10, N1_i=0.5, N2_i=0.5, t_f=100.0, a=1, b=2, c=1, d=3):
    '''
    This function solves the Lotka-Volterra competition and predator/prey  
    ordinary differential equations using Euler's method.  
    
    Parameters
    ----------
    dfx:    function
            Function representing the time derivative of our ODE system.
            It should take arguments (t, N, a, b, c, d) where N is [N1, N2],
            and return (dN1dt, dN2dt).

    dt:     float (default = 10)
            Time step in years.

    N1_i, N2_i: float (defaults = 0.5)
                Initial conditions for N1 and N2 for the ODE.

    t_f:    float (default = 100)
            The final time for our solver in years.

    a, b, c, d: float (defaults = 1, 2, 1, 3)
                The value of the Lotka-Volterra coefficients.

    Returns
    -------
    time:   numpy array
            Time in seconds over the entire solution.

    N1, N2: numpy arrays
            The solutions as functions of time.
    '''

    # Configure the problem, creating a time array and initializing the functions
    time = np.arange(0, t_f, dt)
    N1 = np.zeros(time.size)
    N2 = np.zeros(time.size)
    N1[0] = N1_i
    N2[0] = N2_i

    # Solve the Euler approximations for N1 and N2
    for i in range(time.size - 1):
        dN1dt, dN2dt = dfx(time[i], [N1[i], N2[i]], a=a, b=b, c=c, d=d)
        N1[i+1] = N1[i] + dt * dN1dt 
        N2[i+1] = N2[i] + dt * dN2dt

    return time, N1, N2

# Define a function to solve a one of the LV equations via DOP853
# This solution is based on the lab2 pdf example
def solve_rk8(dfx, dt=10, N1_i=0.5, N2_i=0.5, t_f=100.0, a=1, b=2, c=1, d=3):
    '''
    Solve the Lotka-Volterra competition and predator/prey equations using
    Scipy's ODE class and the adaptive step 8th order solver.

    Parameters
    ----------
    dfx:    function
            A python function that takes `time`, [`N1`, `N2`] as inputs and
            returns the time derivative of N1 and N2.

    N1_i, N2_i: float
                Initial conditions for `N1` and `N2`, ranging from (0,1]

    dt: float (default = 10)
        Largest timestep allowed in years.

    t_f: float (default = 100)
         Integrate until this value is reached, in years.

    a, b, c, d: float (default = 1, 2, 1, 3)
                Lotka-Volterra coefficient values.

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

###################

### Question #1 ###

# Define time step
dt_comp = 0.01      # year
dt_rk8_comp = 0.1   # year
dt_pp = 0.001       # year
dt_rk8_pp = 0.1     # year

# Run competition model
t_e_comp, N1_e_comp, N2_e_comp = solve_euler(dNdt_comp, dt=dt_comp, N1_i=0.3, N2_i=0.6, t_f=100, a=1, b=2, c=1, d=3)
t_rk_comp, N1_rk_comp, N2_rk_comp = solve_rk8(dNdt_comp, dt=dt_rk8_comp, N1_i=0.3, N2_i=0.6, t_f=100, a=1, b=2, c=1, d=3)

# Run predator-prey model
t_e_pp, N1_e_pp, N2_e_pp = solve_euler(dNdt_pp, dt=dt_pp, N1_i=0.3, N2_i=0.6, t_f=100, a=1, b=2, c=1, d=3)
t_rk_pp, N1_rk_pp, N2_rk_pp = solve_rk8(dNdt_pp, dt=dt_rk8_pp, N1_i=0.3, N2_i=0.6, t_f=100, a=1, b=2, c=1, d=3)

# Plot competition model
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(t_e_comp, N1_e_comp, c='orange', linestyle='-', label='Euler N1')
ax[0].plot(t_e_comp, N2_e_comp, c='purple', linestyle='-', label='Euler N2')
ax[0].plot(t_rk_comp, N1_rk_comp, c='orange', linestyle='--', label='RK8 N1')
ax[0].plot(t_rk_comp, N2_rk_comp, c='purple', linestyle='--', label='RK8 N2')

# Plot predator-prey model
ax[1].plot(t_e_pp, N1_e_pp, c='orange', linestyle='-', label='Euler N1')
ax[1].plot(t_e_pp, N2_e_pp, c='purple', linestyle='-', label='Euler N2')
ax[1].plot(t_rk_pp, N1_rk_pp, c='orange', linestyle='--', label='RK8 N1')
ax[1].plot(t_rk_pp, N2_rk_pp, c='purple', linestyle='--', label='RK8 N2')

# Plot format
ax[0].set_title('Lotka-Volterra Competition Model')
ax[0].set_xlabel('Time (years)')
ax[0].set_ylabel('Population (carrying capacity normalized to 1)')
ax[0].legend()
ax[1].set_title('Lotka-Volterra Predator-Prey Model')
ax[1].set_xlabel('Time (years)')
ax[1].set_ylabel('Population')
ax[1].set_ylim(0, 1)
ax[1].legend()

###################

### Question #2 ###

# Part 1: Ranges of Initial Conditions #

# Ranges of inital conditions
N1_arr = np.arange(0, 1.1, 0.1)
N2_arr = 1 - N1_arr

# Define line colors
n_lines = N1_arr.size
cmap_N1 =  mpl.colormaps['winter']
norm_N1 = mpl.colors.Normalize(vmin=np.min(N1_arr), vmax=np.max(N1_arr))
colors_N1 = cmap_N1(np.linspace(0, 1, n_lines))
cmap_N2 =  mpl.colormaps['autumn']
norm_N2 = mpl.colors.Normalize(vmin=np.min(N2_arr), vmax=np.max(N2_arr))
colors_N2 = cmap_N2(np.linspace(0, 1, n_lines))

# Create figure and axis
fig2, ax2 = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

# Iterate through N1 and N2 arrays
for i, [N1_init, N2_init] in enumerate(zip(N1_arr, N2_arr)):

    # Run competition model for current [N1, N2]
    t_e_comp, N1_e_comp, N2_e_comp = solve_euler(dNdt_comp, dt=dt_comp, N1_i=N1_init, N2_i=N2_init, t_f=15, a=1, b=2, c=1, d=3)
    t_rk_comp, N1_rk_comp, N2_rk_comp = solve_rk8(dNdt_comp, dt=dt_rk8_comp, N1_i=N1_init, N2_i=N2_init, t_f=15, a=1, b=2, c=1, d=3)

    # Plot competition models (Euler row 0, RK8 row 1)
    ax2[0, 0].plot(t_e_comp, N1_e_comp, color=colors_N1[i])
    ax2[0, 1].plot(t_e_comp, N2_e_comp, color=colors_N2[i])
    ax2[1, 0].plot(t_rk_comp, N1_rk_comp, color=colors_N1[i])
    ax2[1, 1].plot(t_rk_comp, N2_rk_comp, color=colors_N2[i])

# Large title
plt.suptitle('Comparing Initial Conditions for Competition Models', weight='bold', fontsize=16)

# Subtitle
fig2.text(0.5, 0.945, 'with constant coefficients a=1, b=2, c=1, d=3', ha='center', va='top', fontsize=12, style='italic')

# Individual titles
ax2[0, 0].set_title(f'Euler N1 (dt = {dt_comp})')
ax2[0, 1].set_title(f'Euler N2 (dt = {dt_comp})')
ax2[1, 0].set_title('RK8 N1')
ax2[1, 1].set_title('RK8 N2')

# Labels
ax2[0, 0].set_xlabel('Time (years)')
ax2[0, 1].set_xlabel('Time (years)')
ax2[1, 0].set_xlabel('Time (years)')
ax2[1, 1].set_xlabel('Time (years)')
ax2[0, 0].set_ylabel('Population (carrying capacity normalized to 1)')
ax2[1, 0].set_ylabel('Population (carrying capacity normalized to 1)')

# Adjust layout for room for colorbars 
fig2.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.88, wspace=0.5, hspace=0.3)

# N1 colorbar on the left
cbarax_N1 = fig2.add_axes([0.44, 0.17, 0.02, 0.65])  # recttuple (left, bottom, width, height)
fig2.colorbar(mpl.cm.ScalarMappable(norm=norm_N1, cmap=cmap_N1), cax=cbarax_N1, orientation="vertical", label="N1 Initial Condition")

# N2 colorbar on the right
cbarax_N2 = fig2.add_axes([0.92, 0.17, 0.02, 0.65]) # recttuple (left, bottom, width, height)
fig2.colorbar(mpl.cm.ScalarMappable(norm=norm_N2, cmap=cmap_N2), cax=cbarax_N2, orientation="vertical", label="N2 Initial Condition (N2 = 1 - N1)")


# Part 2: Varying Coefficients #
'''
a and c are population growth due to the species itself.
b and d are population reduction becuase of the other species.

The equations are:
dN1dt = a*N[0]*(1-N[0]) - b*N[0]*N[1]
dN2dt = c*N[1]*(1-N[1]) - d*N[1]*N[0]

We can define competition strengths:
b = N2's limiting strength on N1 
d = N1's limiting strength on N2
'''

# Define a range of coefficients 
coef_range = np.arange(1, 5.5, 0.5) # weak competition to very strong competition

# Fixed coefficient values 
a = 1
b = 2
c = 1
d = 3
coef_vary = 'd' 

# Define line colors
cmap_N1 = mpl.colormaps['winter']
cmap_N2 = mpl.colormaps['autumn']
norm_range = mpl.colors.Normalize(vmin=coef_range.min(), vmax=coef_range.max())
colors_N1 = cmap_N1(np.linspace(0, 1, len(coef_range)))
colors_N2 = cmap_N2(np.linspace(0, 1, len(coef_range)))

# Create figure and axis
fig3, ax3 = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

# Iterate through competition strengths
for i, r in enumerate(coef_range):

    # Get coefficients based on desired varied coefficient
    if coef_vary == 'a':
        a_now = r
        b_now, c_now, d_now = b, c, d
    elif coef_vary == 'b':
        b_now = r
        a_now, c_now, d_now = a, c, d
    elif coef_vary == 'c':
        c_now = r
        a_now, b_now, d_now = a, b, d
    elif coef_vary == 'd':
        d_now = r
        a_now, b_now, c_now = a, b, c

    # Run competition models
    t_e_comp, N1_e_comp, N2_e_comp = solve_euler(dNdt_comp, dt=dt_comp, N1_i=0.4, N2_i=0.6, t_f=15, a=a_now, b=b_now, c=c_now, d=d_now)
    t_rk_comp, N1_rk_comp, N2_rk_comp = solve_rk8(dNdt_comp, dt=dt_rk8_comp, N1_i=0.4, N2_i=0.6, t_f=15, a=a_now, b=b_now, c=c_now, d=d_now)

    # Plot competition models (Euler row 0, RK8 row 1)
    ax3[0, 0].plot(t_e_comp, N1_e_comp, color=colors_N1[i])
    ax3[0, 1].plot(t_e_comp, N2_e_comp, color=colors_N2[i])
    ax3[1, 0].plot(t_rk_comp, N1_rk_comp, color=colors_N1[i])
    ax3[1, 1].plot(t_rk_comp, N2_rk_comp, color=colors_N2[i])

# Large title
plt.suptitle(f'Comparing Varying Coefficient "{coef_vary}" for Competition Models', weight='bold', fontsize=16)

# Subtitle
fig3.text(0.5, 0.945, 'with constant conditions N1 = 0.4 and N2 = 0.6', ha='center', va='top', fontsize=12, style='italic')

# Individual titles
ax3[0, 0].set_title(f'Euler N1 (dt = {dt_comp})')
ax3[0, 1].set_title(f'Euler N2 (dt = {dt_comp})')
ax3[1, 0].set_title('RK8 N1')
ax3[1, 1].set_title('RK8 N2')

# Labels
ax3[0, 0].set_xlabel('Time (years)')
ax3[0, 1].set_xlabel('Time (years)')
ax3[1, 0].set_xlabel('Time (years)')
ax3[1, 1].set_xlabel('Time (years)')
ax3[0, 0].set_ylabel('Population (carrying capacity normalized to 1)')
ax3[1, 0].set_ylabel('Population (carrying capacity normalized to 1)')

# Adjust layout for room for colorbars 
fig3.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.88, wspace=0.5, hspace=0.3)

# N1 colorbar on the left
cbarax_N1 = fig3.add_axes([0.44, 0.17, 0.02, 0.65])  # recttuple (left, bottom, width, height)
fig3.colorbar(mpl.cm.ScalarMappable(norm=norm_range, cmap=cmap_N1), cax=cbarax_N1, orientation='vertical', label=f'{coef_vary} Coefficient Value')

# N2 colorbar on the right
cbarax_N2 = fig3.add_axes([0.92, 0.17, 0.02, 0.65]) # recttuple (left, bottom, width, height)
fig3.colorbar(mpl.cm.ScalarMappable(norm=norm_range, cmap=cmap_N2), cax=cbarax_N2, orientation='vertical', label=f'{coef_vary} Coefficient Value')

###################

### Question #3 ###

# Part 1: Ranges of Initial Conditions #

# Ranges of inital conditions
N1_arr = np.arange(0.08, 1, 0.08)
N2_arr = 1 - N1_arr

# Define line colors
n_lines = N1_arr.size
cmap_N1 =  mpl.colormaps['winter']
norm_N1 = mpl.colors.Normalize(vmin=np.min(N1_arr), vmax=np.max(N1_arr))
colors_N1 = cmap_N1(np.linspace(0, 1, n_lines))
cmap_N2 =  mpl.colormaps['autumn']
norm_N2 = mpl.colors.Normalize(vmin=np.min(N2_arr), vmax=np.max(N2_arr))
colors_N2 = cmap_N2(np.linspace(0, 1, n_lines))

# Create figure and axis
fig4, ax4 = plt.subplots(1, 3, figsize=(15, 6))

# Iterate through N1 and N2 arrays
for i, [N1_init, N2_init] in enumerate(zip(N1_arr, N2_arr)):

    # Run predator-prey model for current [N1, N2]
    t_rk_pp, N1_rk_pp, N2_rk_pp = solve_rk8(dNdt_pp, dt=dt_rk8_pp, N1_i=N1_init, N2_i=N2_init, t_f=20, a=1, b=2, c=1, d=3)

    # Plot predator-prey model
    ax4[1].plot(t_rk_pp, N1_rk_pp, color=colors_N1[i])
    ax4[2].plot(t_rk_pp, N2_rk_pp, color=colors_N2[i])
    ax4[0].plot(N1_rk_pp, N2_rk_pp, color=colors_N1[i])

# Large title
plt.suptitle('Comparing Initial Conditions for Predator-Prey Model', weight='bold', fontsize=16)

# Subtitle
fig4.text(0.5, 0.935, 'with constant coefficients a=1, b=2, c=1, d=3', ha='center', va='top', fontsize=14, style='italic')

# Individual titles
ax4[0].set_title('RK8 N1 vs N2', fontsize=14)
ax4[1].set_title('RK8 N1', fontsize=14)
ax4[2].set_title('RK8 N2', fontsize=14)

# Labels
ax4[0].set_xlabel('Prey (N1) Population', fontsize=12)
ax4[0].set_ylabel('Predator (N2) Population', fontsize=12)
ax4[1].set_xlabel('Time (years)', fontsize=12)
ax4[1].set_ylabel('Population', fontsize=12)
ax4[2].set_xlabel('Time (years)', fontsize=12)
ax4[2].set_ylabel('Population', fontsize=12)
ax4[0].tick_params(axis='x', labelsize=10) 
ax4[0].tick_params(axis='y', labelsize=10) 

# Adjust layout for room for colorbars 
fig4.subplots_adjust(left=0.1, right=0.9, bottom=0.28, top=0.84, wspace=0.25)

# N1 colorbar on the left
cbarax_N1 = fig4.add_axes([0.16, 0.09, 0.4, 0.05])  # recttuple (left, bottom, width, height)
cbarN1 = fig4.colorbar(mpl.cm.ScalarMappable(norm=norm_N1, cmap=cmap_N1), cax=cbarax_N1, orientation="horizontal")
cbarN1.ax.tick_params(labelsize=10)
cbarN1.set_label(label='N1 Initial Condition', fontsize=10)

# N2 colorbar on the right
cbarax_N2 = fig4.add_axes([0.662, 0.09, 0.25, 0.05]) # recttuple (left, bottom, width, height)
cbarN2 = fig4.colorbar(mpl.cm.ScalarMappable(norm=norm_N2, cmap=cmap_N2), cax=cbarax_N2, orientation="horizontal")
cbarN2.ax.tick_params(labelsize=10)
cbarN2.set_label(label='N2 Initial Condition (N2 = 1 - N1)', fontsize=10)


# Part 2: Varying Coefficients #

# Define a range of coefficients 
coef_range = np.arange(1, 5.5, 0.5)

# Fixed coefficient values 
a = 1
b = 2
c = 1
d = 3
coef_vary = 'd' 

# Define line colors
cmap_N1 = mpl.colormaps['winter']
cmap_N2 = mpl.colormaps['autumn']
norm_range = mpl.colors.Normalize(vmin=coef_range.min(), vmax=coef_range.max())
colors_N1 = cmap_N1(np.linspace(0, 1, len(coef_range)))
colors_N2 = cmap_N2(np.linspace(0, 1, len(coef_range)))

# Create figure and axis
fig5, ax5 = plt.subplots(1, 3, figsize=(15, 6))

# Iterate through coefficient range
for i, r in enumerate(coef_range):

    # Get coefficients based on desired varied coefficient
    if coef_vary == 'a':
        a_now = r
        b_now, c_now, d_now = b, c, d
    elif coef_vary == 'b':
        b_now = r
        a_now, c_now, d_now = a, c, d
    elif coef_vary == 'c':
        c_now = r
        a_now, b_now, d_now = a, b, d
    elif coef_vary == 'd':
        d_now = r
        a_now, b_now, c_now = a, b, c

    # Run predator-prey models
    t_rk_pp, N1_rk_pp, N2_rk_pp = solve_rk8(dNdt_pp, dt=dt_rk8_pp, N1_i=0.4, N2_i=0.6, t_f=15, a=a_now, b=b_now, c=c_now, d=d_now)

    # Plot predator-prey model
    ax5[1].plot(t_rk_pp, N1_rk_pp, color=colors_N1[i])
    ax5[2].plot(t_rk_pp, N2_rk_pp, color=colors_N2[i])
    ax5[0].plot(N1_rk_pp, N2_rk_pp, color=colors_N1[i])

# Large title
plt.suptitle(f'Comparing Varying Coefficient "{coef_vary}" for Predator-Prey Model', weight='bold', fontsize=16)

# Subtitle
fig5.text(0.5, 0.935, 'with constant conditions N1 = 0.4 and N2 = 0.6', ha='center', va='top', fontsize=12, style='italic')

# Individual titles
ax5[0].set_title('RK8 N1 vs N2', fontsize=14)
ax5[1].set_title('RK8 N1', fontsize=14)
ax5[2].set_title('RK8 N2', fontsize=14)

# Labels
ax5[0].set_xlabel('Prey (N1) Population', fontsize=12)
ax5[0].set_ylabel('Predator (N2) Population', fontsize=12)
ax5[1].set_xlabel('Time (years)', fontsize=12)
ax5[1].set_ylabel('Population', fontsize=12)
ax5[2].set_xlabel('Time (years)', fontsize=12)
ax5[2].set_ylabel('Population', fontsize=12)
ax5[0].tick_params(axis='x', labelsize=10) 
ax5[0].tick_params(axis='y', labelsize=10) 

# Adjust layout for room for colorbars 
fig5.subplots_adjust(left=0.1, right=0.9, bottom=0.28, top=0.84, wspace=0.25)

# N1 colorbar on the left
cbarax_N1 = fig5.add_axes([0.16, 0.09, 0.4, 0.05])  # recttuple (left, bottom, width, height)
cbarN1 = fig5.colorbar(mpl.cm.ScalarMappable(norm=norm_range, cmap=cmap_N1), cax=cbarax_N1, orientation="horizontal")
cbarN1.ax.tick_params(labelsize=10)
cbarN1.set_label(label=f'{coef_vary} Coefficient Value', fontsize=10)

# N2 colorbar on the right
cbarax_N2 = fig5.add_axes([0.662, 0.09, 0.25, 0.05]) # recttuple (left, bottom, width, height)
cbarN2 = fig5.colorbar(mpl.cm.ScalarMappable(norm=norm_range, cmap=cmap_N2), cax=cbarax_N2, orientation="horizontal")
cbarN2.ax.tick_params(labelsize=10)
cbarN2.set_label(label=f'{coef_vary} Coefficient Value', fontsize=10)

