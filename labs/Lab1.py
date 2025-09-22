#!/usr/bin/ipython3
'''
This program explores a N-layer atmosphere model for Lab 1 and subparts.

To reproduce the values and plots in my report, please do this: tbd
'''

# Imports and style
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-colorblind')
plt.ion()

# Define constants 
sigma = 5.6703e-8 # W m^-2 K^-4
alpha = 0.33

# Create a function to get temperatures for a n-layer atmosphere
def solve_energy_balance(n_layers=5, S0=1350, ep=1, alp=alpha, debug=False):
    '''
    '''
    # Create array of coefficients, an N+1 x N+1 array
    A = np.zeros([n_layers+1, n_layers+1])
    b = np.zeros(n_layers+1)
    temps = np.zeros(n_layers+1)

    # Populate A array based on our model
    for i in range(n_layers + 1):
        for j in range(n_layers + 1):
            
            if i == j:
                A[i, j] = -2 + (j==0) # handle diagonals; [0, 0] is a special case

            else:
                '''
                ep**(i>0): 
                    Accounts for epsilon not being present in the first row.
                (1 - ep)**(np.abs(j-i) - 1): 
                    (1 - epsilon) is raised to the column - row - 1 power.
                '''
                A[i, j] = ( ep**(i>0) ) * ( (1 - ep)**(np.abs(j-i) - 1)  )

            if debug:
                print(f"A[{i},{j}] = {A[i,j]}")

    b[0] = -0.25 * S0 * (1 - alp)

    if debug:
        print(A)

    # Invert matrix
    Ainv = inv(A)

    # Calculate fluxes
    fluxes = np.matmul(Ainv, b)

    # Calculate temperatures
    for i, flux in enumerate(fluxes):
        if i == 0:
            temp = np.power(flux / sigma, 1/4)
        else:
            temp = np.power(flux / (ep * sigma), 1/4)
        temps[i] = temp

    return temps

print(solve_energy_balance(n_layers=6, ep=0.8, debug=False))


    

    





# discussion: why its not accurate, earth surface is increasing in temp, 
# how it could be improved and what would change with changes

# def calc_surf_temp(alp=alpha, sig=sigma):
#     '''
#     From albedo and solar irradiance, calculate Earth's predicted surface temperature.

#     Parameters
#     ----------
#     alp : float
#           albedo constant
#     sig : float
#           Stefan-Boltzmann constant; total energy radiated by a blackbody per unit surface area

#     Returns
#     -------
#     T_model : array of floats
#               The predicted temperature of Earth's surface
#     '''

#     # Solve T_E for the model
#     T_model = np.power( ((1-alp)*S0) / (2*sig), 1/4)

#     return T_model

# # Create year, corresponding solar flux, and corresponding temperature anomaly arrays
# year = np.array([1900, 1950, 2000])
# S0 = np.array([1365, 1366.5, 1368])
# T_anom = np.array([-0.4, 0, 0.4])

# # Call function to get modeled temperature
# T_model = calc_surf_temp()

# # Get the T_E observed value
# T_obs = T_model[1] + T_anom

# # Create our figure and plot
# fig, ax = plt.subplots(1, 1)
# ax.plot(year, T_model, label=f'Predicted Temperature Change')
# ax.plot(year, T_obs, label=f'Observed Temperature Change')

# # Plot format
# ax.legend()
# ax.set_xlabel('Year')
# ax.set_ylabel('Surface Temperature (K)')
# ax.set_title('Rate of Changes of Predicted vs Observed Surface Temperatures Comparison', weight='bold')
# fig.tight_layout()
# ax.grid()
