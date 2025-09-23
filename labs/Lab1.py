#!/usr/bin/ipython3
'''
This program explores a N-layer atmosphere model for Lab 1 and subparts.

To reproduce the values and plots in my report, please simply run the script. 
Change the values at the beginning of this script to adjust the conditions for
a typical model test case.
'''

# Imports and style
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-colorblind')
plt.ion()
plt.close('all')

# THESE VALUES CAN BE VARIED FOR DIFFERENT SCENARIOS:
n = 6           # Number of layers
epsilon = 0.8   # Emissivity of atmospheric layers
S = 1350        # Solar constant (W m^-2)
alpha = 0.33    # Planetary albedo
debug = False   # Debug condition

# Define Stefan-Boltzmann constant
sigma = 5.6703e-8 # W m^-2 K^-4

# Create a function to get temperatures for a n-layer atmosphere
def solve_energy_balance(n_layers=5, S0=1350, ep=1, alp=alpha, debug=False):
    '''
    Calculates the predicted temperatures of Earth's (or another planet's) surface and 
    atmospheric layers based on the number of atmospheric layers, solar constant, 
    atmospheric emissivity, and planetary albedo (reflectivity).

    Parameters
    ----------
    n_layers:   integer
                number of atmospheric layers for which to calculate temperatures
    S0:         integer
                solar constant; average amount of solar energy received at the top 
                of Earth's atmosphere (W m^-2)
    ep:         float
                atmospheric emissivity (or epsilon); the effectiveness of the 
                atmosphere in emitting thermal radiation
    alp:        float
                planetary albedo; how much light a surface reflects
    debug:      boolean condition
                TRUE prints debugging statements throughout the code's output
                FALSE does not

    Returns
    -------
    T_model : array of floats
              the predicted/modeled temperature of Earth's surface and atmospheric layers
    '''
    # Create array of coefficients
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

    # Solar flux reaches the Earth's surface
    b[0] = -0.25 * S0 * (1 - alp)

    if debug:
        print(f'Matrix A is {A}')

    # Invert matrix
    Ainv = inv(A)

    # Calculate fluxes
    fluxes = np.matmul(Ainv, b)

    if debug:
        print(f'The fluxes are {fluxes}')

    # Calculate temperatures
    for i, flux in enumerate(fluxes):
        if i == 0:
            temp = np.power(flux / sigma, 1/4)
        else:
            temp = np.power(flux / (ep * sigma), 1/4)
        temps[i] = temp

    return temps

# Print test 
print(f'For these conditions:\nN layers = {n}, ε = {epsilon}, S₀ = {S}, α = alpha, debug = {debug}')
print(f'The temperatures from the surface to layer {n} are\n{solve_energy_balance(n_layers=n, S0=S, ep=epsilon, alp=alpha, debug=debug)}\n')

#####################

### Experiment #1 ###

# Create a range of emissivities
emiss = np.arange(0.001, 1.001, 0.001)

# Define number of layers (for this experiment, 1)
n_layers_1 = 1

# Create an empty surface temperature array
T_s = np.zeros(emiss.size)

# Populate the temperature array
for i, e in enumerate(emiss):
    T_s[i] = solve_energy_balance(n_layers=n_layers_1, ep=e)[0] 
    # the [0] gets just the first element of the array, which is the sfc temperature

# Create the figure and plot surface temperatures vs emissivities
fig, ax = plt.subplots(1, 1)
ax.plot(emiss, T_s, label='Surface Temperature') 

# Get the emissivity at which T_s = 288K
idx = np.argwhere(np.diff(np.sign(T_s - 288)))[0] # this solution is from stackoverflow
x1 = emiss[idx[0]]
x2 = emiss[idx[0]+1]
y = 288
y1 = T_s[idx[0]]
y2 = T_s[idx[0]+1]
emiss_at_288 = x1 + ( ((y - y1) / (y2 - y1)) * (x2 - x1) ) # linear interpolation

# Print this emissivity
print(f'Emissivity of Earth\'s Atmosphere at 288 K: {emiss_at_288:.4f}')

# Add a line and point marking 288 K
ax.axhline(288, color='green', linestyle='--', label='T = 288 K')
ax.scatter(emiss_at_288, 288, color='lightseagreen', label=f'({emiss_at_288:.4f}, 288 K)', zorder=10)

# Plot formatting
ax.set_title('Surface Temperature vs. Emissivity (1-Layer Atmosphere)', weight='bold')
ax.set_xlabel('Atmosphere Emissivity')
ax.set_ylabel('Temperature (K)')
ax.set_xlim(0.001, 1)
ax.legend(loc='lower right')
ax.grid(True)
plt.tight_layout()

#####################

### Experiment #2 ###

# Create an array of layers
layers_arr = np.arange(0, 16, 1)

# Create an empty surface temperature array
T_s2 = np.zeros(layers_arr.size)

# Populate the temperature array
for i, l in enumerate(layers_arr):
    T_s2[i] = solve_energy_balance(n_layers=l, ep=0.255)[0] 
    # the [0] gets just the first element of the array, which is the sfc temperature

# Create the figure and plot surface temperatures vs n-layers
fig2, ax2 = plt.subplots(1, 1)
ax2.plot(layers_arr, T_s2, label='Surface Temperature') 

# Get index and layers at which T_s2 = 288K
idx2 = np.argwhere(np.diff(np.sign(T_s2 - 288)))[0] # this solution is from stackoverflow
x1_2 = layers_arr[idx2[0]]
x2_2 = layers_arr[idx2[0]+1]
y_2 = 288
y1_2 = T_s2[idx2[0]]
y2_2 = T_s2[idx2[0]+1]
layers_at_288 = x1_2 + ( ((y_2 - y1_2) / (y2_2 - y1_2)) * (x2_2 - x1_2) )
print(f'Number of Layers for Earth\'s Surface Temperature of 288 K: {layers_at_288:.2f}')

# Add a point marking 288 K
ax2.axhline(288, color='green', linestyle='--', label='T = 288 K')
ax2.scatter(layers_at_288, 288, color='lightseagreen', label=f'({layers_at_288:.2f}, 288 K)', zorder=10)

# Plot formatting
ax2.set_title('Surface Temperature vs. N-Layers ($\\epsilon$ = 0.255)', weight='bold')
ax2.set_xlabel('Number of Atmospheric Layers')
ax2.set_ylabel('Temperature (K)')
ax2.legend(loc='lower right')
ax2.grid(True)
plt.tight_layout()

# Call function to calculate layer temps for the number of layers for which Earth's sfc temp is 288 K
model_temps = solve_energy_balance(n_layers=(np.round(layers_at_288)).astype(int), ep=0.255)

# Define layer altitudes as ~10 km per layer
altitudes = np.arange(model_temps.size) * 10 # km

# Plot the function vs altitude with the number of layers for which Earth's sfc temp is 288 K
fig3, ax3 = plt.subplots(1, 1)
ax3.plot(model_temps, altitudes) 

# Add another axis for labeling layers
ax3b = ax3.twinx()
ax3b.set_ylim(ax3.get_ylim())
ax3b.set_yticks(altitudes[::2]) # get every other layer
ax3b.set_yticklabels([f'Layer {int(i)}' for i in (altitudes[::2]/10)])
ax3b.set_ylabel('Atmospheric Layers')

# Plot formatting
ax3.set_title('Layer Temperature vs. Altitude ($\\epsilon$ = 0.255)', weight='bold')
ax3.set_xlabel('Temperature (K)')
ax3.set_ylabel('Altitude (km)')
ax3.grid(True)
plt.tight_layout()

###################

### Venus Model ###

# Define constants
Ven_T_actual = 700  # K
Ven_S0 = 2600       # W m^-2
Ven_ep = 1
Ven_alp = 0.75

# Create an array of layers
layers_Ven = np.arange(1, 100, 1)

# Create an empty surface temperature array
Ven_T_s = np.zeros(layers_Ven.size)

# Populate the temperature array
for i, l in enumerate(layers_Ven):
    Ven_T_s[i] = solve_energy_balance(n_layers=l, S0=Ven_S0, ep=Ven_ep, alp=Ven_alp)[0] 
    # the [0] gets just the first element of the array, which is the sfc temperature

# Get index and layers at which Ven_T_s = Ven_T_actual
V_idx = np.argwhere(np.diff(np.sign(Ven_T_s - Ven_T_actual)))[0] 
Vx1 = layers_Ven[V_idx[0]]
Vx2 = layers_Ven[V_idx[0]+1]
Vy = Ven_T_actual
Vy1 = Ven_T_s[V_idx[0]]
Vy2 = Ven_T_s[V_idx[0]+1]
layers_at_Ven_T = Vx1 + ( ((Vy - Vy1) / (Vy2 - Vy1)) * (Vx2 - Vx1) )
print(f'Number of Layers for Venus\'s Surface Temperature of {Ven_T_actual} K: {layers_at_Ven_T:.2f}')

###############################

### Nuclear Winter Scenario ###

# Edit old function
def energy_balance_nuclear_winter(n_layers=5, S0=1350, ep=0.5, alp=alpha, debug=False):
    '''
    Calculates the predicted temperatures of Earth's (or another planet's) surface and 
    atmospheric layers based on the number of atmospheric layers, solar constant, 
    atmospheric emissivity, and planetary albedo (reflectivity).

    This current function edition makes it so the solar flux is completely absorbed 
    by the top layer of the atmosphere, representing a nuclear winter scenario.

    Parameters
    ----------
    n_layers:   integer
                number of atmospheric layers for which to calculate temperatures
    S0:         integer
                solar constant; average amount of solar energy received at the top 
                of Earth's atmosphere (W m^-2)
    ep:         float
                atmospheric emissivity (or epsilon); the effectiveness of the 
                atmosphere in emitting thermal radiation
    alp:        floatsds
                planetary albedo; how much light a surface reflects
    debug:      boolean condition
                TRUE prints debugging statements throughout the code's output
                FALSE does not

    Returns
    -------
    T_model : array of floats
              the predicted/modeled temperature of Earth's surface and atmospheric layers
              in a nuclear winter scenario
    '''
    # Create array of coefficients
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

    # Solar flux only reaches the top layer of the atmosphere
    b[-1] = -0.25 * S0 * (1 - alp)

    if debug:
        print(f'Matrix A is {A}')

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

# Define ash's albedo
ash_albedo = 0.5

# Calculate the layer temperatures for the nuclear winter scenario
nuclear_T = energy_balance_nuclear_winter(n_layers=5, ep=0.5, alp=ash_albedo)
print(f'Surface Temperature for Nuclear Winter: {nuclear_T[0]}')

# Define layer altitudes as ~10 km per layer
nuclear_altitudes = np.arange(nuclear_T.size) * 10 # km

# Plot the function vs altitude with the number of layers for which Earth's sfc temp is 288 K
fig4, ax4 = plt.subplots(1, 1)
ax4.plot(nuclear_T, nuclear_altitudes) 

# Add another axis for labeling layers
ax4b = ax4.twinx()
ax4b.set_ylim(ax3.get_ylim())
ax4b.set_yticks(nuclear_altitudes[::2]) # get every other layer
ax4b.set_yticklabels([f'Layer {int(i)}' for i in (nuclear_altitudes[::2]/10)])
ax4b.set_ylabel('Atmospheric Layers')

# Plot formatting
ax4.set_title('Layer Temperature vs. Altitude (Nuclear Winter Scenario)', weight='bold')
ax4.set_xlabel('Temperature (K)')
ax4.set_ylabel('Altitude (km)')
ax4.grid(True)
plt.tight_layout()