#!/usr/bin/ipython3

'''
This lab explores the snowball Earth hypothesis using a heat diffusion 
model of Earth's temperature at different latitude slices.

To reproduce the values and plots in my report, please simply run
the script. In the terminal, use the commands "ipython" and then "run Lab5.py". 
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numpy.random import rand
import matplotlib as mpl

plt.style.use('seaborn-v0_8-colorblind')
plt.ion()
plt.close('all')

# Define constants
radearth = 6357000.  # Earth's radius (m)
mxdlyr = 50.         # depth of mixed layer (m)
sigma = 5.67e-8      # Stefan-Boltzman constant (J m^-2 s^-1 K^-4)
C = 4.2e6            # Heat capacity of water (J m^-3 K^-1)
rho = 1020           # Density of sea water (kg/m^3)

def gen_grid(npoints=18):
    '''
    Create a evenly spaced latitudinal grid with 'npoints' cell centers.
    Grid will always run from zero to 180 as the edges of the grid. This
    means that the first grid point will be 'dLat/2' from 0 degrees and the
    last point will be '180 - dLat/2'.

    Parameters
    ----------
    npoints:    int, defaults to 18
                Number of grid points to create

    Returns
    -------
    dLat:   float
            Grid spacing in latitude (degrees)
    lats:   numpy array
            Locations of all grid cell centers
    '''

    dlat = 180 / npoints  # Latitude spacing
    lats = np.linspace(dlat/2., 180-dlat/2., npoints)  # Latitude cell centers

    return dlat, lats


def temp_warm(lats_in):
    '''
    Create a temperature profile for modern day "warm" earth.

    Parameters
    ----------
    lats_in:    Numpy array
                Array of latitudes in degrees where temperature is required

    Returns
    -------
    temp:   Numpy array
            Temperature in Celcius
    '''

    # Set initial temperature curve
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
                        23, 19, 14, 9, 1, -11, -19, -47])
    # Get base grid
    npoints = T_warm.size
    dlat, lats = gen_grid(npoints)

    coeffs = np.polyfit(lats, T_warm, 2)

    # Return fitting
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp

def insolation(S0, lats):
    '''
    Given a solar constant ('S0'), calculate average annual, longitude-averaged
    insolation values as a function of latitude.
    Insolation is returned at position 'lats' in units of W/m^2.

    Parameters
    ----------
    S0:     float
            Solar constant (1370 for typical Earth conditions)
    lats:   Numpy array
            Latitudes to output insolation. Following the grid standards set in
            the diffusion program, polar angle is defined from the south pole.
            In other words, 0 is the south pole, 180 the north.

    Returns
    -------
    insolation: numpy array
                Insolation returned over the input latitudes.
    '''

    # Constants
    max_tilt = 23.5 # tilt of earth in degrees

    # Create an array to hold insolation
    insolation = np.zeros(lats.size)

    # Daily rotation of earth reduces solar constant by distributing the sun
    # energy all along a zonal band
    dlong = 0.01  # Use 1/100 of a degree in summing over latitudes
    angle = np.cos(np.pi/180. * np.arange(0, 360, dlong))
    angle[angle < 0] = 0
    total_solar = S0 * angle.sum()
    S0_avg = total_solar / (360/dlong)

    # Accumulate normalized insolation through a year.
    # Start with the spin axis tilt for every day in 1 year
    tilt = [max_tilt * np.cos(2.0*np.pi*day/365) for day in range(365)]

    # Apply to each latitude zone
    for i, lat in enumerate(lats):
        # Get solar zenith; do not let it go past 180. Convert to latitude.
        zen = lat - 90. + tilt
        zen[zen > 90] = 90
        # Use zenith angle to calculate insolation as function of latitude.
        insolation[i] = S0_avg * np.sum(np.cos(np.pi/180. * zen)) / 365.

    # Average over entire year; multiply by S0 amplitude
    insolation = S0_avg * insolation / 365

    return insolation

def snowball_earth(nlat=18, tfinal=10000, dt=1.0, lam=100., emiss=1.0,
                   init_cond=temp_warm, apply_spherecorr=False, albice=.6,
                   albgnd=.3, apply_insol=False, solar=1370, gamma=1):
    '''
    Solve the snowball Earth problem.

    Parameters
    ----------
    nlat:   int, defaults to 18
            Number of latitude cells.
    tfinal: int or float, defaults to 10,000
            Time length of simulation in years.
    dt:     int or float, defaults to 1.0
            Size of timestep in years.
    lam:    float, defaults to 100
            Set ocean diffusivity
    emiss:  float, defaults to 1.0
            Set emissivity of Earth/ground.
    init_cond:  function, float, or array
                Set the initial condition of the simulation. If a function is given,
                it must take latitudes as input and return temperature as a function
                of lat. Otherwise, the given values are used as-is.
    apply_spherecorr:   bool, defaults to False
                        Apply spherical correction term
    apply_insol:    bool, defaults to False
                    Apply insolation term.
    solar:          float, defaults to 1370
                    Set level of solar forcing in W/m2
    albice, albgnd: float, defaults to .6 and .3
                    Set albedo values for ice and ground.
    gamma:  float, defaults to 1
            Set soolar multiplier factor.

    Returns
    --------
    lats:   Numpy array
            Latitudes representing cell centers in degrees; 
            0 is south pole 180 is north.
    Temp:   Numpy array
            Temperature as a function of latitude.
    '''

    # Set up grid
    dlat, lats = gen_grid(nlat)
    # Y-spacing for cells in physical units
    dy = np.pi * radearth / nlat

    # Create our first derivative operator
    B = np.zeros((nlat, nlat))
    B[np.arange(nlat-1)+1, np.arange(nlat-1)] = -1
    B[np.arange(nlat-1), np.arange(nlat-1)+1] = 1
    B[0, :] = B[-1, :] = 0

    # Create area array
    Axz = np.pi * ((radearth+50.0)**2 - radearth**2) * np.sin(np.pi/180.*lats)
    # Get derivative of Area
    dAxz = np.matmul(B, Axz)

    # Set number of time steps
    nsteps = int(tfinal / dt)

    # Set timestep to seconds
    dt = dt * 365 * 24 * 3600

    # Create insolation
    insol = gamma * insolation(solar, lats)

    # Create temp array & set our initial condition
    Temp = np.zeros(nlat)
    if callable(init_cond):
        Temp = init_cond(lats)
    else:
        Temp += init_cond

    # Create our K matrix
    K = np.zeros((nlat, nlat))
    K[np.arange(nlat), np.arange(nlat)] = -2
    K[np.arange(nlat-1)+1, np.arange(nlat-1)] = 1
    K[np.arange(nlat-1), np.arange(nlat-1)+1] = 1

    # Boundary conditions
    K[0, 1], K[-1, -2] = 2, 2

    # Units
    K *= 1/dy**2

    # Create L matrix
    Linv = np.linalg.inv(np.eye(nlat) - dt * lam * K)

    # Set initial albedo
    albedo = np.zeros(nlat)
    loc_ice = Temp <= -10 # Sea water freezes at ten below
    albedo[loc_ice] = albice
    albedo[~loc_ice] = albgnd

    # SOLVE!
    for istep in range(nsteps):
        # Update Albedo
        loc_ice = Temp <= -10 # Sea water freezes at ten below
        albedo[loc_ice] = albice
        albedo[~loc_ice] = albgnd

        # Create spherical coordinates correction term
        if apply_spherecorr:
            sphercorr = (lam*dt) / (4*Axz*dy**2) * np.matmul(B, Temp) * dAxz
        else:
            sphercorr = 0

        # Apply radiative/insolation term
        if apply_insol:
            radiative = (1-albedo)*insol - emiss*sigma*(Temp+273)**4
            Temp += dt * radiative / (rho*C*mxdlyr)

        # Advance solution
        Temp = np.matmul(Linv, Temp + sphercorr)

    return lats, Temp
    

### Question 1 ###

# Get warm Earth initial condition
dlat, lats = gen_grid()
temp_init = temp_warm(lats)

# Get solution after 10K years for each combination of terms
lats, temp_diff = snowball_earth()
lats, temp_sphe = snowball_earth(apply_spherecorr=True)
lats, temp_alls = snowball_earth(apply_spherecorr=True, apply_insol=True, albice=.3)

# Create a plot
fig, ax = plt.subplots(1, 1)
ax.plot(lats-90, temp_init, label='Initial Condition')
ax.plot(lats-90, temp_diff, label='Diffusion Only')
ax.plot(lats-90, temp_sphe, label='Diffusion + Spherical Corr.')
ax.plot(lats-90, temp_alls, label='Diffusion + Spherical Corr. + Radiative')

# Plot format
ax.set_title('Warm Earth Model after 10,000 Years')
ax.set_ylabel(r'Temp ($^{\circ}C$)')
ax.set_xlabel('Latitude')
ax.legend(loc='best')


### Question 2 ###

# Define ranges
lambda_range = np.arange(0, 160, 10)
emiss_range = np.arange(0, 1.1, 0.1)

# Get the grid & target warm earth curve
dlat, lats = gen_grid()
warm_earth_curve = temp_warm(lats)


# Create a plot
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

# Plot target warm earth curve
ax.plot(lats - 90, warm_earth_curve, color='black', linestyle='--', label='Warm Earth', zorder=2)

# Define line colors
n_lines = lambda_range.size
cmap =  mpl.colormaps['winter']
norm = mpl.colors.Normalize(vmin=np.min(lambda_range), vmax=np.max(lambda_range))
colors = cmap(np.linspace(0, 1, n_lines))

# Loop through all possilbe lambdas
for i, l in enumerate(lambda_range):

    # Get solution after 10K years for warm earth for the current lambda value
    lats, varied_lam_curve = snowball_earth(nlat=18, tfinal=10000, dt=1.0, lam=l, emiss=1.0,
                                            init_cond=temp_warm, apply_spherecorr=True, albice=.3,
                                            albgnd=.3, apply_insol=True, solar=1370)

    # Plot
    ax.plot(lats - 90, varied_lam_curve, color=cmap(norm(l)), label=f'$\lambda$ = {l}', zorder=1)

# Plot format
ax.set_title('Warm Earth Model after 10,000 Years with Variable $\lambda$')
ax.set_ylabel(r'Temp ($^{\circ}C$)')
ax.set_xlabel('Latitude')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()


# Create a plot
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))

# Plot target warm earth curve
ax2.plot(lats - 90, warm_earth_curve, color='black', linestyle='--', label='Warm Earth', zorder=2)

# Define line colors
n_lines = emiss_range.size
cmap =  mpl.colormaps['winter']
norm = mpl.colors.Normalize(vmin=np.min(emiss_range), vmax=np.max(emiss_range))
colors = cmap(np.linspace(0, 1, n_lines))

# Loop through all possilbe emissivities
for i, e in enumerate(emiss_range):

    # Get solution after 10K years for warm earth for the current emissivity value
    lats, varied_emiss_curve = snowball_earth(nlat=18, tfinal=10000, dt=1.0, lam=100, emiss=e,
                                            init_cond=temp_warm, apply_spherecorr=True, albice=.3,
                                            albgnd=.3, apply_insol=True, solar=1370)

    # Plot
    ax2.plot(lats - 90, varied_emiss_curve, color=cmap(norm(e)), label=f'$\epsilon$ = {e:.1f}', zorder=1)

# Plot format
ax2.set_title('Warm Earth Model after 10,000 Years with Variable $\epsilon$')
ax2.set_ylabel(r'Temp ($^{\circ}C$)')
ax2.set_xlabel('Latitude')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()


# Create a plot
fig3, ax3 = plt.subplots(1, 1)
# Plot target warm earth curve
ax3.plot(lats - 90, warm_earth_curve, color='black', linestyle='--', label='Warm Earth', zorder=2)

# Get best solution after 10K years for warm earth
lats, best_curve = snowball_earth(nlat=18, tfinal=10000, dt=1.0, lam=38, emiss=0.74,
                                    init_cond=temp_warm, apply_spherecorr=True, albice=.6,
                                    albgnd=.3, apply_insol=True, solar=1370)

# Plot the curve
ax3.plot(lats - 90, best_curve, zorder=1, label='Best-Fit $\lambda$ and $\epsilon$')

# Plot format
ax3.set_title('Warm Earth Model after 10,000 Years with $\epsilon$ = 0.74 and $\lambda$ = 38')
ax3.set_ylabel(r'Temp ($^{\circ}C$)')
ax3.set_xlabel('Latitude')
ax3.legend(loc='best')
plt.tight_layout()


### Question 3 ###

def temp_hot(lats_in):
    '''
    Create a temperature profile for hot earth (60 C at all locations).

    Parameters
    ----------
    lats_in:    Numpy array
                Array of latitudes in degrees where temperature is required

    Returns
    -------
    temp:   Numpy array
            Temperature in Celcius
    '''

    T_hot = 60 * np.ones_like(lats_in)

    return T_hot

def temp_cold(lats_in):
    '''
    Create a temperature profile for cold earth (-60 C at all locations).

    Parameters
    ----------
    lats_in:    Numpy array
                Array of latitudes in degrees where temperature is required

    Returns
    -------
    temp:   Numpy array
            Temperature in Celcius
    '''

    T_cold = -60 * np.ones_like(lats_in)

    return T_cold

# Create a plot
fig4, ax4 = plt.subplots(1, 3, figsize=(12,4), sharey=True)

# Get solution after 10K years for hot Earth
lats, hot_curve = snowball_earth(nlat=18, tfinal=10000, dt=1.0, lam=38, emiss=0.74,
                                    init_cond=temp_hot, apply_spherecorr=True, albice=.6,
                                    albgnd=.3, apply_insol=True, solar=1370) 
# Plot hot earth equilibrium
ax4[0].plot(lats - 90, hot_curve, label='($\\alpha_g = 0.3$, $\\alpha_i = 0.6$)', color='firebrick')


# Get solution after 10K years for cold Earth
lats, cold_curve = snowball_earth(nlat=18, tfinal=10000, dt=1.0, lam=38, emiss=0.74,
                                    init_cond=temp_cold, apply_spherecorr=True, albice=.6,
                                    albgnd=.3, apply_insol=True, solar=1370) 
# Plot cold earth equilibrium
ax4[1].plot(lats - 90, cold_curve, label='($\\alpha_g = 0.3$, $\\alpha_i = 0.6$)', color='darkblue')


# Get solution after 10K years for flash-frozen Earth
lats, curve_freeze = snowball_earth(nlat=18, tfinal=10000, dt=1.0, lam=38, emiss=0.74,
                                    init_cond=temp_warm, apply_spherecorr=True, albice=.6,
                                    albgnd=.6, apply_insol=True, solar=1370) 

# Plot flash-frozen equilibrium
ax4[2].plot(lats - 90, curve_freeze, label='($\\alpha_g = 0.6$, $\\alpha_i = 0.6$)', color='lightblue')

# Plot format
fig4.suptitle('Models after 10,000 Years with $\epsilon$ = 0.74 and $\lambda$ = 38')
ax4[0].set_title('Initial Hot Earth Equilibrium')
ax4[0].set_ylabel(r'Temp ($^{\circ}C$)')
ax4[0].set_xlabel('Latitude')
ax4[0].legend()
ax4[1].set_title('Initial Cold Earth Equilibrium')
ax4[1].set_xlabel('Latitude')
ax4[1].legend()
ax4[2].set_title('Flash-Frozen Warm Earth Equilibrium')
ax4[2].set_xlabel('Latitude')
ax4[2].legend()
plt.tight_layout()


### Question 4 ###

# Make array of gammas
gam_up = np.arange(0.4, 1.4, 0.05)
gam_down = gam_up[-2::-1]
gammas = np.concatenate((gam_up, gam_down))

# Empty array for mean temperatures
mean_temps = np.zeros_like(gammas)

# Initialize inital condition
temp_initial = temp_cold(lats)

# Iterate through gamma array
for i, g in enumerate(gammas):
    # Get solution
    lats, eq_temp = snowball_earth(nlat=18, tfinal=10000, dt=1.0, lam=38, emiss=0.74,
                                        init_cond=temp_initial, apply_spherecorr=True, albice=.6,
                                        albgnd=.3, apply_insol=True, solar=1370, gamma=g) 

    # Store averages
    mean_temps[i] = np.mean(eq_temp)
    
    # Use the last solution as the new initial condition
    temp_initial = eq_temp

# Get arrays in correct order
n_up = len(gam_up)
temps_up = mean_temps[0:n_up]
gamma_up = gammas[0:n_up]
temps_down = mean_temps[n_up:]
gamma_down = gammas[n_up:]

# Plot mean temperatures 
fig5, ax5 = plt.subplots(1, 1)
ax5.plot(gamma_up, temps_up, marker='o', color='red', label='Increasing $\gamma$')
ax5.plot(gamma_down, temps_down, marker='o', color='green', label='Decreasing $\gamma$')

# Plot format
ax5.set_xlabel(r'Solar Multiplier $\gamma$')
ax5.set_ylabel('Global Mean Temperature ($^{\circ}C$)')
ax5.set_title('Global Mean Temperature vs Solar Multiplier $\gamma$')
ax5.legend()