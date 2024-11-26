#!/usr/bin/env python3

'''
The following code runs and creates plots for Lab 05: Snowball Earth
Shown below is a list of function and a description of what they do.

gen_grid(): Generates a grid from 0 to 180 latitude

temp_warm(): Create a temperature profile for modern day "warm" earth.

insolation():Given a solar constant (`S0`), calculate average annual, 
longitude-averaged insolation values as a function of latitude.

snowball_earth(): Perform snowball earth simulation.

test_snowball(): Answers Question 1 by reproducing example plot in 
lecture/handout

question_2a(): Produces first plot to help answer question 2

question_2b(): Produces second plot to help answer question 2

question_3(): Produces plot for question 3

question_4(): Produces plot for question 4
'''

import numpy as np
import matplotlib.pyplot as plt

# Set MPL style sheet
plt.style.use('default')

# Some constants:
radearth = 6357000.  # Earth radius in meters.
mxdlyr = 50.         # depth of mixed layer (m)
sigma = 5.67e-8      # Steffan-Boltzman constant
C = 4.2e6            # Heat capacity of water
rho = 1020           # Density of sea-water (kg/m^3)
albedo_gnd =0.3
albedo_ice = 0.6


def gen_grid(nbins=18):
    '''
    Generate a grid from 0 to 180 lat (where 0 is south pole, 180 is north)
    where each returned point represents the cell center.

    Parameters
    ----------
    nbins : int, defaults to 18
        Set the number of latitude bins.

    Returns
    -------
    dlat : float
        Grid spacing in degrees.
    lats : Numpy array
        Array of cell center latitudes.
    '''

    dlat = 180 / nbins  # Latitude spacing.
    lats = np.arange(0, 180, dlat) + dlat/2.

    # Alternative way to obtain grid:
    # lats = np.linspace(dlat/2., 180-dlat/2, nbins)

    return dlat, lats


def temp_warm(lats_in):
    '''
    Create a temperature profile for modern day "warm" earth.

    Parameters
    ----------
    lats_in : Numpy array
        Array of latitudes in degrees where temperature is required

    Returns
    -------
    temp : Numpy array
        Temperature in Celcius.
    '''

    # Get base grid:
    dlat, lats = gen_grid()

    # Set initial temperature curve:
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
                       23, 19, 14, 9, 1, -11, -19, -47])
    coeffs = np.polyfit(lats, T_warm, 2)

    # Now, return fitting:
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp


def insolation(S0, lats):
    '''
    Given a solar constant (`S0`), calculate average annual, longitude-averaged
    insolation values as a function of latitude.
    Insolation is returned at position `lats` in units of W/m^2.

    Parameters
    ----------
    S0 : float
        Solar constant (1370 for typical Earth conditions.)
    lats : Numpy array
        Latitudes to output insolation. Following the grid standards set in
        the diffusion program, polar angle is defined from the south pole.
        In other words, 0 is the south pole, 180 the north.

    Returns
    -------
    insolation : numpy array
        Insolation returned over the input latitudes.
    '''

    # Constants:
    max_tilt = 23.5   # tilt of earth in degrees

    # Create an array to hold insolation:
    insolation = np.zeros(lats.size)

    #  Daily rotation of earth reduces solar constant by distributing the sun
    #  energy all along a zonal band
    dlong = 0.01  # Use 1/100 of a degree in summing over latitudes
    angle = np.cos(np.pi/180. * np.arange(0, 360, dlong))
    angle[angle < 0] = 0
    total_solar = S0 * angle.sum()
    S0_avg = total_solar / (360/dlong)

    # Accumulate normalized insolation through a year.
    # Start with the spin axis tilt for every day in 1 year:
    tilt = [max_tilt * np.cos(2.0*np.pi*day/365) for day in range(365)]

    # Apply to each latitude zone:
    for i, lat in enumerate(lats):
        # Get solar zenith; do not let it go past 180. Convert to latitude.
        zen = lat - 90. + tilt
        zen[zen > 90] = 90
        # Use zenith angle to calculate insolation as function of latitude.
        insolation[i] = S0_avg * np.sum(np.cos(np.pi/180. * zen)) / 365.

    # Average over entire year; multiply by S0 amplitude:
    insolation = S0_avg * insolation / 365

    return insolation


def snowball_earth(nbins=18, dt=1., tstop=10000, lam=100., spherecorr=True,
                   debug=False, albedo=0.3, emiss=1, S0=1370, D_albedo=False, 
                   initial_condition = "warm", solar_mult=False, gamma = 0.4, Temp_initial=None):
    '''
    Perform snowball earth simulation.

    Parameters
    ----------
    nbins : int, defaults to 18
        Number of latitude bins.
    dt : float, defaults to 1
        Timestep in units of years
    tstop : float, defaults to 10,000
        Stop time in years
    lam : float, defaults to 100
        Diffusion coefficient of ocean in m^2/s
    spherecorr : bool, defaults to True
        Use the spherical coordinate correction term. This should always be
        true except for testing purposes.
    debug : bool, defaults to False
        Turn  on or off debug print statements.
    albedo : float, defaults to 0.3
        Set the Earth's albedo.
    emiss : float, defaults to 1.0
        Set ground emissivity. Set to zero to turn off radiative cooling.
    S0 : float, defaults to 1370
        Set incoming solar forcing constant. Change to zero to turn off
        insolation.
    D_albedo : bool, defualts to Falso
        Applies dynamic albedo if set to true
    initial_condition : str, defaults to "warm"
        Applies the intial condition for different temperature cases.
        Values: warm, hot, and cold
    solar_mult : bool, defaults to False
        Applies solar multiplier if set to true
    gamma : float, defaults to 0.4
        Solar Multiplier value
    Temp_initial : Any, defaults to None
        Helpful to innumerate in question 4
    Returns
    -------
    lats : Numpy array
        Latitude grid in degrees where 0 is the south pole.
    Temp : Numpy array
        Final temperature as a function of latitude.
    '''
    # Get time step in seconds
    dt_sec = 365 * 24 * 3600 * dt  # Years to seconds.

    # Generate grid
    dlat, lats = gen_grid(nbins)

    # Get grid spacing in meters.
    dy = radearth * np.pi * dlat / 180.

    # Generate insolation
    if solar_mult:
        # Solar insolation with solar multipliar
        insol = gamma * insolation(S0, lats)
    else: 
        # Solar insolation withour solar multipliar
        insol = insolation(S0, lats)

    # Create initial condition:
    if Temp_initial is not None:
        Temp = Temp_initial.copy()  # Use the provided temperature array
    else:
        # Intial Conditions for warm Earth
        if initial_condition == "warm":
            Temp = temp_warm(lats)
        # Initial Conditions for a hot Earth at 60 Degrees Celsius
        elif initial_condition == "hot":
            Temp = np.full_like(lats, 60.0)  
        # Initial Conditions for a cold Earth at -60 Degrees Celsius
        elif initial_condition == "cold":
            Temp = np.full_like(lats, -60.0)
        else:
            raise ValueError("Invalid initial_condition")


    if debug:
        print('Initial temp = ', Temp)

    # Get number of timesteps
    nstep = int(tstop / dt)

    # Debug for problem initialization
    if debug:
        print("DEBUG MODE!")
        print(f"Function called for nbins={nbins}, dt={dt}, tstop={tstop}")
        print(f"This results in nstep={nstep} time step")
        print(f"dlat={dlat} (deg); dy = {dy} (m)")
        print("Resulting Lat Grid:")
        print(lats)

    # Build A matrix
    if debug:
        print('Building A matrix...')
    A = np.identity(nbins) * -2  # Set diagonal elements to -2
    A[np.arange(nbins-1), np.arange(nbins-1)+1] = 1  # Set off-diag elements
    A[np.arange(nbins-1)+1, np.arange(nbins-1)] = 1  # Set off-diag elements
    # Set boundary conditions
    A[0, 1], A[-1, -2] = 2, 2

    # Build "B" matrix for applying spherical correction
    B = np.zeros((nbins, nbins))
    B[np.arange(nbins-1), np.arange(nbins-1)+1] = 1  # Set off-diag elements
    B[np.arange(nbins-1)+1, np.arange(nbins-1)] = -1  # Set off-diag elements
    # Set boundary conditions
    B[0, :], B[-1, :] = 0, 0

    # Set the surface area of the "side" of each latitude ring at bin center.
    Axz = np.pi * ((radearth+50.0)**2 - radearth**2) * np.sin(np.pi/180.*lats)
    dAxz = np.matmul(B, Axz) / (Axz * 4 * dy**2)

    if debug:
        print('A = ', A)
    # Set units of A derp
    A /= dy**2
    # Get our "L" matrix:
    L = np.identity(nbins) - dt_sec * lam * A
    L_inv = np.linalg.inv(L)
    
    # Create Dynamic Albedo if statement
    if D_albedo:
        # Convert albedo to an array if it's not already one
        if isinstance(albedo, float):
            # Create an array with the same shape as lats 
            albedo = np.full_like(lats, albedo)           
        # Update albedo based on conditions:
        loc_ice = Temp <= -10
        albedo[loc_ice] = albedo_ice
        albedo[~loc_ice] = albedo_gnd   

    if debug:
        print('Time integrating...')
    for i in range(nstep):
        # Add spherical correction term:
        if spherecorr:
            Temp += dt_sec * lam * dAxz * np.matmul(B, Temp)

        # Apply insolation and radiative losses:
        # print('T before insolation:', Temp)
        radiative = (1-albedo) * insol - emiss*sigma*(Temp+273.15)**4
        # print('\t Rad term = ', dt_sec * radiative / (rho*C*mxdlyr))
        Temp += dt_sec * radiative / (rho*C*mxdlyr)
        # print('\t T after rad:', Temp)

        Temp = np.matmul(L_inv, Temp)
 

    return lats, Temp


def test_snowball(tstop=10000):
    '''
    Reproduce example plot in lecture/handout.

    Using our DEFAULT values (grid size, diffusion, etc.) and a warm-Earth
    initial condition, plot:
        - Initial condition
        - Plot simple diffusion only
        - Plot simple diffusion + spherical correction
        - Plot simple diff + sphere corr + insolation
    '''
    nbins = 18

    # Generate very simple grid
    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Create initial condition:
    initial = temp_warm(lats)

    # Get simple diffusion solution:
    lats, t_diff = snowball_earth(tstop=tstop, spherecorr=False, S0=0, emiss=0)

    # Get diffusion + spherical correction:
    lats, t_sphe = snowball_earth(tstop=tstop, S0=0, emiss=0)

    # Get diffusion + sphercorr + radiative terms:
    lats, t_rad = snowball_earth(tstop=tstop)

    # Create figure and plot!
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(lats, initial, color ='blue',label='Warm Earth Init. Cond.')
    ax.plot(lats, t_diff,  label='Basic Diffusion')
    ax.plot(lats, t_sphe, label='Diffusion + Sphere. Corr.')
    ax.plot(lats, t_rad, label='Diffusion + Sphere. Corr. + Radiative')
    # Label Axes
    ax.set_xlabel('Latitude (0=South Pole)')
    ax.set_ylabel('Temperature ($^{\circ} C$)')
    # Create grid and legend
    ax.grid()
    ax.legend(loc='best')
    # Show plot
    plt.show()
def question_2a(tstop = 10000):
    '''
    Produces first plot to help answer question 2
    '''
    nbins =18
    
    # Generate very simple grid
    # Generate grid:
    dlat, lats = gen_grid(nbins)
    
    # Get diffusion + sphercorr + radiative terms:
    lats, t_rad = snowball_earth(tstop=tstop)  

    # Change diffusivity from 0 to 150 
    lats, t_rad0 = snowball_earth(tstop=tstop, lam=0)
    lats, t_rad50 = snowball_earth(tstop=tstop, lam = 50)
    lats, t_rad150 = snowball_earth(tstop=tstop, lam = 150)

    # Change emissivity from 0 to 1
    lats, t_rad00 = snowball_earth(tstop=tstop, emiss= 0.)
    lats, t_rad_quart = snowball_earth(tstop=tstop, emiss= 0.25)
    lats, t_rad_half = snowball_earth(tstop=tstop, emiss = 0.5)  
    lats, t_rad_3quart = snowball_earth(tstop=tstop, emiss = 0.75)   

    # Create figure and plot!
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12, 6))
    

    ax1.plot(lats, t_rad0, label="Diffusion = 0 $\mathrm{m}^2/\mathrm{s}$")
    ax1.plot(lats, t_rad50, label="Diffusion = 50 $\mathrm{m}^2/\mathrm{s}$")
    ax1.plot(lats, t_rad, label='Diffusion = 100 $\mathrm{m}^2/\mathrm{s}$')    
    ax1.plot(lats, t_rad150, label="Diffusion = 150 $\mathrm{m}^2/\mathrm{s}$")
    ax1.set_xlim(0,181)
    ax1.set_title("Differences in Diffusion along Snowball Earth Curve")
    ax1.set_xlabel("Latitude ($^\circ$)")
    ax1.set_ylabel("Temperature ($^\circ$C)")
    ax1.legend(loc ='best')

    
    ax2.plot(lats, t_rad00, label='Emissivity = 0.0')
    ax2.plot(lats, t_rad_quart, label="Emissivity = 0.25")
    ax2.plot(lats, t_rad_half, label='Emissivity = 0.5')
    ax2.plot(lats, t_rad_3quart, label='Emissivity = 0.75')
    ax2.plot(lats, t_rad, label='Emissivity = 1.0')
    ax2.set_xlim(0,181)
    ax2.set_title("Differences in Emissivity along Snowball Earth Curve")
    ax2.set_xlabel("Latitude ($^\circ$)")
    ax2.set_ylabel("Temperature ($^\circ$C)")
    ax2.legend(loc ='best')

    plt.show()



def question_2b(tstop=10000):
    '''
    Produces second plot to help answer question 2
    '''
    nbins =18
    
    # Generate very simple grid
    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Create initial condition:
    initial = temp_warm(lats)
    
    # Get diffusion + sphercorr + radiative terms:
    lats, t_rad = snowball_earth(tstop=tstop) 

    # Find Values that match Warm Earth Curve
    # Get diffusion + sphercorr + radiative terms:
    lats, t_rad1 = snowball_earth(tstop=tstop, lam = 50, emiss = 0.5)    
    lats, t_rad2 = snowball_earth(tstop=tstop, lam = 80, emiss = 0.75)
    lats, t_rad3 = snowball_earth(tstop=tstop, lam = 70, emiss = 0.712)   
    # Create figure and plot!
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(lats, initial, color = 'red', label='Warm Earth Init. Cond.')
    ax.plot(lats,t_rad3, color = 'blue', label = 'Lam = 74 & emiss = 0.712')
    # Label axes and title
    ax.set_title("Snowball Earth Reproduced Warm-Earth Equilibrium")
    ax.set_xlabel("Latitude ($^\circ$)")
    ax.set_ylabel("Temperature ($^\circ$C)")
    # Create grid and legend
    ax.grid()
    ax.legend(loc='best')
    # Show Plot
    plt.show()

def question_3(tstop=10000):
    '''
    Produces plot for question 3.
    '''
    nbins =18
    # Generate very simple grid
    # Generate grid:
    dlat, lats = gen_grid(nbins)  
    
    # Find temperature curves with different initial conditions
    lats, t_rad = snowball_earth(tstop=tstop, lam = 74, emiss = 0.712, 
                                 D_albedo=True, initial_condition="hot")
    lats, t_rad1 = snowball_earth(tstop=tstop, lam = 74, emiss = 0.712, 
                                 D_albedo=True, initial_condition="cold")
    lats, t_rad2 = snowball_earth(tstop=tstop, lam = 74, emiss = 0.712, 
                                 albedo = 0.6, initial_condition="warm")    
    
    # Create figure and plot!
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(lats,t_rad, color = 'red', label='Hot Earth')
    ax.plot(lats,t_rad1, color='blue',label= 'Cold Earth')
    ax.plot(lats,t_rad2, color = 'cyan', label = 'Flash Freeze Earth')
    # Label axes and title
    ax.set_title("Changes in Initial Conditions for Snowball Earth after 10,000 years")
    ax.set_xlabel("Latitude ($^\circ$)")
    ax.set_ylabel("Temperature ($^\circ$C)")
    # Create grid and legend
    ax.grid()
    ax.legend(loc='best')
    # Show plot
    plt.show()

def question_4(tstop = 10000):
    '''
    Produces plot for question 4.
    '''
    # Define gamma range
    gamma_values = np.arange(0.4, 1.45, 0.05).tolist() + np.arange(1.35, 0.35, -0.05).tolist()

    # Storage for results
    results = []
    lats, Temp = None, None

    # Initial setup: start with "cold Earth" initial condition
    for i, gamma in enumerate(gamma_values):
        if i == 0:
            # Start with a cold Earth initial condition
            lats, Temp = snowball_earth(tstop=10000, lam=74, emiss=0.712,D_albedo=True,
                                         initial_condition="cold", solar_mult=True, gamma=gamma)
        else:
            # Use the previous temperature profile as the initial condition
            lats, Temp = snowball_earth(tstop=10000, lam=74, emiss=0.712, D_albedo=True,
                                        solar_mult = True, gamma=gamma, Temp_initial=Temp)

        # Store the result
        results.append((gamma, Temp.copy()))

    # Extract average temperature for plotting (optional)
    avg_temps = [np.mean(temp) for _, temp in results]

    # Plot figure
    plt.figure(figsize=(8, 6))
    plt.plot(gamma_values, avg_temps)
    # Label axes and title
    plt.xlabel("γ (Solar Multiplier)")
    plt.ylabel("Average Equilibrium Temperature (°C)")
    plt.title("Average Global Temperature vs. Gamma")
    # Create grid
    plt.grid()
    # Show plot
    plt.show()