#!/usr/bin/env python3

'''
This file performs fire/disease spread simulations.

To get solution for lab 1: Run these commands:

>>> fire_spread helps give the answers to any problems associated with the wildfires
>>> buckeyeitis is basically the same function as fire_spread, however it is targeted 
    to answer the disease spread questions.
>>> Below the functions are averages to help support the answers in the lab report.

'''
# Import Needed Libraries
import numpy as np
import matplotlib.pyplot as plt
# Let's import an object for creating our own special color map:
from matplotlib.colors import ListedColormap

# Declare constants for functions
nx, ny = 100, 100  # Grid size.
prob_spread = 1.00  # probability of fire/disease spreading.
prob_bare = 0.00   # Probability of bare/immune patches.
p_fatal = 0.50     # Fatality probability.
prob_start = 0.0   # Chance of cell to start on fire.



def fire_spread(nNorth= ny, nEast= nx, maxiter=999, prob_spread = prob_spread, \
                prob_bare=prob_bare, fig_size = (16,12)):
    '''
    This function performs a fire spread simultion.

    Parameters
    ==========
    nNorth, nEast : integer, defaults to ny & nx
        Set the north-south (i) and east-west (j) size of grid.
        Default is ny & nx from constants declared above
    maxiter : int, defaults to 100
        Set the maximum number of iterations including initial condition
    prob_spread: float, 
        Probability of fire spreading.
    prob_bare: float, 
        Probability of ground starting bare. This means that the ground cannot catch fire.
    fig_size: tuple,
        Figure size for the plots during each iteration

    
    '''

    # Create forest and set initial condition
    forest = np.zeros([maxiter, nNorth, nEast]) + 2

    # Apply prob_bare: some cells start as bare (state 1)
    bare_mask = np.random.rand(nNorth, nEast) < prob_bare
    forest[0, bare_mask] = 1  # Set these cells as bare

    # Set fire! To the center of the forest.
    istart, jstart = nNorth//2, nEast//2
    forest[0, istart, jstart] = 3


    # Generate our custom segmented color map for this project.
    # We can specify colors by names and then create a colormap that only uses
    # those names. We have 3 funadmental states, so we want only 3 colors.
    # Color info: https://matplotlib.org/stable/gallery/color/named_colors.html
    forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])

    # Plot initial condition
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    contour = ax.pcolor(forest[0, :, :], cmap = forest_cmap, vmin=1, vmax=3)
    ax.set_title(f'Iteration = {0:03d}')
     # Create colorbar with custom labels for each iteration
    cbar = plt.colorbar(contour, ax=ax, ticks=[1.33, 2, 2.67])
    # Label Colorbar
    cbar.ax.set_yticklabels(['Bare', 'Forest', 'Fire'])
    #Label Axes
    ax.set_ylabel('y (km)')
    ax.set_xlabel('x (km)')
    # Save initial condition figure
    fig.savefig(f'fig{0:04d}.png')  

    # Propagate the solution.
    for k in range(maxiter-1):
        # Set change to burn:
        ignite = np.random.rand(nNorth, nEast)
        # Use current step to set next step:
        forest[k+1, :, :] = forest[k, :, :]

        # Burn from North to South
        doburnS = (forest[k,0:-1,:]==3) & (forest[k,1:,:] ==2) & \
            (ignite[1:,:] <= prob_spread)
        forest[k+1,1:,:][doburnS] = 3
        # Burn in each cardinal direction.

        # From South to North:
        doburnN = (forest[k,1:,:]==3) & (forest[k,0:-1,:] ==2) & \
            (ignite[0:-1,:] <= prob_spread)
        forest[k+1,0:-1,:][doburnN] = 3

        # From East to West       
        doburnE = (forest[k,:,0:-1]==3) & (forest[k,:,1:] ==2) & \
            (ignite[:,1:] <= prob_spread)
        forest[k+1,:,1:][doburnE] = 3
        # From West to East
        doburnW = (forest[k,:,1:]==3) & (forest[k,:,0:-1] ==2) & \
            (ignite[:,0:-1] <= prob_spread)
        forest[k+1,:,0:-1][doburnW] = 3        

        # Set currently burning to bare:
        wasburn = forest[k, :, :] == 3  # Find cells that WERE burning
        forest[k+1, wasburn] = 1       # ...they are NOW bare.

        fig, ax = plt.subplots(1, 1, figsize = fig_size)
        contour = ax.pcolor(forest[k+1, :, :],cmap = forest_cmap, vmin=1, vmax=3)
        # Label Title and Axes
        ax.set_title(f'Iteration = {k+1:03d}')
        ax.set_ylabel('y (km)')
        ax.set_xlabel('x (km)')
        # Create colorbar with custom labels for each iteration
        cbar = plt.colorbar(contour, ax=ax, ticks=[1.33, 2, 2.67])
        cbar.ax.set_yticklabels(['Bare', 'Forest', 'Fire'])
        

        fig.savefig(f'fig{k+1:04d}.png')
        plt.close('all')

        # Quit if no spots are on fire
        nBurn = (forest[k+1, :,:] == 3).sum()
        if nBurn == 0:
            print(f'Burn completed in {k+1} steps')
            break
    return k+1


def explore_burnrate():
    '''
    Vary burn rate to see how fast fire is
    '''
    prob = np.arange(0,1,0.5)
    nsteps = np.zeros(prob.size)

    for i,p in enumerate(prob):
        print(f'Burning for prob_spread = {p}')
        nsteps[i] = fire_spread(nEast =5, prob_spread = p, maxiter =100)

     
        


def buckeyeitis(nUp= ny, nDown= nx, maxiter=999, prob_spread = prob_spread, G0_Blu3=prob_bare, \
                p_fatal = p_fatal, fig_size = (16,12)):
    '''
    This function performs a disease spread simulation.

    Parameters:
    ==========
    nUp, nDown : int, 
        Set the up-down (i) and left-right (j) size of grid.
        Default is ny & nx from constants declared above
    maxiter : int,
        Set the maximum number of iterations including initial condition
    prob_spread : float, 
        Probability of disease spreading
    G0_Blu3 : float, 
        Probability of starting as an immune patient (bare)
    p_fatal : float, 
        Probability of becoming fatal when sick
    fig_size : tuple, 
        Figure size for the plots during each iteration
    '''

    # Create population and set initial condition
    population = np.zeros([maxiter, nUp, nDown]) + 2  # Initialize with healthy state (2)

    # Apply prob_bare: some cells start as immune (state 1)
    fatal_mask = np.random.rand(nUp, nDown) < G0_Blu3
    population[0, fatal_mask] = 1  # Set these cells as immune

    # Set disease in the center of the grid (state 3 = sick)
    istart, jstart = nUp // 2, nDown // 2
    population[0, istart, jstart] = 3

    # Custom color map
    forest_cmap = ListedColormap(['black', 'blue', 'yellow', 'crimson'])

    # Plot initial condition
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    contour = ax.pcolor(population[0, :, :], cmap=forest_cmap, vmin=0, vmax=3)
    ax.set_title(f'Iteration = {0:03d}')
     # Create colorbar with custom labels for each iteration
    cbar = plt.colorbar(contour, ax=ax, ticks=[0.33, 1.1, 1.9, 2.67])
    cbar.ax.set_yticklabels(['Fatality', 'Immune', 'Healthy', 'Sick'])
    # Label Axes
    ax.set_ylabel('People')
    ax.set_xlabel('People')
    # Save figures
    fig.savefig(f'fig{0:04d}.png')

    # Propagate the simulation
    for k in range(maxiter-1):
        ignite = np.random.rand(nUp, nDown)
        population[k+1, :, :] = population[k, :, :]

        # Spread the disease in all four directions
        # Spread from top to bottom
        dospreadS = (population[k, 0:-1, :] == 3) & (population[k, 1:, :] == 2) & \
            (ignite[1:, :] <= prob_spread)
        population[k+1, 1:, :][dospreadS] = 3
       
        # Spread from bottom to top
        dospreadN = (population[k, 1:, :] == 3) & (population[k, 0:-1, :] == 2) & \
            (ignite[0:-1, :] <= prob_spread)
        population[k+1, 0:-1, :][dospreadN] = 3
       
        # Spread from left to right
        doburnE = (population[k, :, 0:-1] == 3) & (population[k, :, 1:] == 2) & \
            (ignite[:, 1:] <= prob_spread)
        population[k+1, :, 1:][doburnE] = 3
        
        # Spread from right to left
        doburnW = (population[k, :, 1:] == 3) & (population[k, :, 0:-1] == 2) & \
            (ignite[:, 0:-1] <= prob_spread)
        population[k+1, :, 0:-1][doburnW] = 3

        # Determine fate of sick individuals
        fatal_mask = np.random.rand(nUp, nDown) < p_fatal
        sick_mask = population[k, :, :] == 3

        # Sick individuals either die (0) or recover (become immune, 1)
        population[k+1, sick_mask & fatal_mask] = 0  # Fatality
        population[k+1, sick_mask & ~fatal_mask] = 1  # Immune

        # Plot each iteration
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        contour = ax.pcolor(population[k+1, :, :], cmap=forest_cmap, vmin=0, vmax=3)
        ax.set_title(f'Iteration = {k+1:03d}')
         # Create colorbar with custom labels for each iteration
        cbar = plt.colorbar(contour, ax=ax, ticks=[0.33, 1.1, 1.9, 2.67])
        cbar.ax.set_yticklabels(['Fatality', 'Immune', 'Healthy', 'Sick'])
        #Label Axes
        ax.set_ylabel('people')
        ax.set_xlabel('people')
        # Save Figure
        fig.savefig(f'fig{k+1:04d}.png')
        # Close all figures
        plt.close('all')

        # Quit if no spots are sick
        nBurn = (population[k+1, :, :] == 3).sum()
        if nBurn == 0:
            print(f'Spread completed in {k+1} steps')
            break

    return k+1

# fire_spread average for p_spread = 0.75
spread_avg = (106 + 114 + 103 + 103 + 106)/5
print(f'The fire_spread average for p_spread = 0.75 is{spread_avg}')
# fire_spread average for p_spread = 0.50
spread_avg1 = (19 + 65 + 38 + 16 + 12)/5
print(f'The fire_spread average for p_spread = 0.50 is{spread_avg1}')
# fire_spread average for p_spread = 0.25
spread_avg2 = (1 + 9 + 1 + 4 + 1)/5
print(f'The fire_spread average for p_spread = 0.25 is{spread_avg2}')

# fire_spread average for p_bare = 0.25
spread_avg3 = (103 + 118 + 104 + 103 + 108)/5
print(f'The fire_spread average for p_bare = 0.25 is{spread_avg3}')
# fire_spread average for p_spread = 0.50
spread_avg4 = (5 + 25 + 2 + 45 + 21)/5
print(f'The fire_spread average for p_bare = 0.50 is{spread_avg4}')
# fire_spread average for p_spread = 0.75
spread_avg5 = (6 + 3 + 8 + 1 + 3)/5
print(f'The fire_spread average for p_spread = 0.75 is{spread_avg5}')

# buckeyeits average for p_fatal = 0.25
spread_avg6 = (107 + 103 + 104 + 108 + 105)/5
print(f'The fire_spread average for p_fatal = 0.25 is{spread_avg6}')
# buckeyeitis average for p_fatal = 0.50
spread_avg7 = (101 + 103 + 103 + 104 + 105)/5
print(f'The fire_spread average for p_fatal = 0.50 is{spread_avg7}')
# buckeyeitis average for p_fatal = 0.75
spread_avg8 = (104 + 102 + 103 + 106 + 101)/5
print(f'The fire_spread average for p_fatal = 0.75 is{spread_avg8}')

# buckeyeits average for G0_Blu3 = 0.25
spread_avg9 = (196 + 155 + 128 + 167 + 201)/5
print(f'The fire_spread average for G0_Blu3 = 0.25 is{spread_avg9}')
# buckeyeitis average for G0_Blu3 = 0.50
spread_avg10 = (11 + 21 + 3 + 3 + 5)/5
print(f'The fire_spread average for G0_Blu3 = 0.50 is{spread_avg10}')
# buckeyeitis average for G0_Blu3 = 0.75
spread_avg11 = (1 + 2 + 2 + 3 + 6)/5
print(f'The fire_spread average for G0_Blu3 = 0.75 is{spread_avg11}')