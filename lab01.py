#!/usr/bin/env python3

'''
This file performs fire/disease spread simulations.

To get solution for lab 1: Run these commands:

>>> blah
>>> blah blah

'''

import numpy as np
import matplotlib.pyplot as plt
# Let's import an object for creating our own special color map:
from matplotlib.colors import ListedColormap

# Example of updated parameters
nx, ny = 100, 100  # Grid size.
prob_spread = 0.69  # Reduced probability of spread.
prob_bare = 0.1    # Lowered the chance of bare patches.
p_fatal = 0.43      # Lowered fatality probability.
prob_start = 0.0   # Chance of cell to start on fire.

# plt.style.use('fivethirtyeight')


def fire_spread(nNorth= ny, nEast= nx, maxiter=100, prob_spread = prob_spread, prob_bare=prob_bare, fig_size = (16,12)):
    '''
    This function performs a fire/disease spread simultion.

    Parameters
    ==========
    nNorth, nEast : integer, defaults to 3
        Set the north-south (i) and east-west (j) size of grid.
        Default is 3 squares in each direction.
    maxiter : int, defaults to 4
        Set the maximum number of iterations including initial condition
    
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
    cbar.ax.set_yticklabels(['Bare', 'Forest', 'Fire'])
    ax.set_ylabel('y (km)')
    ax.set_xlabel('x (km)')
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
    Vary burn rate an see how fast fire is
    '''
    prob = np.arange(0,1,0.5)
    nsteps = np.zeros(prob.size)

    for i,p in enumerate(prob):
        print(f'Burning for prob_spread = {p}')
        nsteps[i] = fire_spread(nEast =5, prob_spread = p, maxiter =100)

     
        


def buckeyeitis(nUp= ny, nDown= nx, maxiter=999, prob_spread = prob_spread, G0_Blu3=prob_bare, \
                p_fatal = p_fatal, fig_size = (16,12)):
    '''
    This function performs a fire/disease spread simulation.

    Parameters:
    ==========
    nUp, nDown : int : Grid size in the y and x directions
    maxiter : int : Maximum number of iterations
    prob_spread : float : Probability of disease/fire spreading
    G0_Blu3 : float : Probability of starting as a fatal case (bare)
    p_fatal : float : Probability of becoming fatal when sick
    fig_size : tuple : Figure size for the plots
    '''

    # Create population and set initial condition
    population = np.zeros([maxiter, nUp, nDown]) + 2  # Initialize with healthy state (2)

    # Apply prob_bare: some cells start as fatalities (state 0)
    fatal_mask = np.random.rand(nUp, nDown) < G0_Blu3
    population[0, fatal_mask] = 1  # Set these cells as immune

    # Set disease in the center of the grid (state 3 = sick)
    istart, jstart = nUp // 2, nDown // 2
    population[0, istart, jstart] = 3

    # Custom color map
    forest_cmap = ListedColormap(['black', 'tan', 'darkgreen', 'crimson'])

    # Plot initial condition
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    contour = ax.pcolor(population[0, :, :], cmap=forest_cmap, vmin=0, vmax=3)
    ax.set_title(f'Iteration = {0:03d}')
    cbar = plt.colorbar(contour, ax=ax, ticks=[0.5, 1.5, 2.5, 3.5])
    cbar.ax.set_yticklabels(['Fatality', 'Immune', 'Healthy', 'Sick'])
    ax.set_ylabel('y (km)')
    ax.set_xlabel('x (km)')
    fig.savefig(f'fig{0:04d}.png')

    # Propagate the simulation
    for k in range(maxiter-1):
        ignite = np.random.rand(nUp, nDown)
        population[k+1, :, :] = population[k, :, :]

        # Spread the fire/disease in all four directions
        doburnS = (population[k, 0:-1, :] == 3) & (population[k, 1:, :] == 2) & \
            (ignite[1:, :] <= prob_spread)
        population[k+1, 1:, :][doburnS] = 3

        doburnN = (population[k, 1:, :] == 3) & (population[k, 0:-1, :] == 2) & \
            (ignite[0:-1, :] <= prob_spread)
        population[k+1, 0:-1, :][doburnN] = 3

        doburnE = (population[k, :, 0:-1] == 3) & (population[k, :, 1:] == 2) & \
            (ignite[:, 1:] <= prob_spread)
        population[k+1, :, 1:][doburnE] = 3

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
        cbar = plt.colorbar(contour, ax=ax, ticks=[0.5, 1.5, 2.5, 3.5])
        cbar.ax.set_yticklabels(['Fatality', 'Immune', 'Healthy', 'Sick'])
        ax.set_ylabel('y (km)')
        ax.set_xlabel('x (km)')
        fig.savefig(f'fig{k+1:04d}.png')
        plt.close('all')

        # Quit if no spots are sick
        nBurn = (population[k+1, :, :] == 3).sum()
        if nBurn == 0:
            print(f'Spread completed in {k+1} steps')
            break

    return k+1
