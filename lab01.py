#!/usr/bin/env python3

'''
This file performs fire/disease spread simulations.

To get solution for lab 1: Run these commands:

>>> blah
>>> blah blah

'''

import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

plt.style.use('fivethirtyeight')


def fire_spread(nNorth=3, nEast=3, maxiter=4):
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

    # Set fire! To the center of the forest.
    forest[0, 1, 1] = 3

    # Plot initial condition
    fig, ax = plt.subplots(1, 1)
    contour = ax.matshow(forest[0, :, :], vmin=1, vmax=3)
    ax.set_title(f'Iteration = {0:03d}')
    plt.colorbar(contour, ax=ax)

    # Propagate the solution.
    for k in range(maxiter-1):
        # Use current step to set next step:
        forest[k+1, :, :] = forest[k, :, :]

        # Burn in each cardinal direction.
        # From north to south:
        for i in range(nNorth - 1):
            for j in range(nEast):
                # Is current patch burning AND adjacent forested?
                if (forest[k, i, j] == 3) & (forest[k, i+1, j] == 2):
                        # Spread fire to new square:
                        forest[k+1, i+1, j] = 3
        # From South to North:
        for i in range(1, nNorth):  # Start from index 1 to avoid out of bounds
            for j in range(nEast):
                if (forest[k, i, j] == 3) & (forest[k, i-1, j] == 2):
                    forest[k+1, i-1, j] = 3
        # From East to West       
        for i in range(nNorth):
            for j in range(nEast - 1):
                # Is current patch burning AND adjacent forested?
                if (forest[k, i, j] == 3) & (forest[k, i, j+1] == 2):
                    # Spread fire to new square:
                    forest[k+1, i, j+1] = 3  
        # From West to East
        for i in range(nNorth):  # Start from index 1 to avoid out of bounds
            for j in range(1, nEast):
                if (forest[k, i, j] == 3) & (forest[k, i, j-1] == 2):
                    forest[k+1, i, j-1] = 3            



        # Set currently burning to bare:
        wasburn = forest[k, :, :] == 3  # Find cells that WERE burning
        forest[k+1, wasburn] = 1       # ...they are NOW bare.

        fig, ax = plt.subplots(1, 1)
        contour = ax.matshow(forest[k+1, :, :], vmin=1, vmax=3)
        ax.set_title(f'Iteration = {k+1:03d}')
        plt.colorbar(contour, ax=ax)
        plt.show()

