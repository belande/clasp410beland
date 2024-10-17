#!/usr/bin/env python3
'''
A set of tools and routines for solving Lab 03: Enerrgy Balance Model.

Code is organized by prompts in numerical order and are sorted by 
### Prompt #.

n_layer_atmos is a function that calculates the  Energy Balance Atmosphere 
Model.

nuclear_winter is a function that is basically the same as n_layer_atmos. 
Except for the fact that it absorbs incoming shortwave radiation flux at 
the top of the atmosphere.

'''
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
# Let's import an object for creating our own special color map:
from matplotlib.colors import ListedColormap

# Set MPL style sheet
plt.style.use('default')

### Prompts 1 and 2

#  Define some useful constants here.
# Steffan-Boltzman constant.
sigma = 5.67E-8  
# Number of layers
N = 5  
# Emissivity
epsilon = 0.5  

def n_layer_atmos(N, epsilon, S0=1350, albedo=0.33, debug=False):
    '''
    Solve the n-layer atmosphere problem and return temperature at each layer.

    Parameters
    ----------
    N : int
        Set the number of layers.
    epsilon : float, default=1.0
        Set the emissivity of the atmospheric layers.
    albedo : float, default=0.33
        Set the planetary albedo from 0 to 1.
    S0 : float, default=1350
        Set the incoming solar shortwave flux in Watts/m^2.
    debug : boolean, default=False
        Turn on debug output.

    Returns
    -------
    temps : Numpy array of size N+1
        Array of temperatures from the Earth's surface (element 0) through
        each of the N atmospheric layers, from lowest to highest.
    '''

    # Create matrices:
    A = np.zeros([N+1, N+1])
    b = np.zeros(N+1)
    b[0] = -S0/4 * (1-albedo)

    if debug:
        print(f"Populating N+1 x N+1 matrix (N = {N})")

    # Populate our A matrix piece-by-piece.
    for i in range(N+1):
        for j in range(N+1):
            if debug:
                print(f"Calculating point i={i}, j={j}")
            # Diagonal elements are always -2 ('cept at Earth's surface.)
            if i == j:
                A[i, j] = -1*(i > 0) - 1
                # print(f"Result: A[i,j]={A[i,j]}")
            else:
                # This is the pattern we solved for in class!
                m = np.abs(j-i) - 1
                A[i, j] = epsilon * (1-epsilon)**m
    # At Earth's surface, epsilon =1, breaking our pattern.
    # Divide by epsilon along surface to get correct results.
    A[0, 1:] /= epsilon

    # Verify our A matrix.
    if debug:
        print(A)

    # Get the inverse of our A matrix.
    Ainv = np.linalg.inv(A)

    # Multiply Ainv by b to get Fluxes.
    fluxes = np.matmul(Ainv, b)

    # Convert fluxes to temperatures.
    # Fluxes for all atmospheric layers
    temps = (fluxes/epsilon/sigma)**0.25
    temps[0] = (fluxes[0]/sigma)**0.25  # Flux at ground: epsilon=1.

    return temps


### Prompt 3


# Run the model for a single layer atmosphere across a range of emissivities
emissivities = np.linspace(0.01, 1, 100)
surface_temps = np.array([n_layer_atmos(N=1, epsilon=e)[0] for e in emissivities])

# Plot surface temperature vs emissivity
plt.figure(figsize=(8, 6))
plt.plot(emissivities, surface_temps, label='Surface Temperature', color = "red")
# Label Axes
plt.xlabel('Emissivity')
plt.ylabel('Surface Temperature (K)')
plt.title('Surface Temperature vs. Emissivity for a Single-Layer Atmosphere')
# Create Legend and Grid
plt.legend()
plt.grid(True)
# Show Plot
plt.show()

# Create variable temps to store data
temps = n_layer_atmos(N, epsilon)

# Plot altitude vs. temperature
plt.figure(figsize=(8, 6))
altitude = np.arange(N+1)
plt.plot(temps, altitude, marker='o', color='blue', label='Temperature Profile')

# Label axes
plt.xlabel('Temperature (K)')
plt.ylabel('Atmosphere Layer')
plt.title('Altitude vs. Temperature Profile in a Multi-Layer Atmosphere')
# Create Legend
plt.legend()
# Create Grid
plt.grid(True)
# Show plot
plt.show()

### Prompt 4

# Create variable temps_v to store data
temps_v = n_layer_atmos(N, epsilon, 2600)

# Plot altitude vs. temperature
plt.figure(figsize=(8, 6))
altitude = np.arange(N+1)
plt.plot(temps_v, altitude, marker='o', color='blue', label='Temperature Profile')

# Label axes
plt.xlabel('Temperature (K)')
plt.ylabel('Atmosphere Layer')
plt.title('Altitude vs. Temperature Profile in a 31-Layer Atmosphere on Venus')
# Create Legend
plt.legend()
#Create Grid
plt.grid(True)

# Get the temperature at the surface layer (layer 0)
surface_temp = temps_v[0]
print(f"Venus Surface Temperature: {surface_temp:.2f} K")
# Show plot
plt.show()

### Prompt 5

def nuclear_winter(N, epsilon, S0=1350, albedo=0.33, debug=False):
    '''
    Solve the n-layer atmosphere problem with solar flux fully absorbed by the top layer.

    Parameters
    ----------
    Parameters
    ----------
    N : int
        Set the number of layers.
    epsilon : float, default=1.0
        Set the emissivity of the atmospheric layers.
    albedo : float, default=0.33
        Set the planetary albedo from 0 to 1.
    S0 : float, default=1350
        Set the incoming solar shortwave flux in Watts/m^2.
    debug : boolean, default=False
        Turn on debug output.

    Returns
    -------
    temps : Numpy array of size N+1
        Array of temperatures from the Earth's surface (element 0) through
        each of the N atmospheric layers, from lowest to highest.
    '''


    # Create matrices:
    A = np.zeros([N+1, N+1])
    b = np.zeros(N+1)

    # Now, solar flux is absorbed by the top atmospheric layer, so set it to the last element in b.
    b[N] = -S0 / 4 * (1 - albedo)  # Incoming solar flux, fully absorbed by the top layer.

    if debug:
        print(f"Populating N+1 x N+1 matrix (N = {N})")

    # Populate the A matrix
    for i in range(N+1):
        for j in range(N+1):
            if debug:
                print(f"Calculating point i={i}, j={j}")
            # Diagonal elements are always -2, except at the Earth's surface
            if i == j:
                A[i, j] = -1 * (i > 0) - 1
            else:
                m = np.abs(j - i) - 1
                A[i, j] = epsilon * (1 - epsilon) ** m
    # Modify the first row for the surface (epsilon = 1)
    A[0, 1:] /= epsilon

    if debug:
        print(A)

    # Invert the A matrix and solve for fluxes
    Ainv = np.linalg.inv(A)
    fluxes = np.matmul(Ainv, b)

    # Convert fluxes to temperatures
    temps_n = (fluxes / epsilon / sigma) ** 0.25
    temps_n[0] = (fluxes[0] / sigma) ** 0.25

    return temps_n


# Create variable temps_n to store data
temps_n = nuclear_winter(N, epsilon)
# Plot Figure
plt.figure(figsize=(8, 6))
altitude = np.arange(N+1)
plt.plot(temps_n, altitude, marker='o', color='blue', label='Temperature Profile')

# Label axes
plt.xlabel('Temperature (K)')
plt.ylabel('Atmosphere Layer')
plt.title('Altitude vs. Temperature Profile in a Nuclear Winter on Earth')
# Create Legend
plt.legend()
# Turn on grid
plt.grid(True)
# Example: Get the temperature at the surface layer (layer 0)
surface_temp = temps_n[0]
print(f"Nuclear Winter Temperature: {surface_temp:.2f} K")
# Show plot
plt.show()