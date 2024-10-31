#!/usr/bin/env python3
'''
A set of tools and routines for solving Lab 03: Permafrost Thawing

heatdiff is a function that calculates the diffusion equation using 
a forward difference. This function is helpful for answering prompt 1
from the lab.

temp_kanger is a function returning the temperature at the surface
in Kangerlussuaq, Greenland.

permafrost is the heatdiff function, but applied specifically to 
permfrost. This means that the initial and boundary conditions are
different and the function is dependent on temp_kanger.



'''
# import libraries
import numpy as np
import matplotlib.pyplot as plt

# Parameters
xmax = 100  # 100 meters depth
# Change to value of tmax to find steady state.
tmax = 100 *365 # Run for 100 years to approximate steady-state
dx = 1      # Spatial step in meters
dt = 1      # Time step in days
# switch units from c2 from mm^2/s to m^2/day
c2 = (0.25 * 24 *3600)/(1000**2)


def heatdiff(xmax, tmax, dx, dt, c2=1, debug=False):
    '''
    Parameters:
    -----------
    xmax : float
        The upper boundary of the spatial domain in meters.
    tmax : float
        The maximum time for which the solution is computed, in seconds.
    dx : float
        The spatial step size, or interval between points in the spatial 
        domain.
    dt : float
        The time step size, or interval between points in the temporal 
        domain.
    c2 : float, optional
        A constant related to the speed of diffusion. Default is 1.
    debug : bool, optional
        If True, prints debugging information about grid dimensions, 
        steps, and initial grids.

    Returns:
    --------
    xgrid : np.ndarray
        1D array representing the spatial grid, from 0 to xmax with step dx.
    tgrid : np.ndarray
        1D array representing the time grid, from 0 to tmax with step dt.
    U : np.ndarray
        2D array where each row represents temperature at each spatial point over time steps. 
        Size is (M, N) where M is the number of spatial points and N is the number of time points.
    '''
    if dt > dx**2/(2*c2):
        raise ValueError('dt is too large! Must be less than dx**2 / (2*c2) for stability')
    # Start by calculating size of array: MxN
    M = int(np.round(xmax / dx + 1))
    N = int(np.round(tmax / dt + 1))

    xgrid, tgrid = np.arange(0, xmax+dx, dx), np.arange(0, tmax+dt, dt)

    if debug:
        print(f'Our grid goes from 0 to {xmax}m and 0 to {tmax}s')
        print(f'Our spatial step is {dx} and time step is {dt}')
        print(f'There are {M} points in space and {N} points in time.')
        print('Here is our spatial grid:')
        print(xgrid)
        print('Here is our time grid:')
        print(tgrid)

    # Initialize our data array:
    U = np.zeros((M, N))

    # Set initial conditions:
    U[:, 0] = 4*xgrid - 4*xgrid**2

    # Set boundary conditions:
    U[0, :] = 0
    U[-1, :] = 0

    # Set our "r" constant.
    r = c2 * dt / dx**2

    # Solve! Forward differnce ahoy.
    for j in range(N-1):
        U[1:-1, j+1] = (1-2*r) * U[1:-1, j] + \
            r*(U[2:, j] + U[:-2, j])
            

    # Return grid and result:
    return xgrid, tgrid, U

# Print verification to help answer prompt 1
print(heatdiff(1, 0.2,0.2,0.02))




# Kangerlussuaq average temperature data (in °C)
t_kanger = np.array([-19.7, -21.0, -17.0, -8.4, 2.3, 8.4, 10.7, 8.5, \
                     3.1, -6.0, -12.0, -16.9])

# Calcuate uniform changes in temperature for prompt 3
t_kanger_half = t_kanger + 0.5
t_kanger1 = t_kanger + 1
t_kanger3 = t_kanger + 3

def temp_kanger(t):
    '''
    Function returning the temperature at the surface in Kangerlussuaq, Greenland.

    In order to apply which version of kanger temperature array you want, just 
    change the t_kanger values in the function to the values you want.
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()
    return t_amp * np.sin(2 * np.pi / 365 * t - np.pi / 2) + t_kanger.mean()

def permafrost(xmax, tmax, dx, dt, c2 = c2, debug=False):
    '''
    Parameters:
    -----------
    xmax : float
        The upper boundary of the spatial domain in meters.
    tmax : float
        The maximum time for which the solution is computed, in seconds.
    dx : float
        The spatial step size, or interval between points in the spatial domain.
    dt : float
        The time step size, or interval between points in the temporal domain.
    c2 : float, optional
        A constant related to the speed of diffusion. Default is 1.
    debug : bool, optional
        If True, prints debugging information about grid dimensions, steps, and initial grids.

    Returns:
    --------
    xgrid : np.ndarray
        1D array representing the spatial grid, from 0 to xmax with step dx.
    tgrid : np.ndarray
        1D array representing the time grid, from 0 to tmax with step dt.
    U : np.ndarray
        2D array where each row represents temperature at each spatial point over time steps. 
        Size is (M, N) where M is the number of spatial points and N is the number of time points.    
    '''
    if dt > dx**2 / (2 * c2):
        raise ValueError('dt is too large! Must be less than dx**2 / (2*c2) for stability')
    
    # Calculate grid size
    M = int(np.round(xmax / dx + 1))
    N = int(np.round(tmax / dt + 1))

    xgrid = np.arange(0, xmax + dx, dx)
    tgrid = np.arange(0, tmax + dt, dt)

    if debug:
        print(f'Spatial grid points: {M}, Time grid points: {N}')

    # Initialize temperature array U (depth x time)
    U = np.zeros((M, N))

    # Initial and boundary conditions
    U[:, 0] = 0  # Initial temperature set to 0°C throughout
    U[-1, :] = 5  # Lower boundary at 5°C (geothermal warming)
    
    r = c2 * dt / dx**2

    # Time-stepping loop
    for j in range(N - 1):
        # Update the surface boundary condition based on time tgrid[j]
        U[0, j] = temp_kanger(j * dt)
        
        # Update the interior points
        U[1:-1, j + 1] = (1 - 2 * r) * U[1:-1, j] + r * (U[2:, j] + U[:-2, j])

    return xgrid, tgrid, U

# Run the permafrost function to simulate heat diffusion
xgrid, tgrid, U = permafrost(xmax, tmax, dx, dt)

# Plot heatmap of time vs depth
plt.figure(figsize=(10, 6))
plt.imshow(U, aspect='auto', extent=[tgrid.min()/365, tgrid.max()/365, xgrid.max(), xgrid.min()], cmap='seismic', vmin=-25, vmax=25)
# Plot colorbar
plt.colorbar(label='Temperature (°C)')
# Label axes and title
plt.xlabel('Time (years)')
plt.ylabel('Depth (meters)')
plt.title('Ground Temperature: Kangerlussauq, Greenland')
plt.show()

# Second Plot: Winter and Summer profile of temperatures over the final year
# Index for the final 365 days, considering dt=0.25 days
loc = int(-365 / dt)
# Minimum temperature profile for each depth
winter = U[:, loc:].min(axis=1) 
# Maximum temperature profile for each depth 
summer = U[:, loc:].max(axis=1)  

# Find the shallowest and deepest points where the summer temperature profile stays below 0°C
shallow_permafrost_idx = np.where((summer < 0) & (xgrid >= 0) & (xgrid <= 20))[0][0]  # Shallowest within 0-20m
deep_permafrost_idx = np.where(summer < 0)[0][-1]  # Deepest point where temp < 0

# Get the depth range for permafrost shading
shallow_permafrost_depth = xgrid[shallow_permafrost_idx]
deep_permafrost_depth = xgrid[deep_permafrost_idx]

# Print the results
print(f"Active layer ends at approximately {shallow_permafrost_depth:.2f} meters depth.")
print(f"Permafrost ends at approximately {deep_permafrost_depth:.2f} meters depth.")

# Plot second figure
fig, ax2 = plt.subplots(1, 1)
ax2.plot(winter, xgrid, label='Winter')
ax2.plot(summer, xgrid, linestyle='--', color='red', label='Summer')

# Shade the permafrost region between the shallow and deep boundaries
ax2.fill_betweenx(xgrid, -15, 15, where=(xgrid >= shallow_permafrost_depth) & (xgrid <= deep_permafrost_depth), 
                  color='cyan', alpha=0.3, label='Permafrost')

# Label title and axes
ax2.set_xlabel('Temperature (°C)')
ax2.set_ylabel('Depth (meters)')
ax2.set_title('Ground Temperature: Kangerlussauq, Greenland')
# Set limits for x and y axes
ax2.set_ylim(0, 100)
ax2.set_xlim(-15, 15)
 # Invert y-axis so depth increases downward
ax2.invert_yaxis() 
# Create Legend
ax2.legend()
plt.show()
