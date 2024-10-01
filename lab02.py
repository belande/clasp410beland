#!/usr/bin/env python3

'''
This file performs Ordinary Differential Equation Solvers

To get solution for lab 1: Run these commands:

dNdt_comp is a function that calculates the Lotka-Volterra competition equation set.

dNdt_pred is a function that calculates the Lotka-Volterra predator-prey equation set.

N_init is a function that calculates what the initial states of N should be when trying to find 
equilibrium states in the Lotka-Volterra competition equation set.

Euler_solve is the first order Euler solver function to help calculate dNdt_comp and dNdt_pred.

solve_RK8 is the 8th order solver function from scipy to help calculate a more accurate result.

Solve Equation Section: Area of code that you modify initial conditions, coefficients, and timesteps 
to help answer questions for the lab 02.

Three plots are below the Solve Equation Section to help answer questions for Lab02.


'''
# Import Needed Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# Let's import an object for creating our own special color map:
from matplotlib.colors import ListedColormap

# Set MPL style sheet
plt.style.use('fivethirtyeight')

# Creat function for competition equations
def dNdt_comp(t, N, a=1, b=2, c=1, d=3, model = 'comp'):
    '''
This function calculates the Lotka-Volterra competition equations for
two species. Given normalized populations, `N1` and `N2`, as well as the
Lab 1 - 2
four coefficients representing population growth and decline,
calculate the time derivatives dN_1/dt and dN_2/dt and return to the
caller.
This function accepts `t`, or time, as an input parameter to be
compliant with Scipy's ODE solver. However, it is not used in this
function.
Parameters
----------
t : float
The current time (not used here).
N : two-element list
The current value of N1 and N2 as a list (e.g., [N1, N2]).
a, b, c, d : float, defaults=1, 2, 1, 3
The value of the Lotka-Volterra coefficients.
Returns
-------
dN1dt, dN2dt : floats
The time derivatives of `N1` and `N2`.
    '''
    # Here, N is a two-element list such that N1=N[0] and N2=N[1]

    dN1dt = a * N[0] * (1 - N[0]) - b * N[0] * N[1]
    dN2dt = c * N[1] * (1 - N[1]) - d * N[1] * N[0]
    
    return dN1dt, dN2dt

# Create Function for predator-prey equations
def dNdt_pred(t, N, a=1, b=2, c=1, d=3, model = 'pred'):
    '''
    This function calculates the Lotka-Volterra Predator-Prey equations for
    two species. Given normalized populations, `N1` and `N2`, as well as the
    Lab 1 - 2
    four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the
    caller.
    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this
    function.
    Parameters
    ----------
    t : float
    The current time (not used here).
    N : two-element list
    The current value of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d : float, defaults=1, 2, 1, 3
    The value of the Lotka-Volterra coefficients.
    Returns
    -------
    dN1dt, dN2dt : floats
    The time derivatives of `N1` and `N2`.
    '''


    dN1dt = a * N[0] - b * N[0] * N[1]
    dN2dt = -c * N[1] + d * N[1] * N[0]


    return dN1dt, dN2dt
# Create equilibrium state function
def N_init(a, b, c, d):
    '''
    Calculate the equilibrium points N1 and N2 given the coefficients
    a, b, c, and d for the Lotka-Volterra system.

    Parameters
    ----------
    a, b, c, d : float
        Coefficients of the Lotka-Volterra model.
    
    Returns
    -------
    N1 : float
        The equilibrium population for species 1.
    N2 : float
        The equilibrium population for species 2.
    '''
    # Calcualte denometer for equation
    denominator = (c * a - b * d)
    
    # Find N1 and N2
    N1 = (c * (a - b)) / denominator
    N2 = (a * (c - d)) / denominator
    
    return N1, N2

# Create Euler Function
def euler_solve(func, N1_init=0.5, N2_init=0.5, dT=0.1, t_final=100.0, \
                a=1, b=2, c=1, d=3, model='comp'):
    '''
    Solve the Lotka-Volterra competition and predator/prey equations using
    a first order Euler.
    Parameters
    ----------
    func : function
    A python function that takes `time`, [`N1`, `N2`] as inputs and
    returns the time derivative of N1 and N2.
    N1_init, N2_init : float
    Initial conditions for `N1` and `N2`, ranging from (0,1]
    dT : float, default=10
    Largest timestep allowed in years.
    t_final : float, default=100
    Integrate until this value is reached, in years.
    a, b, c, d : float, default=1, 2, 1, 3
    Lotka-Volterra coefficient values
    Returns
    -------
    time : Numpy array
    Time elapsed in years.
        N1, N2 : Numpy arrays
    Normalized population density solutions.
    '''
    
    # Initialize time array
    time = np.arange(0, t_final + dT, dT)
    
    # Initialize arrays for storing N1 and N2 over time
    N1 = np.zeros_like(time)
    N2 = np.zeros_like(time)
    
    # Set initial conditions
    N1[0] = N1_init
    N2[0] = N2_init
    
    # Iterate through time, applying the Euler method
    for i in range(1, time.size):
        dN1, dN2 = func(time[i-1], [N1[i-1], N2[i-1]], a, b, c, d, model=model)
        N1[i] = N1[i-1] + dN1 * dT
        N2[i] = N2[i-1] + dN2 * dT
    
    return time, N1, N2

# Create RK8 function
def solve_rk8(func, N1_init=.5, N2_init=.5, dT=10, t_final=100.0,
a=1, b=2, c=1, d=3, model = 'pred'):
    '''
    Solve the Lotka-Volterra competition and predator/prey equations using
    Scipy's ODE class and the adaptive step 8th order solver.
    Parameters
    ----------
    func : function
    A python function that takes `time`, [`N1`, `N2`] as inputs and
    returns the time derivative of N1 and N2.
    N1_init, N2_init : float
    Initial conditions for `N1` and `N2`, ranging from (0,1]
    dT : float, default=10
    Largest timestep allowed in years.
    t_final : float, default=100
    Integrate until this value is reached, in years.
    a, b, c, d : float, default=1, 2, 1, 3
    Lotka-Volterra coefficient values
    Returns
    -------
    time : Numpy array
    Time elapsed in years.
        N1, N2 : Numpy arrays
    Normalized population density solutions.
    '''
    from scipy.integrate import solve_ivp
    # Configure the initial value problem solver
    result = solve_ivp(func, [0, t_final], [N1_init, N2_init],
                args=[a, b, c, d], method='DOP853', max_step=dT)
    # Perform the integration
    time, N1, N2 = result.t, result.y[0, :], result.y[1, :]
    # Return values to caller.
    return time, N1, N2

# Solve Equations Section
# Solve comp equations with Euler method
time_euler_comp, N1_euler_comp, N2_euler_comp = \
    euler_solve(dNdt_comp, N1_init=0.3, N2_init=0.6, dT=1.0, t_final=100, a=1, b=2, c=1, d=3)
# Solve comp equations with RK8 method
time_rk8_comp, N1_rk8_comp, N2_rk8_comp = \
    solve_rk8(dNdt_comp, N1_init=0.3, N2_init=0.6, dT=10, t_final=100, a=1, b=2, c=1, d=3)

# Solve pred equations with Euler method
time_euler_pred, N1_euler_pred, N2_euler_pred = \
    euler_solve(dNdt_pred, N1_init=0.3, N2_init=0.6, dT=0.05, t_final=100, a=1, b=2, c=1, d=3)
# Solve pred equations with RK8 method
time_rk8_pred, N1_rk8_pred, N2_rk8_pred = \
    solve_rk8(dNdt_pred, N1_init=0.3, N2_init=0.6, dT=10, t_final=100, a=1, b=2, c=1, d=3)

# Plot the results
fig, (ax1, ax2) = plt.subplots(1 , 2 ,figsize=(12, 6))

# Euler method results
ax1.plot(time_euler_comp, N1_euler_comp, label='Euler Species 1', linestyle='-', color='blue', alpha = 0.5)
ax1.plot(time_euler_comp, N2_euler_comp, label='Euler Species 2', linestyle='-', color='red', alpha = 0.5)

# RK8 method results
ax1.plot(time_rk8_comp, N1_rk8_comp, label='RK8 Species 1', linestyle='dotted', color='blue')
ax1.plot(time_rk8_comp, N2_rk8_comp, label='RK8 Species 2', linestyle='dotted', color='red')

# Euler method results
ax2.plot(time_euler_pred, N1_euler_pred, label='Euler Species 1', linestyle='-', color='blue', alpha = 0.5)
ax2.plot(time_euler_pred, N2_euler_pred, label='Euler Species 2', linestyle='-', color='red', alpha = 0.5)

# RK8 method results
ax2.plot(time_rk8_pred, N1_rk8_pred, label='RK8 Species 1', linestyle='dotted', color='blue')
ax2.plot(time_rk8_pred, N2_rk8_pred, label='RK8 Species 2', linestyle='dotted', color='red')

# Add labels and legends to individual subplots
ax1.set_xlabel('Time (years)')
ax1.set_ylabel('Population')
ax1.set_title('Competition')
ax1.legend()
ax1.grid(True)
ax2.set_xlabel('Time (years)')
ax2.set_title('Predator-Prey')
ax2.legend()
ax2.grid(True)
# Add a centered title for both subplots
fig.suptitle('Comparison of Euler and RK8 Methods', fontsize=24)

# Adjust the layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the suptitle
plt.show()


# Create figure to help answer questions 2 & 3.
# Can change the plot based whether you trying to answer competion 
# or preditor-prey questions.

# Plot the results
plt.figure(figsize=(10, 6))

# Euler method results
plt.plot(time_euler_pred, N1_euler_pred, label='Euler Species 1', linestyle='-', color='blue')
plt.plot(time_euler_pred, N2_euler_pred, label='Euler Species 2', linestyle='-', color='red')

# RK8 method results
plt.plot(time_rk8_pred, N1_rk8_pred, label='RK8 Species 1', linestyle='dotted', color='blue')
plt.plot(time_rk8_pred, N2_rk8_pred, label='RK8 Species 2', linestyle='dotted', color='red')

# Label plot
plt.xlabel('Time (years)')
plt.ylabel('Population')
plt.title('Comparison of Euler and RK8 Methods')
plt.legend()
plt.grid(True)
plt.show()


# Create plot to help answer phase diagram section of question 3
plt.figure(figsize=(12,8))
# Phase Diagram Results
plt.plot(N1_euler_pred, N2_euler_pred, color='blue')

# Label Plot
plt.xlabel('Prey')
plt.ylabel('Predator')
plt.title('Phase Diagram of Prey vs. Predators')
plt.show()
