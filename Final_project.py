#!/usr/bin/env python3
'''
The following code runs and creates plots for Final Project: Sea Level Rise
Shown below is a list of function and a description of what they do.

simulater_sea_level(): Performs sea level rise simulation

question_1(): Produces verification plot that helps answer question 1

question_2(): Produces plot to help answer question 2

question_3(): Produces plot for question 3

question_4(): Produces plot for question 4
'''
# Import Libraries
import numpy as np
import matplotlib.pyplot as plt

def simulate_sea_level(time_steps=100000, dt=0.1, alpha=0.4, tau=1000, 
                       T_growth_rate=0.02, max_temp=50, initial_temp=0.1, 
                       initial_sea_level=0):
    """
    Simulates sea level rise using a Lotka-Volterra-inspired system.

    Parameters:
    - time_steps (int): defaults to 100000
        Number of time steps to simulate.
    - dt (float): defaults to 0.1
        Time step size in years.
    - alpha (float): defaults to 0.4
        Sensitivity of sea level to temperature (m/°C).
    - tau (float): defaults to 1000
        Time constant in years.
    - T_growth_rate (float):  defaults to 0.02
        Growth rate of temperature anomaly.
    - max_temp (float): defaults to 50
        Maximum temperature anomaly (°C).
    - initial_temp (float): defaults to 0.1
        Initial temperature anomaly (°C).
    - initial_sea_level (float): defaults to 0
        Initial sea level (m).

    Returns:
    - time (numpy array): 
        Array of time values.
    - S (numpy array): 
        Simulated sea level rise over time.
    - delta_T (numpy array): 
        Simulated temperature anomaly over time.
    """
    # Time array
    time = np.linspace(0, time_steps * dt, time_steps)
    
    # Initialize arrays for sea level and temperature anomaly
    S = np.zeros(time_steps)
    delta_T = np.zeros(time_steps)
    
    # Set initial conditions
    delta_T[0] = initial_temp
    S[0] = initial_sea_level
    
    # Define rate of change functions
    def dS_dt(S, delta_T):
        S_eq_te = alpha * delta_T
        return (S_eq_te - S) / tau

    def dDeltaT_dt(delta_T):
        return T_growth_rate * delta_T * (1 - delta_T / max_temp)

    # Numerical integration (Lotka-Volterra-inspired coupling)
    for i in range(1, time_steps):
        dS = dS_dt(S[i-1], delta_T[i-1]) * dt
        dT = dDeltaT_dt(delta_T[i-1]) * dt

        S[i] = S[i-1] + dS
        delta_T[i] = delta_T[i-1] + dT


    return time, S, delta_T

def question_1():
    '''
    Creates plot for question 1.

    '''
    # Define your variables using function
    time, S, delta_T = simulate_sea_level()
    # Plotting
    plt.figure(figsize=(10, 6))
    # Plot sea level rise
    plt.plot(time, S, label="Sea Level Rise (S(t))", color="blue", lw=2)

    # Set labels and legend
    plt.xlabel("Time (years)", fontsize=12)
    plt.ylabel("Sea Level (m)", fontsize=12)
    plt.title("Sea Level Rise Model ", fontsize=14)
    plt.xlim(0,1001)
    plt.legend(fontsize=12)
    plt.grid()

    # Show the plot
    plt.show()

def question_2():
    '''
    # Create plots for question 2.
    '''
    # Create variables with different alpha values
    time, S, delta_T = simulate_sea_level(alpha=0.2)
    time1, S1, delta_T1 = simulate_sea_level(alpha= 0.35)
    time2, S2, delta_T2 = simulate_sea_level(alpha = 0.5 )
    time3, S3, delta_T3 = simulate_sea_level(alpha = 0.63)
    # Create variables with different tau valeus
    time4, S4, delta_T4 = simulate_sea_level(tau = 82)
    time5, S5, delta_T5 = simulate_sea_level(tau = 400)
    time6, S6, delta_T6 = simulate_sea_level(tau = 800)
    time7, S7, delta_T7 = simulate_sea_level(tau = 1290)

    # Create figure and plot!
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(14, 6))
    # Plot Values
    ax1.plot(time, S, label=r"Alpha = 0.2 m/°C")
    ax1.plot(time1, S1, label=r"Alpha = 0.35 m/°C")
    ax1.plot(time2, S2, label=r"Alpha = 0.5 m/°C")    
    ax1.plot(time3, S3, label=r"Alpha = 0.63 m/°C")
    # Create limit on x-axis
    ax1.set_xlim(0,1001)
    # Label title and axes
    ax1.set_title("Differences in Commitment Factor for Sea Level Model")
    ax1.set_xlabel("Years")
    ax1.set_ylabel("Sea Level (m)")
    # Create legend
    ax1.legend(loc ='best')

    # Plot values
    ax2.plot(time4, S4, label='τ = 82 years')
    ax2.plot(time5, S5, label="τ = 400 years")
    ax2.plot(time6, S6, label='τ= 800 years')
    ax2.plot(time7, S7, label='τ = 1290 years')
    # Create limit on x-axis
    ax2.set_xlim(0,1001)
    # Label title and axes
    ax2.set_title("Differences in Timescale for Sea Level Model")
    ax2.set_xlabel("Years")
    ax2.set_ylabel("Sea Level (m)")
    # Create legend
    ax2.legend(loc ='best')
    # Show Plot
    plt.show()

def question_3():
    '''
    Creates plot for third question.
    '''
    # Create variables with different temperature growth rates.
    time, S, delta_T = simulate_sea_level()
    time2, S2, delta_T2 = simulate_sea_level(T_growth_rate = 0.05 )
    time3, S3, delta_T3 = simulate_sea_level(T_growth_rate = 0.1)
    time4, S4, delta_T4 = simulate_sea_level(T_growth_rate = 0.5)
    
    # Plot figure
    plt.figure(figsize=(10, 6))
    # Plot sea level rise
    plt.plot(time, S, label="T anomaly = 0.02 °C/year")
    plt.plot(time2, S2, label="T anomaly = 0.05 °C/year")
    plt.plot(time3, S3, label="T anomaly = 0.1 °C/year")
    plt.plot(time4, S4, label="T anomaly = 0.5 °C/year")

    # Set labels and legend
    plt.xlabel("Time (years)", fontsize=12)
    plt.ylabel("Sea Level (m)", fontsize=12)
    plt.title("Sea Level Rise Model with Different Temp Anomalies ", fontsize=14)
    plt.xlim(0,1001)
    plt.legend(fontsize=12)
    plt.grid()
    # Show plot
    plt.show()




def question_4():   
    '''
    Creates plots for question 4.
    '''
    # Create figure and plot!
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(13, 6))
    time, S, delta_T = simulate_sea_level()

    # Mark Miami's elevation and plot values
    ax1.axhline(8.6, color="red", linestyle="--", label="Miami Elevation (8.6 m)")
    ax1.plot(time, S, label="Sea Level Rise")
    # Set labels and legend
    ax1.set_xlabel("Time (years)", fontsize=12)
    ax1.set_ylabel("Sea Level (m)", fontsize=12)
    ax1.set_title("Simulated Sea Level Rise Until Miami is Submerged", fontsize=14)
    ax1.legend(fontsize=12)
    # Create x-limit and grid
    ax1.set_xlim(0,1001)
    ax1.grid()
    # Mark Floridaexi's elevation and plot values
    ax2.axhline(105, color="red", linestyle="--", label="Florida's Highest Elevation (105 m)") 
    ax2.plot(time, S, label="Sea Level Rise")   
    # Set labels and legend
    ax2.set_xlabel("Time (years)", fontsize=12)
    ax2.set_ylabel("Sea Level (m)", fontsize=12)
    ax2.set_title("Simulated Sea Level Rise Until Florida is Submerged", fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid()
    # Show the plot
    plt.show()

