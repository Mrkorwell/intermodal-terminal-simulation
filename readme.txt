# Final Development Package for Intermodal Terminal Simulation

## Overview:

This package contains the simulation model code that analyzes the operations of an intermodal terminal under various resource and truck-train synchronization scenarios. It utilizes different combinations of input files to examine the effect of road gates, cranes, and truck arrival patterns on the terminal's performance.

## Simulation Script:

- **sim.py**: The core Python simulation script, responsible for modeling the terminal operations. It tests multiple scenarios with varying resources (number of cranes and road gates) and different truck and train synchronization patterns.

## Requirements:

### Python Dependencies:
To run the simulation, you need the following Python packages:
- `salabim`, and `simpy`: For discrete event simulation and animation.
- `pandas`: For handling and manipulating the CSV data files.
- `numpy`: For numerical operations, especially for random distributions (Poisson, Gaussian).
- `matplotlib`: For generating visualizations of the results.

### Install the dependencies using:

pip install simpy salabim pandas numpy matplotlib


## Running the Simulation:

To run the simulation with different scenarios, follow these steps:

1. Place the input files (ITU, truck, and train data) in the same directory as the sim.py script.
2. Open a terminal or command prompt in the folder where `sim.py` is located.
3. Run the script using:


## Input Files and Scenario Combinations:

The input files correspond to the combinations of truck, train, and ITU data used in different simulation scenarios. Each scenario tests a different synchronization pattern and resource allocation configuration.

### Scenario 1:
- **Truck Input**: truck1.csv
- **Train Input**: train1.csv
- **ITU Input**: ITU2.csv
- **Description**: Baseline scenario with no synchronization between truck and train schedules.

### Scenario 2:
- **Truck Input**: truck2.csv
- **Train Input**: train2.csv
- **ITU Input**: ITU3.csv
- **Description**: A modified scenario with randomness introduced to truck arrival times.

### Scenario 3:
- **Truck Input**: truck3_poisson.csv
- **Train Input**: train1.csv
- **ITU Input**: ITU2.csv
- **Description**: Poisson-distributed truck arrivals to simulate stochastic patterns.

### Scenario 4:
- **Truck Input**: truck3_gaussian.csv
- **Train Input**: train1.csv
- **ITU Input**: ITU2.csv
- **Description**: Gaussian-distributed truck arrivals to test more variability in arrival patterns.

### Scenario 5 (High Demand):
- **Truck Input**: truck4.csv
- **Train Input**: train3.csv
- **ITU Input**: ITU2.csv
- **Description**: High-demand scenario simulating increased load on terminal resources.

## Additional Scenario Combinations:
You can modify the number of cranes and road gates in the `sim.py` file for each scenario to analyze their effects on performance metrics such as truck waiting time, crane utilization, and storage yard utilization.
