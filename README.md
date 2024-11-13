# Writing the README content to a text file
readme_content = """
# Auto Driving Simulation Project

## Project Overview

This project involves designing a speed control and path-following system for a simulated autonomous vehicle. The main objective is to track a predefined GPS path at a target speed of 5 mph, based on provided GPS data. The simulation is implemented in Python and includes vehicle dynamics models, sensor noise filtering, and path planning, along with real-time visualization.

### Key Objectives
1. **Vehicle Speed Control:** Ensure the vehicle maintains the target speed using a PID controller.
2. **Path-Following Control:** Develop and simulate a control system to follow the given GPS path accurately.
3. **Verification in Simulation:** Implement and verify the control models within a Python-based simulation.

## Problem Statement

The system must:
- Track a path defined by GPS data with a target speed of 5 mph.
- Incorporate a dynamic vehicle model to account for factors like inertia and tire slip.
- Handle sensor noise using filtering techniques to maintain trajectory accuracy.

## Data Description

The project relies on GPS data contained in the file `Loyd_nobel_nav_rosbag2_2024_11_11-11_24_51.csv`. This file provides time-stamped coordinates, vehicle speed, and brake status. Additionally, latitude and longitude tables are used to convert GPS data into local X-Y coordinates.

### Key Data Columns
- **GPS Data:** Includes `Latitude`, `Longitude`, `Speed[mps]`, and `Brake_status`.
- **Latitude and Longitude Conversion Tables:** `1deg`, `1'`, and `1''` conversions for latitude and longitude, in various units (miles, feet, meters).

## Code Overview

### 1. Vehicle Dynamics and Control
- **Stanley Controller:** Utilizes the Stanley method to adjust steering angle and minimize trajectory error.
- **PID Controller with Anti-Windup:** Controls the vehicle's speed, using proportional, integral, and derivative gains to maintain stability and prevent overshooting.
- **Dynamic Bicycle Model:** Models the vehicle's dynamics considering its speed, position, and orientation.

### 2. Kalman Filter
The Kalman Filter smooths GPS data by predicting and correcting the vehicle's position, reducing noise and enhancing path accuracy.

### 3. GPS to Local X-Y Conversion
To facilitate path tracking, GPS coordinates are converted to local X-Y coordinates using interpolation functions based on latitude and longitude conversion tables.

### 4. Path Planning with Cubic Spline
The path is generated using cubic spline interpolation for smoother and more accurate path tracking.

## Dependencies

To run this project, the following Python libraries are required:

- `numpy`
- `matplotlib`
- `pandas`
- `scipy`
- `numba`

Install these packages using:
```bash
pip install numpy matplotlib pandas scipy numba
