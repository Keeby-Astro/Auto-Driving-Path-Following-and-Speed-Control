# Auto-Driving-Path-Following-and-Speed-Control

## Project Overview

This project involves designing a vehicle speed control and path-following control system to track a GPS-defined path for a simulated autonomous vehicle. The main objective is to track a predefined GPS path at a target speed of 5 mph, based on provided GPS data. The simulation is implemented in Python and includes enhanced vehicle dynamics models, sensor noise filtering using an Extended Kalman Filter, and path planning with cubic spline interpolation, along with real-time visualization.

### Key Objectives
1. Vehicle Speed Control: Ensure the vehicle maintains the target speed using an enhanced PID controller with anti-windup, low-pass filtering, and smoother deceleration.
2. Path-Following Control: Develop and simulate a control system utilizing the Stanley method to follow the given GPS path accurately.
3. Verification in Simulation: Implement and verify the control models within a Python-based simulation environment.

## Problem Statement

The system must:
- Track a path defined by GPS data with a target speed of 10 mph.
- Incorporate a dynamic vehicle model with real-world parameters (e.g., 2014 Nissan Leaf) to account for factors like inertia, tire slip, aerodynamic drag, and rolling resistance.
- Handle sensor noise using filtering techniques, specifically an Extended Kalman Filter (EKF), to maintain trajectory accuracy.

## Data Description

The project relies on GPS data contained in the file `Loyd_nobel_nav_rosbag2_2024_11_11-11_24_51.csv`. This file provides time-stamped coordinates, vehicle speed, and brake status. Additionally, latitude and longitude tables are used to convert GPS data into local X-Y coordinates.

## Key Data Files

- GPS Data File:
  - Loyd_nobel_nav_rosbag2_2024_11_11-11_24_51.csv

- Latitude Conversion Table:
  - latitude_distance_to_latitude.csv

- Longitude Conversion Table:
  - longitude_distance_to_longitude.csv

### Key Data Columns
- **GPS Data:** Includes `Latitude`, `Longitude`, `Speed[mps]`, and `Brake_status`.
- **Latitude and Longitude Conversion Tables:** `1deg`, `1'`, and `1''` conversions for latitude and longitude, in various units (miles, feet, meters).

## Code Overview

### 1. Vehicle Dynamics and Control
- **Stanley Controller:** Utilizes the Stanley method to adjust steering angle and minimize trajectory error.
- **PID Controller with Anti-Windup, Low-Pass Filtering, and Smoother Deceleration:** Controls the vehicle's speed using proportional, integral, and derivative gains. Implements anti-windup to prevent integrator windup during saturation and includes low-pass filtering for smoother acceleration and deceleration.
- **Enhanced Dynamic Bicycle Model:** Incorporates real-world parameters of a 2014 Nissan Leaf. Accounts for aerodynamic drag, rolling resistance, and realistic steering limits.

### 2. Extended Kalman Filter (EKF)
The Extended Kalman Filter smooths GPS data by predicting and correcting the vehicle's position through the uses of a state transition model and observation model, reducing noise and enhancing path accuracy.

### 3. GPS to Local X-Y Conversion
To facilitate path tracking, GPS coordinates are converted to local X-Y coordinates using interpolation functions based on latitude and longitude conversion tables.

### 4. Path Planning with Cubic Spline
The path is generated using cubic spline interpolation for smoother and more accurate path tracking.

### 5. Hypothetical Path Planner (Optional)
Demonstrates the structure of a path planning module, includes path evaluation considering distance, obstacles, and smoothness.

## Dependencies

To run this project, the following Python libraries are required:

- `numpy`
- `matplotlib`
- `pandas`
- `scipy`
- `numba`
- `cubic_spline_planner`

Install these packages using:
```bash
pip install numpy matplotlib pandas scipy numba
```

## Running the Simulation

1. Place `latitude_distance_to_latitude.csv`, `longitude_distance_to_longitude.csv`, `Reaves_Park_acc_group_3_2024_11_22-15_49_50.csv`, and `Loyd_nobel_nav_rosbag2_2024_11_11-11_24_51.csv` in the same directory as the script.
2. Run the script:
    ```bash
    python auto_driving_final_project_controller.py
    ```
3. **Animation Mode:** If `show_animation` is set to `True`, a live plot of the vehicle’s path tracking and speed profile will be displayed.

4. **Save Mode:** If `save_animation` is set to `True`, the vehicle’s path tracking and speed profile will be saved in a GIF file format.

### Output Visualization
- **Path Tracking Animation:** Displays the GPS path with start and end markers, and the vehicle's trajectory as it follows the path.

![Car_Data/Simulation - Lloyd Noble.gif](https://github.com/Keeby-Astro/Auto-Driving-Path-Following-and-Speed-Control/blob/main/Car_Data/Simulation%20-%20Lloyd%20Noble.gif)

![Car_Data/Simulation - Reaves Park.gif](https://github.com/Keeby-Astro/Auto-Driving-Path-Following-and-Speed-Control/blob/main/Car_Data/Simulation%20-%20Reaves%20Park.gif)

- **Lloyd Noble Plots:**
![Results - Lloyd Noble.png](https://github.com/Keeby-Astro/Auto-Driving-Path-Following-and-Speed-Control/blob/main/Car_Data/Results%20-%20Lloyd%20Noble.png)

- **Reaves Park Plots:**
![Results - Reaves Park.png](https://github.com/Keeby-Astro/Auto-Driving-Path-Following-and-Speed-Control/blob/main/Car_Data/Results%20-%20Reaves%20Park.png)

## Conclusion

This project provides a comprehensive simulation of autonomous vehicle path tracking and speed control using Python. By integrating advanced control algorithms and data filtering techniques, it achieves a realistic and robust performance suitable for educational and research purposes.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.
