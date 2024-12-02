# Auto Driving Final Project
# Names: Caden Matthews, Jacob Ferrell, James Ouellette, Lin Tan, and David Yun

'''
--Problem Statement:
Design a vehicle speed control and path following control to track the GPS path below
(Target speed is 5 [mph]; Original data is attached).
Develop the controls in simulation using Python for verification purposes.

--Original Data is attached:
Loyd_nobel_nav_rosbag2_2024_11_11-11_24_51.csv

--Latitude Column Names:
Latitude - deg
1deg of Latitude (Miles), 1deg of Latitude (Feet), 1deg of Latitude (Metres)
1' of Latitude (Miles), 1' of Latitude (Feet), 1' of Latitude (Metres)
1'' of Latitude (Miles), 1'' of Latitude (Feet), 1'' of Latitude (Metres)

--Longitude Column Names:
Longitude - deg
1deg of Longitude (Miles), 1deg of Longitude (Feet), 1deg of Longitude (Metres)
1' of Longitude (Miles), 1' of Longitude (Feet), 1' of Longitude (Metres)
1'' of Longitude (Miles), 1'' of Longitude (Feet), 1'' of Longitude (Metres)

--Test Data Column Names:
Time, Longitude, Latitude, Speed[mps], Brake_status
'''

# Importing Required Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd
from scipy.interpolate import interp1d
from scipy.constants import mile, hour
from scipy.signal import savgol_filter
from numba import njit, float64
from numba.experimental import jitclass
import concurrent.futures
import cubic_spline_planner

# Gain Parameters
k  = 0.22 # Stanley control gain
Kp = 0.8   # PID control gain: P
Ki = 0.005 # PID control gain: I
Kd = 0.6   # PID control gain: D

# Constants
dt     = 0.1   # Time step
g      = 9.81  # Gravity constant (m/s^2)
rho    = 1.293 # Air density (kg/m^3)
f_roll = 0.015 # Rolling resistance coefficient
F_xr   = 217.3 # Rear tire longitudinal force
beta   = 0.0   # Road slope angle (radians) [Assumed to be flat]

# Real World Vehicle Parameters
L         = 2.7                  # Wheelbase of 2014 Nissan Leaf (m)
C_d       = 0.28                 # Drag coefficient of 2014 Nissan Leaf
A         = 2.303995             # Frontal area of 2014 Nissan Leaf (m^2)
m         = 1477                 # Vehicle mass of 2014 Nissan Leaf (kg)
I         = 1098.603             # Moment of inertia of 2014 Nissan Leaf (kg*m^2)
L_veh     = 4.445                # Length of 2014 Nissan Leaf (m)
L_f       = 1.188                # Distance from the center of mass to the front axle (m)
L_r       = 1.512                # Distance from the center of mass to the rear axle (m)
C_af      = 0.12                 # Cornering Stiffness Coefficient of front bias tires
C_ar      = 0.12                 # Cornering Stiffness Coefficient of rear bias tires
battery   = 24                   # Battery capacity of 2014 Nissan Leaf (kWh)
max_steer = np.radians(31.30412) # Maximum steering angle of 2014 Nissan Leaf (radians)

# Acceleration Output Limit for Anti-Windup
output_limit = 3.8873 # (m/s^2)

# Set to True to show the Animation and save the Animation
show_animation = True
save_animation = False

# Target Speed in mph to m/s
TARGET_SPEED_MPH = 10 # (mph)
TARGET_SPEED_MPS = TARGET_SPEED_MPH * mile / hour # (m/s)

# Define the State Specification
state_spec = [
    ('x', float64),
    ('y', float64),
    ('yaw', float64),
    ('v', float64)
]

# Define the State Class for the Vehicle
@jitclass(state_spec)
class State:
    '''
    Class representing the state of a vehicle.

    :param x: (float) x-coordinate
    :param y: (float) y-coordinate
    :param yaw: (float) yaw angle
    :param v: (float) speed

    :return: (State) State of the vehicle
    '''
    # Initialize the state of the vehicle
    def __init__(self, x, y, yaw, v):
        self.x   = x   # x-coordinate
        self.y   = y   # y-coordinate
        self.yaw = yaw # yaw angle
        self.v   = v   # velocity

    # Update the state of the vehicle with enhanced dynamics
    def update(self, acceleration, delta):
        '''
        Update the state of the vehicle with enhanced dynamics using real-world parameters.

        :param acceleration: (float) Acceleration command (m/s^2)
        :param delta: (float) Steering angle (radians)

        :return: None
        '''
        # Limit the steering angle to the maximum steering angle
        delta = min(max(delta, -max_steer), max_steer)

        # Compute resistance forces
        F_drag = 0.5 * rho * C_d * A * self.v ** 2 # Aerodynamic drag force
        F_rolling = m * g * f_roll                 # Rolling resistance force
        F_resistance = F_drag + F_rolling          # Total resistance force

        # Compute net acceleration
        a_net = acceleration - F_resistance / m

        # Update the vehicle's velocity
        self.v += a_net * dt

        # Update the vehicle's position
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt

        # Update the vehicle's yaw angle using bicycle model
        self.yaw += (self.v / L) * np.tan(delta) * dt
        self.yaw = normalize_angle(self.yaw)

        # Prevent velocity from becoming negative
        self.v = max(self.v, 0.0)

# Normalize the Angle
@njit(cache=True)
def normalize_angle(angle):
    '''
    Normalize the angle to the range [-pi, pi]

    :param angle: (float) Angle in radians

    :return: (float) Normalized angle
    '''
    return (angle + np.pi) % (2 * np.pi) - np.pi

# PID Controller
@njit(cache=True)
def pid_control(target, current, prev_error, integral, is_decelerating):
    '''
    PID controller with Anti-Windup, low-pass filtering, and smoother deceleration

    :param target: (float) Target value
    :param current: (float) Current value
    :param prev_error: (float) Previous error
    :param integral: (float) Integral term
    :param is_decelerating: (bool) Flag for deceleration

    :return: (float) Output value, Error, Integral term
    '''
    # PID constants (tuned for smoother response)
    Kp_mod = Kp * (0.8 if is_decelerating else 1.0)  # Reduce proportional gain during deceleration
    Kd_mod = Kd * (0.24 if is_decelerating else 1.0) # Reduce derivative gain during deceleration

    # Error calculation
    error = target - current
    integral += error * dt

    # Anti-windup: Clamp the integral term to prevent overshooting
    integral = min(max(integral, -output_limit / Ki), output_limit / Ki)

    # Compute PID terms
    p_term = Kp_mod * error
    d_term = Kd_mod * (error - prev_error) / dt
    i_term = Ki * integral

    # Calculate raw output
    raw_output = p_term + i_term + d_term

    # Low-pass filter on the output (for smoother acceleration/deceleration)
    alpha = 0.14  # Smoothing factor (0 < alpha < 1, higher is less smoothing)
    smoothed_output = alpha * raw_output + (1 - alpha) * prev_error

    # Apply anti-windup on the smoothed output
    if smoothed_output > output_limit:
        smoothed_output = output_limit
        integral -= error * dt
    elif smoothed_output < -output_limit:
        smoothed_output = -output_limit
        integral -= error * dt

    return smoothed_output, error, integral

# Stanley Control
@njit(cache=True)
def stanley_control(state, cx, cy, cyaw, last_target_idx):
    '''
    Stanley steering control.

    :param state: (State object)
    :param cx: (np.ndarray)
    :param cy: (np.ndarray)
    :param cyaw: (np.ndarray)
    :param last_target_idx: (int)

    :return: (float, int)
    '''
    epsilon = 1e-5
    current_target_idx, error_front_axle = calc_target_index(state, cx, cy, last_target_idx)
    if current_target_idx < last_target_idx:
        current_target_idx = last_target_idx
    theta_e = normalize_angle(cyaw[current_target_idx] - state.yaw) # Heading error
    theta_d = np.arctan2(k * error_front_axle, state.v + epsilon)   # Cross track error
    delta = theta_e + theta_d
    delta *= 0.8 # Steering gain factor

    return delta, current_target_idx

# Calculate the Target Index
@njit(cache=True)
def calc_target_index(state, cx, cy, last_target_idx):
    '''
    Calculate the target index for the vehicle to follow the path.

    :param state: (State) State of the vehicle
    :param cx: (np.ndarray) x-coordinates of the path
    :param cy: (np.ndarray) y-coordinates of the path
    :param last_target_idx: (int) Last target index

    :return: (int, float) Target index, Error at the front axle
    '''
    fx = state.x + L * np.cos(state.yaw)
    fy = state.y + L * np.sin(state.yaw)

    dx = cx[last_target_idx:] - fx
    dy = cy[last_target_idx:] - fy
    d = np.hypot(dx, dy)
    target_idx_rel = np.argmin(d)
    target_idx = target_idx_rel + last_target_idx

    look_ahead_distance = k * state.v + 0.05

    while True:
        if target_idx + 1 >= len(cx):
            break
        dx = cx[target_idx + 1] - fx
        dy = cy[target_idx + 1] - fy
        dist = np.hypot(dx, dy)
        if dist > look_ahead_distance:
            break
        target_idx += 1

    front_axle_vec = np.array([-np.cos(state.yaw + np.pi / 2), -np.sin(state.yaw + np.pi / 2)])
    error_front_axle = np.dot(np.array([fx - cx[target_idx], fy - cy[target_idx]]), front_axle_vec)

    return target_idx, error_front_axle

# Function to convert Latitude and Longitude to local x and y
def convert_to_local_x_y(test_data, lat_table, lon_table):
    '''
    Convert latitude and longitude to local x and y coordinates.

    :param test_data: (pd.DataFrame) Test data
    :param lat_table: (pd.DataFrame) Latitude distance to latitude table
    :param lon_table: (pd.DataFrame) Longitude distance to longitude table

    :return: (np.array, np.array) Local x and y coordinates
    '''
    lat_interp = interp1d(lat_table['Latitude - deg'], lat_table['1deg of Latitude (Metres)'], fill_value="extrapolate")
    latitude_meters_per_degree = lat_interp(test_data['Latitude'].values)
    latitude_meters = (test_data['Latitude'].values - test_data['Latitude'].values[0]) * latitude_meters_per_degree

    lon_interp = interp1d(lon_table['Latitude - deg'], lon_table['1deg of Longitude (Metres)'], fill_value="extrapolate")
    longitude_meters_per_degree = lon_interp(test_data['Latitude'].values)
    longitude_meters = (test_data['Longitude'].values - test_data['Longitude'].values[0]) * longitude_meters_per_degree

    return longitude_meters, latitude_meters

# Extended Kalman Filter for GPS Data
@njit(cache=True)
def ekf_predict(x_est, P_est, F, Q):
    '''
    EKF Prediction step.

    :param x_est: (np.array) State estimate
    :param P_est: (np.array) Estimate covariance
    :param F: (np.array) State transition matrix
    :param Q: (np.array) Process noise covariance

    :return: (np.array, np.array) Predicted state estimate, Predicted estimate covariance
    '''
    x_pred = F @ x_est           # State prediction
    P_pred = F @ P_est @ F.T + Q # Covariance prediction
    return x_pred, P_pred

@njit(cache=True)
def ekf_update(x_pred, P_pred, z, H, R):
    '''
    EKF Update step.

    :param x_pred: (np.array) Predicted state estimate
    :param P_pred: (np.array) Predicted estimate covariance
    :param z: (np.array) Measurement
    :param H: (np.array) Observation matrix
    :param R: (np.array) Measurement noise covariance

    :return: (np.array, np.array) Updated state estimate, Updated estimate covariance
    '''
    y = z - H @ x_pred                        # Measurement residual
    S = H @ P_pred @ H.T + R                  # Residual covariance
    K = P_pred @ H.T @ np.linalg.inv(S)       # Kalman gain
    x_est = x_pred + K @ y                    # State update
    P_est = (np.eye(len(K)) - K @ H) @ P_pred # Covariance update
    return x_est, P_est

@njit(cache=True)
def run_ekf(local_x, local_y, F, H, Q, R, x_est, P_est):
    '''
    Run the Extended Kalman Filter over the data.

    :param local_x: (np.array) Local X positions
    :param local_y: (np.array) Local Y positions
    :param F: (np.array) State transition matrix
    :param H: (np.array) Observation matrix
    :param Q: (np.array) Process noise covariance
    :param R: (np.array) Measurement noise covariance
    :param x_est: (np.array) Initial state estimate
    :param P_est: (np.array) Initial estimate covariance

    :return: (np.array) Filtered data
    '''
    n = len(local_x)
    filtered_data = np.zeros((n, 2))
    for i in range(n):
        z = np.array([local_x[i], local_y[i]])
        if i > 0:
            x_pred, P_pred = ekf_predict(x_est, P_est, F, Q)
        else:
            x_pred, P_pred = x_est, P_est
        x_est, P_est = ekf_update(x_pred, P_pred, z, H, R)
        filtered_data[i, :] = x_est[:2]
    return filtered_data

# Path Planner class (Placeholder for future implementation)
class PathPlanner:
    def __init__(self, start, goal, obstacles):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles

    def evaluate_path(self, path):
        '''
        Calculate the cost of a path considering distance, obstacles, and smoothness
        :param path: ([(float, float)]) List of waypoints
        :return: (float) Total cost of the path
        '''
        # Placeholder implementation
        return 0.0

    def generate_candidate_paths(self, num_paths=10):
        '''
        Generate candidate paths (random paths for simplicity)

        :param num_paths: (int) Number of paths to generate

        :return: ([[(float, float)]]) List of paths
        '''
        # Placeholder implementation
        return []

def evaluate_paths_in_parallel(paths, path_planner):
    '''
    Evaluate the paths in parallel using the path planner.

    :param paths: ([[(float, float)]]) List of paths
    :param path_planner: (PathPlanner) Path planner object

    :return: ([float]) List of costs for each path
    '''
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(path_planner.evaluate_path, paths))
    return results

def update_vehicle_polygon(polygon, x, y, yaw):
    """
    Update the vehicle polygon to represent the current state of the vehicle.

    :param polygon: (matplotlib.patches.Polygon) Polygon object representing the vehicle
    :param x: (float) x-coordinate of the vehicle
    :param y: (float) y-coordinate of the vehicle
    :param yaw: (float) yaw angle of the vehicle
    """
    # Vehicle parameters
    W = 1.8   # Width of the vehicle (m)
    LF = 3.0  # Distance from center to front end (m)
    LB = 1.0  # Distance from center to back end (m)

    # Vehicle outline in the vehicle coordinate system
    outline = np.array([
        [LF,  W/2],
        [LF, -W/2],
        [-LB, -W/2],
        [-LB,  W/2],
    ])

    # Rotation matrix based on the yaw angle
    Rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw),  np.cos(yaw)]])
    # Rotate and translate the outline to the world coordinate system
    outline = (outline @ Rot.T) + np.array([x, y])

    # Update the polygon vertices
    polygon.set_xy(outline)

### MAIN FUNCTION ###
def main():
    latitude_distance_to_latitude = pd.read_csv('latitude_distance_to_latitude.csv')
    longitude_distance_to_longitude = pd.read_csv('longitude_distance_to_longitude.csv')
    # Ask the user to select the dataset
    dataset_choice = input("Type 1 for Lloyd Noble or type 2 for Reaves Park: ")

    if dataset_choice == '1':
        test_data_final = pd.read_csv('Loyd_nobel_nav_rosbag2_2024_11_11-11_24_51.csv')
        test_data_final = test_data_final.iloc[15:].reset_index(drop=True)
    elif dataset_choice == '2':
        test_data_final = pd.read_csv('Reaves_Park_acc_group_3_2024_11_22-15_49_50.csv')
        test_data_final = test_data_final.iloc[123:].reset_index(drop=True)
    else:
        raise ValueError("Invalid input. Please type 1 or 2.")
    test_data_final['Local_X'], test_data_final['Local_Y'] = convert_to_local_x_y(
        test_data_final, latitude_distance_to_latitude, longitude_distance_to_longitude
    )

    # Smooth the Local_X and Local_Y data
    smoothed_X = savgol_filter(test_data_final['Local_X'], window_length=11, polyorder=2)
    smoothed_Y = savgol_filter(test_data_final['Local_Y'], window_length=11, polyorder=2)

    # Estimate the measurement noise by subtracting the smoothed signal from the noisy data
    noise_X = test_data_final['Local_X'] - smoothed_X
    noise_Y = test_data_final['Local_Y'] - smoothed_Y

    # Calculate the standard deviation of the noise
    noise_std_X = 5 * np.std(noise_X)
    noise_std_Y = 5 * np.std(noise_Y)

    # Initialize EKF parameters
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float64) # State transition matrix
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=np.float64) # Observation matrix
    Q = np.diag([0.5, 0.5, 1.0, 1.0]) ** 2         # Process noise covariance
    R = np.diag([noise_std_X, noise_std_Y]) ** 2   # Measurement noise covariance
    x_est = np.zeros(4, dtype=np.float64)          # Initial state estimate
    P_est = np.eye(4, dtype=np.float64)            # Initial estimate covariance

    # Extract data into numpy arrays
    local_x = test_data_final['Local_X'].values.astype(np.float64)
    local_y = test_data_final['Local_Y'].values.astype(np.float64)

    # Apply EKF to the data
    filtered_data = run_ekf(local_x, local_y, F, H, Q, R, x_est, P_est)

    # Smooth the filtered data with Savitzky-Golay filter
    filtered_data[:, 0] = savgol_filter(filtered_data[:, 0], window_length=11, polyorder=2)
    filtered_data[:, 1] = savgol_filter(filtered_data[:, 1], window_length=11, polyorder=2)

    # Update the filtered data in the test data
    test_data_final['Filtered_X'] = filtered_data[:, 0]
    test_data_final['Filtered_Y'] = filtered_data[:, 1]

    # Plot the GPS trajectory
    if show_animation:
        plt.figure(figsize=(8, 8))
        size = 100
        plt.scatter(test_data_final['Local_X'], test_data_final['Local_Y'], label='GPS Trajectory', color='b', s=10)
        plt.plot(test_data_final['Filtered_X'], test_data_final['Filtered_Y'], label='Filtered Trajectory', color='r')
        plt.scatter(test_data_final['Local_X'].iloc[0], test_data_final['Local_Y'].iloc[0], marker='*',
                    color='g', s=size, label='Start')
        plt.scatter(test_data_final['Local_X'].iloc[-1], test_data_final['Local_Y'].iloc[-1], marker='<',
                    color='k', s=size, label='End')
        plt.xlabel('Local X (m)')
        plt.ylabel('Local Y (m)')
        plt.title('Local X and Y Trajectory')
        plt.axis('equal')
        plt.legend()
        plt.show()

    # Cubic Spline Path Planning
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        test_data_final['Filtered_X'].tolist(), test_data_final['Filtered_Y'].tolist(), ds=0.1)

    # Path Tracking
    target_speed = TARGET_SPEED_MPS
    max_simulation_time = test_data_final['Time'].iloc[-1] - test_data_final['Time'].iloc[0]

    # Initial state based on the first data row
    initial_x, initial_y = test_data_final['Filtered_X'].iloc[0], test_data_final['Filtered_Y'].iloc[0]
    dx_init = test_data_final['Filtered_X'].iloc[1] - initial_x
    dy_init = test_data_final['Filtered_Y'].iloc[1] - initial_y
    initial_yaw = np.arctan2(dy_init, dx_init)
    state = State(float(initial_x), float(initial_y), float(initial_yaw), 0.0)

    # Set cx, cy, cyaw as numpy arrays
    cx = np.array(cx, dtype=np.float64)
    cy = np.array(cy, dtype=np.float64)
    cyaw = np.array(cyaw, dtype=np.float64)

    # Simulation
    last_idx = len(cx) - 1
    n_steps = int(max_simulation_time / dt) + 1
    x_history = np.zeros(n_steps)
    y_history = np.zeros(n_steps)
    yaw_history = np.zeros(n_steps)
    v_history = np.zeros(n_steps)
    t_history = np.zeros(n_steps)
    target_idx_history = np.zeros(n_steps, dtype=int)
    time = 0.0
    x_history[0], y_history[0], yaw_history[0], v_history[0], t_history[0] = state.x, state.y, state.yaw, state.v, time

    # Ensure target_idx starts from the beginning
    target_idx = 0
    prev_error, integral = 0.0, 0.0

    # Simulation loop
    i = 0
    while target_idx < len(cx) - 1 and time <= max_simulation_time:
        is_decelerating = state.v > target_speed
        ai, new_error, new_integral = pid_control(target_speed, state.v, prev_error, integral, is_decelerating)
        prev_error, integral = new_error, new_integral
        di, target_idx = stanley_control(state, cx, cy, cyaw, target_idx)

        # Adjust target speed based on the turning angle
        angle_diff = abs(normalize_angle(cyaw[target_idx] - state.yaw))
        if angle_diff > np.radians(5):  # Slow down if the turning angle is greater than 5 degrees
            target_speed = max(TARGET_SPEED_MPS * 0.45, TARGET_SPEED_MPS * (1 - angle_diff / np.pi))
        else:
            target_speed = TARGET_SPEED_MPS

        # Update the state
        state.update(ai, di)
        time += dt

        # Store the state in the history
        if i < n_steps:
            x_history[i], y_history[i], yaw_history[i], v_history[i], t_history[i] = state.x, state.y, state.yaw, state.v, time
            target_idx_history[i] = target_idx
        else:
            break  # Prevent index out of bounds

        i += 1

    # Truncate the arrays
    x_history = x_history[:i]
    y_history = y_history[:i]
    yaw_history = yaw_history[:i]
    v_history = v_history[:i]
    t_history = t_history[:i]
    target_idx_history = target_idx_history[:i]

    # Print the simulation results
    print("Goal reached!" if target_idx >= len(cx) - 1 else "Goal not reached within the simulation time.")

    # Create the animation function
    def animate(frame):
        idx = frame % len(x_history)
        line_trajectory.set_data(x_history[:idx + 1], y_history[:idx + 1])
        point_target.set_data([cx[target_idx_history[idx]]], [cy[target_idx_history[idx]]])
        update_vehicle_polygon(vehicle_polygon, x_history[idx], y_history[idx], yaw_history[idx])
        ax.set_title(f"Speed [mph]: {v_history[idx] * (hour / mile):.2f}")
        return line_trajectory, point_target, vehicle_polygon

    if show_animation:
        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(8, 8))
        size = 100
        line_course, = ax.plot(cx, cy, "r", label="Course")
        line_trajectory, = ax.plot([], [], "-b", label="Trajectory")
        point_target, = ax.plot([], [], "xg", label="Target")
        ax.scatter(cx[0], cy[0], marker='*', color='g', s=size, label='Start')
        ax.scatter(cx[-1], cy[-1], marker='<', color='k', s=size, label='End')
        ax.axis("equal")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title("Path Tracking")
        vehicle_polygon = Polygon(np.zeros((4, 2)), closed=True, color='k', alpha=0.5)
        ax.add_patch(vehicle_polygon)
        ax.legend()
        plt.tight_layout()

        # Create the animation
        ani = FuncAnimation(
            fig, animate, frames=len(x_history), interval=dt * 1000, blit=False
        )

        if save_animation:
            # Save the animation as a GIF
            ani.save("path_tracking_animation.gif", writer=PillowWriter(fps=60))

        plt.show()

    # Plotting results
    if show_animation:
        plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(2, 3, width_ratios=[2, 1, 1])

        # Path Following plot
        ax0 = plt.subplot(gs[:, 0])  # Span both rows
        ax0.scatter(cx[0], cy[0], marker='*', color='g', s=size, label='Start')
        ax0.scatter(cx[-1], cy[-1], marker='<', color='k', s=size, label='End')
        ax0.plot(cx, cy, "r", label="Course")
        ax0.plot(x_history, y_history, "-b", label="Trajectory")
        ax0.set_xlabel("X [m]")
        ax0.set_ylabel("Y [m]")
        ax0.set_title("Path Tracking")
        ax0.set_facecolor('lightgray')
        ax0.axis("equal")
        ax0.legend()

        # X-Coordinate vs Time
        ax1 = plt.subplot(gs[0, 1])
        ax1.plot(t_history, x_history, "-b")
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("X [m]")
        ax1.set_title("X-Coordinate vs Time")

        # Y-Coordinate vs Time
        ax2 = plt.subplot(gs[0, 2])
        ax2.plot(t_history, y_history, "-g")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Y [m]")
        ax2.set_title("Y-Coordinate vs Time")

        # Yaw vs Time
        ax3 = plt.subplot(gs[1, 1])
        ax3.plot(t_history, np.degrees(yaw_history), "-k")
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("Yaw [deg]")
        ax3.set_title("Yaw vs Time")

        # Speed vs Time
        ax4 = plt.subplot(gs[1, 2])
        ax4.plot(t_history, v_history * (hour / mile), "-r")
        ax4.set_xlabel("Time [s]")
        ax4.set_ylabel("Speed [mph]")
        ax4.set_title("Speed vs Time")

        plt.tight_layout()
        if save_animation:
            plt.savefig("path_tracking_results1.png", dpi=600)
        plt.show()

        # Plot velocity, acceleration, and jerk profiles over time on a single plot within the same plot window, all in mph, mph/s, and mph/s^2
        plt.figure(figsize=(8, 8))
        plt.plot(t_history, v_history * (hour / mile), "-r", label="Speed [mph]")
        plt.plot(t_history, np.gradient(v_history, dt) * (hour / mile), "-b", label="Acceleration [mph/s]")
        plt.plot(t_history, np.gradient(np.gradient(v_history, dt), dt) * (hour / mile), "-g", label="Jerk [mph/s^2]")
        plt.xlabel("Time [s]")
        plt.ylabel("Value")
        plt.title("Velocity, Acceleration, and Jerk Profiles")
        plt.legend()
        plt.tight_layout()
        plt.show()
         
if __name__ == '__main__':
    main()
