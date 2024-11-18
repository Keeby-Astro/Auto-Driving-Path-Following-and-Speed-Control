# Auto Driving Final Project
# Names: Caden Matthews, Jacob Ferrell, James Ouellette, Lin Tan, and David Yun

'''
--Problem Statement:
Design a vehicle speed control and path following control to track the gps path below
(Target speed is 5 [mph]; Original data is attached).
Develop the controls in simulation using python for verification purpose.

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
import pandas as pd
from scipy.interpolate import interp1d
from scipy.constants import mile, kilo, hour
from scipy.signal import savgol_filter
from numba import njit, float64
from numba.experimental import jitclass
import cubic_spline_planner
import concurrent.futures
import matplotlib.gridspec as gridspec

# Gain Parameters
k  = 0.02  # Stanley control gain
Kp = 0.8   # PID control gain: P
Ki = 0.005 # PID control gain: I
Kd = 0.6   # PID control gain: D

# Constants
dt     = 0.1   # Time step
g      = 9.81  # Gravity constant (m/s^2)
rho    = 1.293 # Air density (kg/m^3)
F_xr   = 0     # Rolling resistance force (N)
f_roll = 0.015 # Rolling resistance coefficient
beta   = 0     # Road slope angle (radians) [Assumed to be 0]

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

# Output Limit for Anti-Windup
output_limit = 3.8873 # Acceleration limit of 2014 Nissan Leaf (m/s^2)

# Set to True to show the animation
show_animation = True

# Target speed in mph to m/s
TARGET_SPEED_MPH = 5 # (mph)
TARGET_SPEED_MPS = TARGET_SPEED_MPH * mile / hour # (m/s)

# Define the state specification
state_spec = [
    ('x', float64),
    ('y', float64),
    ('yaw', float64),
    ('v', float64)
]

# Define the state class for the vehicle
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
        F_drag = 0.5 * rho * C_d * A * self.v**2 # Aerodynamic drag force
        F_rolling = m * g * f_roll               # Rolling resistance force
        F_resistance = F_drag + F_rolling        # Total resistance force

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

# Function to normalize the angle
@njit(cache=True)
def normalize_angle(angle):
    '''
    Normalize the angle to the range [-pi, pi]

    :param angle: (float) Angle in radians

    :return: (float) Normalized angle
    '''
    return (angle + np.pi) % (2 * np.pi) - np.pi

# PID controller with Anti-Windup
@njit(cache=True)
def pid_control(target, current, prev_error=0.0, integral=0.0, is_decelerating=False):
    '''
    PID controller with Anti-Windup and smoother deceleration

    :param target: (float) Target value
    :param current: (float) Current value
    :param prev_error: (float) Previous error
    :param integral: (float) Integral term
    :param is_decelerating: (bool) Flag for deceleration

    :return: (float) Output value, Error, Integral term
    '''
    error = target - current
    p_term = Kp * error
    integral += error * dt

    # Smooth deceleration by reducing the effect of the derivative term
    if is_decelerating:
        integral = min(max(integral, -output_limit / Ki), output_limit / Ki)  # Clamp the integral term
        d_term = Kd * (error - prev_error) / (3 * dt)  # Reduce derivative effect by 1/3

        # Lower the proportional term to avoid overshooting
        if error < 0:
            p_term = 0.5 * p_term
        else:
            p_term = 1.5 * p_term
    else:
        d_term = Kd * (error - prev_error) / dt

    i_term = Ki * integral
    output = p_term + i_term + d_term

    # Anti-Windup
    if output > output_limit:
        output = output_limit
        integral -= error * dt
    elif output < -output_limit:
        output = -output_limit
        integral -= error * dt

    return output, error, integral

# Stanley control function
def stanley_control(state, cx, cy, cyaw, last_target_idx):
    '''
    Stanley steering control.

    :param state: (State object)
    :param cx: ([float])
    :param cy: ([float])
    :param cyaw: ([float])
    :param last_target_idx: (int)

    :return: (float, int)
    '''
    epsilon = 1e-5
    current_target_idx, error_front_axle = calc_target_index(state, cx, cy, last_target_idx)
    current_target_idx = max(last_target_idx, current_target_idx)
    theta_e = normalize_angle(cyaw[current_target_idx] - state.yaw) # Heading error
    theta_d = np.arctan2(k * error_front_axle, state.v + epsilon)   # Cross track error
    delta = theta_e + theta_d

    return delta, current_target_idx

# Function to calculate the target index
def calc_target_index(state, cx, cy, last_target_idx):
    '''
    Calculate the target index for the vehicle to follow the path.

    :param state: (State) State of the vehicle
    :param cx: ([float]) x-coordinates of the path
    :param cy: ([float]) y-coordinates of the path
    :param last_target_idx: (int) Last target index

    :return: (int, float) Target index, Error at the front axle
    '''
    fx, fy = state.x + L * np.cos(state.yaw), state.y + L * np.sin(state.yaw)
    dx, dy = cx[last_target_idx:] - fx, cy[last_target_idx:] - fy
    d = np.hypot(dx, dy)
    target_idx = np.argmin(d) + last_target_idx
    look_ahead_distance = k * state.v + 1.0

    # Check if the target index is out of bounds
    while target_idx + 1 < len(cx) and np.hypot(cx[target_idx + 1] - fx, cy[target_idx + 1] - fy) <= look_ahead_distance:
        target_idx += 1

    front_axle_vec = np.array([-np.cos(state.yaw + np.pi / 2), -np.sin(state.yaw + np.pi / 2)])
    error_front_axle = np.dot([fx - cx[target_idx], fy - cy[target_idx]], front_axle_vec)

    return target_idx, error_front_axle

# Function to convert latitude and longitude to local x and y
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

def main():
    latitude_distance_to_latitude = pd.read_csv('latitude_distance_to_latitude.csv')
    longitude_distance_to_longitude = pd.read_csv('longitude_distance_to_longitude.csv')
    test_data_final = pd.read_csv('Loyd_nobel_nav_rosbag2_2024_11_11-11_24_51.csv')
    test_data_final['Local_X'], test_data_final['Local_Y'] = convert_to_local_x_y(
        test_data_final, latitude_distance_to_latitude, longitude_distance_to_longitude
    )
    test_data_final = test_data_final.iloc[15:]

    # Smooth the Local_X and Local_Y data
    smoothed_X = savgol_filter(test_data_final['Local_X'], window_length=11, polyorder=2)
    smoothed_Y = savgol_filter(test_data_final['Local_Y'], window_length=11, polyorder=2)

    # Estimate the measurement noise by subtracting the smoothed signal from the noisy data
    noise_X = test_data_final['Local_X'] - smoothed_X
    noise_Y = test_data_final['Local_Y'] - smoothed_Y

    # Calculate the standard deviation of the noise
    noise_std_X = 5 * np.std(noise_X)
    noise_std_Y = 5 * np.std(noise_Y)

    # Extended Kalman Filter for GPS data
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

    # Initialize EKF parameters
    dt = 0.1  # Time step
    F = np.array([[1, 0, dt, 0], # State transition matrix
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])  
    H = np.array([[1, 0, 0, 0],  # Observation matrix
                  [0, 1, 0, 0]])  
    Q = np.diag([0.5, 0.5, 1.0, 1.0]) ** 2 # Process noise covariance
    R = np.diag([noise_std_X, noise_std_Y]) ** 2 # Measurement noise covariance
    x_est = np.zeros(4)  # Initial state estimate [x, y, v_x, v_y]
    P_est = np.eye(4)    # Initial estimate covariance

    # Apply EKF to the data
    filtered_data = []
    for i in range(len(test_data_final)):
        z = np.array([test_data_final['Local_X'].iloc[i], test_data_final['Local_Y'].iloc[i]])

        # If not the first measurement, perform prediction
        if i > 0:
            x_pred, P_pred = ekf_predict(x_est, P_est, F, Q)
        else:
            x_pred, P_pred = x_est, P_est

        # Perform update
        x_est, P_est = ekf_update(x_pred, P_pred, z, H, R)
        filtered_data.append(x_est[:2])

    # Convert the filtered data to a numpy array
    filtered_data = np.array(filtered_data)

    # Smooth the filtered data with Savitzky-Golay filter
    filtered_data[:, 0] = savgol_filter(filtered_data[:, 0], window_length=11, polyorder=2)
    filtered_data[:, 1] = savgol_filter(filtered_data[:, 1], window_length=11, polyorder=2)

    # Update the filtered data in the test data
    test_data_final['Filtered_X'] = filtered_data[:, 0]
    test_data_final['Filtered_Y'] = filtered_data[:, 1]

    # Plot the GPS trajectory
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

    # Set cx, cy, cyaw, and ck as arrays generated from cubic spline planner
    cx, cy, cyaw, ck = map(np.array, [cx, cy, cyaw, ck])

    # Simulation
    last_idx = len(cx) - 1
    time, n_steps = 0.0, int(max_simulation_time / dt) + 1
    x_history, y_history, yaw_history, v_history, t_history = [np.zeros(n_steps) for _ in range(5)]
    x_history[0], y_history[0], yaw_history[0], v_history[0], t_history[0] = state.x, state.y, state.yaw, state.v, time

    # Ensure target_idx starts from the beginning
    target_idx = 0
    prev_error, integral = 0.0, 0.0

    # Simulation loop
    if show_animation:
        # Initial setting for visualization
        plt.figure(figsize=(8, 8))
        line_course, = plt.plot(cx, cy, "r", label="Course")
        line_trajectory, = plt.plot([], [], "-b", label="Trajectory")
        point_target, = plt.plot([], [], "xg", label="Target")
        plt.scatter(cx[0], cy[0], marker='*', color='g', s=size, label='Start')
        plt.scatter(cx[-1], cy[-1], marker='<', color='k', s=size, label='End')
        plt.axis("equal")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title("Path Tracking")
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

    # Simulation loop with end condition for a single complete path loop
    i = 0
    while target_idx < len(cx) - 1:
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

        # Update loop variables
        state.update(ai, di) # Update the state of the vehicle
        time += dt           # Update the time
        i += 1               # Update the step count

        # Extend history arrays if needed
        if i >= len(x_history):
            x_history = np.append(x_history, np.zeros(100))
            y_history = np.append(y_history, np.zeros(100))
            yaw_history = np.append(yaw_history, np.zeros(100))
            v_history = np.append(v_history, np.zeros(100))
            t_history = np.append(t_history, np.zeros(100))

        # Store the state in the history
        x_history[i], y_history[i], yaw_history[i], v_history[i], t_history[i] = state.x, state.y, state.yaw, state.v, time

        # Update the visualization
        if show_animation:
            line_trajectory.set_data(x_history[:i+1], y_history[:i+1])
            point_target.set_data([cx[target_idx]], [cy[target_idx]])
            plt.title(f"Speed [km/h]: {state.v * 3.6:.2f}")
            plt.pause(0.001)

    # Truncate the arrays
    x_history = x_history[:i+1]
    y_history = y_history[:i+1]
    yaw_history = yaw_history[:i+1]
    v_history = v_history[:i+1]
    t_history = t_history[:i+1]

    # Print the simulation results
    print("Goal reached!" if target_idx >= len(cx) - 1 else "Goal not reached within the simulation time.")

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
        ax4.plot(t_history, v_history * kilo / hour, "-r")
        ax4.set_xlabel("Time [s]")
        ax4.set_ylabel("Speed [km/h]")
        ax4.set_title("Speed vs Time")

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
     main()
