import numpy as np
# import cvxpy as cp
import matplotlib.pyplot as plt

# Parameters
L = 2.0  # Wheelbase of the car
T = 10.0  # Total simulation time (seconds)
dt = 0.01  # Time step (seconds)

# Obstacle parameters
x_c, y_c = 5.0, 5.0  # Obstacle center
r = 1.0  # Obstacle radius

# Initial state [x, y, theta]
state = np.array([0.0, 0.0, 0.0])

# Desired trajectory (nominal control inputs)
v_nominal = 2.0  # Constant velocity (m/s)
theta_nominal = 0.0  # Desired heading angle

# CBF parameters
alpha = 1.0  # Alpha for CBF

# Kinematic model
def kinematic_model(state, v, delta, dt):
    x, y, theta = state
    x_dot = v * np.cos(theta)
    y_dot = v * np.sin(theta)
    theta_dot = v / L * np.tan(delta)

    x += x_dot * dt
    y += y_dot * dt
    theta += theta_dot * dt

    return np.array([x, y, theta])

# Barrier function
def barrier_function(x, y):
    return (x - x_c)**2 + (y - y_c)**2 - r**2

# Barrier function derivative
def barrier_derivative(x, y, v, theta):
    return 2 * (x - x_c) * v * np.cos(theta) + 2 * (y - y_c) * v * np.sin(theta)

# Control with CBF
def control_with_cbf(state, v_nominal, delta_nominal):
    x, y, theta = state

    # Define control variables
    v = cp.Variable()
    delta = cp.Variable()

    # Define nominal control input
    u_nominal = np.array([v_nominal, delta_nominal])

    # Define the barrier function constraint
    B = barrier_function(x, y)
    B_dot = barrier_derivative(x, y, v_nominal, theta)
    constraint = B_dot + alpha * B >= 0

    # Objective: minimize (v - v_nominal)^2 + (delta - delta_nominal)^2
    objective = cp.Minimize((v - v_nominal)**2 + (delta - delta_nominal)**2)

    # Optimization problem
    prob = cp.Problem(objective, [constraint])

    # Solve the problem
    prob.solve()

    return v.value, delta.value

# Simulate the vehicle
trajectory = []
for t in np.arange(0, T, dt):
    trajectory.append(state)

    # Compute control with CBF
    v, delta = control_with_cbf(state, v_nominal, theta_nominal)

    # Update state using kinematic model
    state = kinematic_model(state, v, delta, dt)

trajectory = np.array(trajectory)

# Plot the results
plt.figure()
plt.plot(trajectory[:, 0], trajectory[:, 1], label="Vehicle trajectory")
circle = plt.Circle((x_c, y_c), r, color='r', fill=False)
plt.gca().add_patch(circle)
plt.plot(x_c, y_c, 'ro', label="Obstacle center")
plt.title("Vehicle Trajectory with CBF for Obstacle Avoidance")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.axis("equal")
plt.show()
