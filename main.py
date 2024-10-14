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
def kinematic_model(state, v, delta, dt, L):
    x, y, theta = state # state = (x,y,theta)
    x_dot = v * np.cos(theta)
    y_dot = v * np.sin(theta)
    theta_dot = v / L * np.tan(delta)

    x += x_dot * dt
    y += y_dot * dt
    theta += theta_dot * dt

    return np.array([x, y, theta])

# Barrier function
def barrier_function(x):
    return (x[0])**2 + (x[1])**2 - r**2 #Create a Barrier function that x1^2 + x2^2 = r^2

def safe_region(x):
    return barrier_function(x) >=0  #B(X)>= 0 in all safe region where x in part of Xs

def unsafe_region(x):
    return barrier_function(x) < 0 #B(X) < 0 in all safe region where x in part of Xu

# Barrier function derivative)
def gradient_B(x):
    return np.array([2*x[0], 2*x[1]])

def Lv_B(x,theta,v):
    gradient = gradient_B(x)
    dynamics = np.array([v*np.cos(theta), v * np.sin(theta)])
    return np.dot(gradient,dynamics)

#the set C 
def set_C(x):
    return x[0] >=0

#The zero level set for the barrier function
def area_B0_level():
    return 2*np.pi*r

def C_intersect_B():
    return np.pi*r/2 #not right

def check_ratio(epsilon):
    ratio = C_intersect_B/area_B0_level
    return ratio <= epsilon


# 
trajectory = []
for t in np.arange(0, T, dt):
    trajectory.append(state)

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
