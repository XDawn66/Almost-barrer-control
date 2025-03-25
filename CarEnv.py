#I refered the following link for SAC set up: https://urldefense.com/v3/__https://stable-baselines3.readthedocs.io/en/master/modules/sac.html__;!!Mih3wA!Dks5NJjC-9qHMg6oI07BtQzQixhX7euHpyaMmbC7SsG0hmmYesMG7iKNvaxOcMXngcUw75LYzGnczlBBBA$

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import newcar
import random
import pygame
from newcar import dataset
import torch
import torch.nn as nn

class BarrierFunctionNet(nn.Module):
    def __init__(self, input_dim):
        super(BarrierFunctionNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        hidden = self.tanh(self.fc1(x))
        return self.fc2(hidden)  # Output barrier function value

class CarEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(self,game_map,screen):
        super(CarEnv,self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        #self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        
        # [steering, throttle]
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        #creating a car object and set up the gamemap
        self.car = newcar.Car()
        self.game_map = game_map
        self.screen = screen

        input_dim = 4  # Set based on your state vector size
        self.barrier_net = BarrierFunctionNet(input_dim)
        self.barrier_net.load_state_dict(torch.load("checkpoint9.pth")["model_state_dict"])
        self.barrier_net.eval()  # Set to evaluation mode
        self.Certified = 0
        self.total = 0


    def step(self, action):
        steering, throttle = action

        # Predict next state
        predicted_state = self._predict_next_state(steering, throttle)

        # Evaluate safety using barrier function
        state_tensor = torch.tensor([self.car.position[0], self.car.position[1], self.car.angle, self.car.speed], dtype=torch.float32).unsqueeze(0)

        B_value = self.barrier_net(state_tensor).item()
        # if B_value < 0:
        # print("B_value", B_value)
        # # Modify action if unsafe
        # if B_value < 0:  # Unsafe state detected
        #     steering *= -0.5  # Reverse steering to avoid danger
        #     throttle *= 0.5   # Reduce speed

        if B_value > 0:
            self.Certified +=1
        self.total += 1
        Certified_rate = self.Certified/self.total
        print("Certified_reate: ",Certified_rate)

        # Apply action
        self.car.angle += steering * 10  
        self.car.speed = np.clip(self.car.speed + throttle * 2, 10, 30)
        self.car.update(self.game_map)
        
        # Check termination
        done = not self.car.is_alive()
        reward = self.car.get_reward(self.car.previous_position, self.game_map)
        
        observation = self._get_observation()
        info = {"TimeLimit.truncated": done}

        return observation, reward, done, done, info

    def _predict_next_state(self, steering, throttle):
        """Estimate next state based on current state and action."""
        state = np.array([self.car.position[0], self.car.position[1], self.car.angle])
        next_state = self.car_dynamics(state, self.car.speed, steering, throttle)
        return next_state

    import numpy as np

    def car_dynamics(self, state, v, steering, throttle, dt=0.1, L=2.5):
        """
        Simulates car dynamics using the kinematic bicycle model.

        Args:
            state (array): [x, y, theta] - current position and heading
            v (float): current velocity
            steering (float): steering angle input (-1 to 1)
            throttle (float): acceleration input (-1 to 1)
            dt (float): time step
            L (float): wheelbase of the car

        Returns:
            next_state (array): [x_next, y_next, theta_next]
        """
        #print("state", state)
        x, y, theta = state

        # Convert steering input to angle
        max_steering_angle = np.radians(30)  # Limit steering to 30 degrees
        delta = steering * max_steering_angle

        # Update velocity
        v = np.clip(v + throttle * 2, 10, 30)  # Limit speed between 10 and 30

        # Kinematic bicycle model equations
        x_next = x + v * np.cos(theta) * dt
        y_next = y + v * np.sin(theta) * dt
        theta_next = theta + (v / L) * np.tan(delta) * dt

        return np.array([x_next, y_next, theta_next])


    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
        self.car = newcar.Car()
        #self.car.set_start(self.game_map)
        #self.car.position = [830, 920]
        observation = self._get_observation()
        #self.car.checkpoint_reached = 0
        self.car.current_step = 0
        return observation, {}

    def render(self):
        self.screen.blit(self.game_map, (0, 0))
        self.car.draw(self.screen)
        pygame.display.flip()
    
    #this code I refered ChatGpt, I am not really know how to get observation
    def _get_observation(self):
        radar_distances = np.array(self.car.get_data()) / 25.0  # Normalize radar distances # i tried to change values here and the training improved
        speed = self.car.speed / 30.0  # Normalize speed
        angle = self.car.angle / 360.0  # Normalize angle
        
        return np.concatenate((radar_distances, [speed, angle])) 

    def close(self):
        pygame.quit()