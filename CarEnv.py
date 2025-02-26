#I refered the following link for SAC set up: https://urldefense.com/v3/__https://stable-baselines3.readthedocs.io/en/master/modules/sac.html__;!!Mih3wA!Dks5NJjC-9qHMg6oI07BtQzQixhX7euHpyaMmbC7SsG0hmmYesMG7iKNvaxOcMXngcUw75LYzGnczlBBBA$

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import newcar
import random
import pygame
from newcar import dataset

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


    def step(self, action):
        steering, throttle = action
        self.car.angle += steering * 10  # Scale steering for more effect
        self.car.current_step += 1
        self.car.speed = np.clip(self.car.speed + throttle * 2, 10, 30)  # Control speed
        self.car.update(self.game_map) 
        self.car.get_safe_data()
        #print(self.car.safe_data_set)
        #check the car  is still alive
        #done = False
        # if not self.car.is_alive() or self.car.current_step >= self.car.max_steps:
        #     done = True  # Terminate episode if the car crashes or reaches max steps
        #try to see if the car still alive after running the trained model, so that I can use it for somewhere else
        done = not self.car.is_alive()
        # if done:
        #     reward = -15 #get punishment if the car crash
        # else:    
        #     reward = self.car.get_reward(self.car.previous_position)
        # self.car.check_checkpoint_reached(self.car.position, self.game_map)
        # if self.car.checkpoint_reached > 5:
        #     self.car.checkpoint_reached = 5
            
        reward = self.car.get_reward(self.car.previous_position,self.game_map)
        observation = self._get_observation()

        
        self.car.previous_position = self.car.position
        #I just copy the example for the info, have it there just for requirement. No sure how to adject it.
        info = {
        "TimeLimit.truncated": done,  # Example key, adjust as necessary
    }
        #I just following the syntax for the SAC(required 5 return variables)
        return observation, reward, done, done, info
    
        #I tried to return in this syntax but somehow the model will not be trained at all
        #return observation, reward, done, end, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
        self.car = newcar.Car()
        self.car.set_start(self.game_map)
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