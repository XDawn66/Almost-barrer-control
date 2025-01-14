# This Code is Heavily Inspired By The YouTuber: Cheesy AI
# Code Changed, Optimized And Commented By: NeuralNine (Florian Dedov)

#I refered the following link for SAC set up: https://urldefense.com/v3/__https://stable-baselines3.readthedocs.io/en/master/modules/sac.html__;!!Mih3wA!Dks5NJjC-9qHMg6oI07BtQzQixhX7euHpyaMmbC7SsG0hmmYesMG7iKNvaxOcMXngcUw75LYzGnczlBBBA$
import math
import random
import sys
import os
import numpy as np
#import neat
import pygame
import gymnasium as gym
from stable_baselines3 import SAC
import CarEnv as Env

# Constants
# WIDTH = 1600
# HEIGHT = 880

WIDTH = 1920
HEIGHT = 1080

CAR_SIZE_X = 60    
CAR_SIZE_Y = 60

BORDER_COLOR = (255, 255, 255, 255) # Color To Crash on Hit
CHECKPOINT_COLOR = (227, 24, 45, 255) # Color To Checkpoint

current_generation = 0 # Generation counter
sharp_turn_threshold = 0.1  # Example value, adjust based on curvature calculation


class Car:

    def __init__(self):
        # Load Car Sprite and Rotate
        self.sprite = pygame.image.load('car.png').convert() # Convert Speeds Up A Lot
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite 
        # self.position = [690, 740] # Starting Position
        self.position = [830, 920] # Starting Position
        self.angle = 0
        self.speed = 0

        self.speed_set = False # Flag For Default Speed Later on

        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2] # Calculate Center

        self.radars = [] # List For Sensors / Radars
        self.drawing_radars = [] # Radars To Be Drawn

        self.alive = True # Boolean To Check If Car is Crashed
        self.previous_position = [830, 920]
        self.distance = 0 # Distance Driven
        self.time = 0 # Time Passed
        self.total_checkpoints = 5 # Total Checkpoints Passed
        self.checkpoint_reached = 0

        self.GOAL_AREA = {
        "x_min": 800,  # Minimum x-coordinate of the goal
        "x_max": 829,  # Maximum x-coordinate of the goal
        "y_min": 880,  # Minimum y-coordinate of the goal
        "y_max": 960,  # Maximum y-coordinate of the goal
}


    def draw(self, screen):
        if isinstance(self.position, (list, tuple)) and len(self.position) == 2:
            blit_position = (int(self.position[0]), int(self.position[1])) 
            screen.blit(self.rotated_sprite, blit_position) # Draw Sprite
        self.draw_radar(screen) #OPTIONAL FOR SENSORS

    def draw_radar(self, screen):
        # Optionally Draw All Sensors / Radars
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            # If Any Corner Touches Border Color -> Crash
            # Assumes Rectangle
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                #self.checkpoint_reached = 0 # Reset checkpoints
                break

    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # While We Don't Hit BORDER_COLOR AND length < 300 (just a max) -> go further and further
        while not game_map.get_at((x, y)) == BORDER_COLOR and length < 300:
            length = length + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # Calculate Distance To Border And Append To Radars List
        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])
    
    def update(self, game_map):
        # Set The Speed To 20 For The First Time
        # Only When Having 4 Output Nodes With Speed Up and Down
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        # Get Rotated Sprite And Move Into The Right X-Direction
        # Don't Let The Car Go Closer Than 20px To The Edge
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        # Increase Distance and Time
        self.distance += self.speed
        self.time += 1
        
        # Same For Y-Position
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIDTH - 120)

        # Calculate New Center
        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

        # Calculate Four Corners
        # Length Is Half The Side
        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        #left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        #right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        #self.corners = [left_top, right_top, left_bottom, right_bottom]
        self.corners = [left_top, right_top]


        # Check Collisions And Clear Radars
        self.check_collision(game_map)
        self.radars.clear()

        # From -90 To 120 With Step-Size 45 Check Radar
        # for d in range(-90, 120, 45):
        #     self.check_radar(d, game_map)
        for d in range(-45, 135, 90):
             self.check_radar(d, game_map)

    def get_data(self):
        # Get Distances To Border
        radars = self.radars
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)

        return return_values
    
    def check_goal_reached(self, car_position):
        car_x, car_y = car_position  # Get the car's current position
        
        if (self.GOAL_AREA["x_min"] <= car_x <= self.GOAL_AREA["x_max"] and 
            self.GOAL_AREA["y_min"] <= car_y <= self.GOAL_AREA["y_max"]):
            return True  # Goal is reached
        return False

    def check_checkpoint_reached(self, car_position, game_map):
        #print(self.checkpoint_reached, "checkpoint reached")
        x, y = int((car_position[0])), int((car_position[1]))
        #color = game_map.get_at((x, y))[:4]  # Get the RGB color of the current position
        #print("color",color)
        if game_map.get_at((x,  y)) == CHECKPOINT_COLOR:
            self.checkpoint_reached += 1
            #print("checkpoint reached", self.checkpoint_reached)
        
    def is_alive(self):
        # Basic Alive Function
        return self.alive

    def get_reward(self, last_position, game_map):
        # Calculate Reward (Maybe Change?)
        # return self.distance / 50.0
        # return self.distance / (CAR_SIZE_X / 2)
        #print("last",last_position)
        #print("now:", self.position)
        #print(radar_data)

        X = self.position[0]
        Y = self.position[1]
        goal = [830, 920]
        distance = np.sqrt(float(X-last_position[0])**2 + float(Y-last_position[1])**2)
        
        reward =  distance/(CAR_SIZE_X / 2)
        if not self.is_alive():
            reward -= 30  # Harsh penalty for crashin

        # sharp_turn_threshold = 0.3  # Example threshold for sharp curvature
        # max_speed = 30.0  # Maximum desired speed
        # if curvature > sharp_turn_threshold:
        #     reward += (max_speed - self.speed) * 0.15  # Encourage slowing down in sharp turns
        stay_center = abs(self.radars[0][1]-self.radars[1][1]) 
        reward -= stay_center * 0.1  # Penalize deviation from center

        desired_speed = 12.0
        speed_penalty = 0.5 * (self.speed - desired_speed) ** 2
        reward -= speed_penalty

        # Check if the goal is reached
        if self.check_goal_reached(self.position):
            reward += 200  # Reward for reaching the goal   
            #print("goal reached")
        x, y = int(X), int(Y)
        reward += 10 * self.checkpoint_reached 
        
        dis_from_goal = min(0,self.total_checkpoints - self.checkpoint_reached)
        reward -= dis_from_goal * 8
        color = game_map.get_at((x, y))[:3]
        #if color != (0,0,0) and color !=(255, 255, 255) and color !=((28, 28, 28)):
            #print(color)
        return reward

    def rotate_center(self, image, angle):
        # Rotate The Rectangle
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image
    


#This is the default neat ML which i didn't use
# def run_simulation(genomes, config):
    
#     # Empty Collections For Nets and Cars
#     nets = []
#     cars = []

#     # Initialize PyGame And The Display
#     pygame.init()
#     screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)

#     # For All Genomes Passed Create A New Neural Network
#     for i, g in genomes:
#         # net = neat.nn.FeedForwardNetwork.create(g, config)
#         # nets.append(net)
#         g.fitness = 0

#         cars.append(Car())

#     # Clock Settings
#     # Font Settings & Loading Map
#     clock = pygame.time.Clock()
#     generation_font = pygame.font.SysFont("Arial", 30)
#     alive_font = pygame.font.SysFont("Arial", 20)
#     game_map = pygame.image.load('map2.png').convert() # Convert Speeds Up A Lot

#     global current_generation
#     current_generation += 1

#     # Simple Counter To Roughly Limit Time (Not Good Practice)
#     counter = 0

#     while True:
#         # Exit On Quit Event
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 sys.exit(0)

#         # For Each Car Get The Acton It Takes
#         for i, car in enumerate(cars):
#             output = nets[i].activate(car.get_data())
#             choice = output.index(max(output))
#             if choice == 0:
#                 car.angle += 10 # Left
#             elif choice == 1:
#                 car.angle -= 10 # Right
#             elif choice == 2:
#                 if(car.speed - 2 >= 12):
#                     car.speed -= 2 # Slow Down
#             else:
#                 car.speed += 2 # Speed Up
        
#         # Check If Car Is Still Alive
#         # Increase Fitness If Yes And Break Loop If Not
#         still_alive = 0
#         for i, car in enumerate(cars):
#             if car.is_alive():
#                 still_alive += 1
#                 car.update(game_map)
#                 genomes[i][1].fitness += car.get_reward()

#         if still_alive == 0:
#             break

#         counter += 1
#         if counter == 30 * 40: # Stop After About 20 Seconds
#             break

#         # Draw Map And All Cars That Are Alive
#         screen.blit(game_map, (0, 0))
#         for car in cars:
#             if car.is_alive():
#                 car.draw(screen)
        
#         # Display Info
#         text = generation_font.render("Generation: " + str(current_generation), True, (0,0,0))
#         text_rect = text.get_rect()
#         text_rect.center = (900, 450)
#         screen.blit(text, text_rect)

#         text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
#         text_rect = text.get_rect()
#         text_rect.center = (900, 490)
#         screen.blit(text, text_rect)

#         pygame.display.flip()
#         clock.tick(60) # 60 FPS

    def runsimulation(self):
        #a timer to force stop
        total_steps = 60000
        map_paths = ['map.png', 'map2.png', 'map3.png','map4.png','map5.png']
        map_weights = [1, 2, 3, 4, 5]
        weight_sum = np.sum(map_weights)
        current_map_index = 0  # Start with the first map
        timeforcurrentmap = int((1/weight_sum) * total_steps)

        #setting up the pygame
        info = pygame.display.Info()
        screen_width, screen_height = info.current_w, info.current_h
        game_map = pygame.image.load(map_paths[current_map_index]).convert()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
    
        #seting up the SAC
        env = Env.CarEnv(game_map,screen)
        #loaded the model I trained so far
        model = SAC.load("MlpPolicy_try_5",env, tensorboard_log="./sac_car_env/")
        #pygame.init()  # Initialize Pygame
        #this is for creating a new model
        #model = SAC("MlpPolicy", env,verbose=1, tensorboard_log="./sac_car_env/")
        #model.load("MlpPolicy3")

        # Train the agent for a set number of timesteps
        model.learn(total_timesteps=total_steps, tb_log_name="SAC_run")
        # Save the trained model 
        model.save("MlpPolicy_try_5")

        # Initialize previous position
        self.previous_position = self.position

        obs, info = env.reset()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
            #print("step start", self.previous_position)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, end, info = env.step(action)
            if end:
                self.previous_position = [830, 920] #if the car crash go back to initial position
            else:
                self.previous_position = env.car.position #update the current car positon
            #print("step done", self.previous_position)

            env.render()  # Call render to draw the car
            total_steps += 1  # Increment step count
            
            
            #if total_steps % timeforcurrentmap == 0: #for training, comment out when testing
            if total_steps % 3000 == 0: #testing purpose comment out when training
                print("time step for current map : ",timeforcurrentmap)
                timeforcurrentmap = int([current_map_index + 1]/weight_sum * total_steps) # Increase interval gradually
                current_map_index = (current_map_index + 1) % len(map_paths)  # Rotate through maps
                new_map_png = map_paths[current_map_index]
                new_map = pygame.image.load(new_map_png).convert()
                env = Env.CarEnv(new_map,screen)
                obs, info = env.reset()

            if total_steps % 5000 == 0:
                print(f"Total Steps: {total_steps}, Reward: {reward}")
                print("obs",obs)
                print("info", info)
                print(model)
                print("terminated",terminated)
                print("end",env.car.is_alive())

            if terminated:
                obs, info = env.reset()
                
            
        env.close()



if __name__ == "__main__":
    pygame.init()  # Initialize Pygame
    info = pygame.display.Info()
    screen_width, screen_height = info.current_w, info.current_h
    pygame.display.set_mode((WIDTH, HEIGHT))
    mycar = Car()
    mycar.runsimulation()
