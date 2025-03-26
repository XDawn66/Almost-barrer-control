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
import pickle
import os
from sklearn.neighbors import NearestNeighbors
# Constants
# WIDTH = 1600
# HEIGHT = 880

WIDTH = 1920
HEIGHT = 1080

CAR_SIZE_X = 60    
CAR_SIZE_Y = 60

BORDER_COLOR = (255, 255, 255, 255) # Color To Crash on Hit
CHECKPOINT_COLOR = (227, 24, 45, 255) # Color To Checkpoint

class Dataset:
    def __init__(self):
        self.safe_data_set = np.empty((0, 4))  # Initialize with an empty array
        self.unsafe_data_set = np.empty((0, 4)) 
        # self.load_safe_data()  # Load the data if it exists
        # self.load_unsafe_data()

    def load_safe_data(self):
        """Load previously collected data if exists."""
        if os.path.exists("data/safe_data_set.pkl"):
            # Check if the file is not empty
            if os.path.getsize("data/safe_data_set.pkl") > 0:
                try:
                    with open("data/safe_data_set.pkl", "rb") as f:
                        self.safe_data_set = pickle.load(f)
                    print("Loaded safe dataset.")
                except EOFError:
                    print("File is empty or corrupted, initializing a new dataset.")
                    self.safe_data_set = np.empty((0, 4))  # Initialize as empty if the file is corrupted
            else:
                print("File is empty, initializing a new dataset.")
                self.safe_data_set = np.empty((0, 4))  # Initialize as empty
        else:
            print("No dataset found, initializing a new dataset.")
            self.safe_data_set = np.empty((0, 4))  # Initialize as empty

    def load_unsafe_data(self):
        """Load previously collected data if exists."""
        if os.path.exists("data/unsafe_data_set.pkl"):
            # Check if the file is not empty
            if os.path.getsize("data/unsafe_data_set.pkl") > 0:
                try:
                    with open("data/unsafe_data_set.pkl", "rb") as f:
                        self.unsafe_data_set = pickle.load(f)
                    print("Loaded unsafe dataset.")
                except EOFError:
                    print("File is empty or corrupted, initializing a new dataset.")
                    self.unsafe_data_set = np.empty((0, 4))  # Initialize as empty if the file is corrupted
            else:
                print("File is empty, initializing a new dataset.")
                self.unsafe_data_set = np.empty((0, 4))  # Initialize as empty
        else:
            print("No dataset found, initializing a new dataset.")
            self.unsafe_data_set = np.empty((0, 4))  # Initialize as empty

    def save_safe_data(self):
        """Save collected data to file."""
        os.makedirs("data", exist_ok=True)
        save_path = os.path.join("data", "safe_data_set3.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(self.safe_data_set, f)
        print("Dataset saved.")
    
    def save_unsafe_data(self):
        """Save collected data to file."""
        os.makedirs("data", exist_ok=True)
        save_path = os.path.join("data", "unsafe_data_set.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(self.unsafe_data_set, f)
        print("Dataset saved.")

# Create Dataset object
dataset = Dataset()


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

        self.max_steps = 1000  # Fixed horizon length
        self.current_step = 0
        self.car_safe_data_set = dataset.safe_data_set
        self.unsafe_data_set = []

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
        map_width, map_height = game_map.get_size()
        for point in self.corners:
            # If Any Corner Touches Border Color -> Crash
            # Assumes Rectangle
            x,y = int(point[0]), int(point[1])
            if 0 <= x < map_width and 0 <= y < map_height:
                if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR :
                    self.alive = False
                    #self.checkpoint_reached = 0 # Reset checkpoints
                    break
            else:
                self.alive = False
                break

    def check_radar(self, degree, game_map):
        length = 0
        map_width, map_height = game_map.get_size()
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # While We Don't Hit BORDER_COLOR AND length < 300 (just a max) -> go further and further
        while length < 300:
            length = length + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)
            if 0 <= x < map_width and 0 <= y < map_height:
                if game_map.get_at((x, y)) == BORDER_COLOR:
                    break  # Stop if we hit the border
            else:
                break  # Stop if out of bounds
        # Calculate Distance To Border And Append To Radars List
        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])
    
    def update(self, game_map):
        # Set The Speed To 20 For The First Time
        # Only When Having 4 Output Nodes With Speed Up and Down
        if not self.speed_set:
            self.speed = 10
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
        
    def is_alive(self):
        # Basic Alive Function
        return self.alive

    def get_reward(self, last_position, game_map):

        X = self.position[0]
        Y = self.position[1]
        goal = [830, 920]
        distance = np.sqrt(float(X-last_position[0])**2 + float(Y-last_position[1])**2)
        reward =  distance/(CAR_SIZE_X)

        if not self.is_alive():
            reward -= 25  # Harsh penalty for crashin
            remaining_steps = self.max_steps - self.current_step
            early_crash_penalty = 30 * (remaining_steps / self.max_steps)  # More penalty for early crashes
            reward -= early_crash_penalty 
            

        max_derivation = 30 # for not penalitzing too much
        stay_center = abs(self.radars[0][1]-self.radars[1][1]) /max_derivation
        reward -= stay_center * 5  # Penalize deviation from center

        desired_speed = 10.0
        speed_penalty =  2 * (self.speed - desired_speed) ** 2
        reward -= speed_penalty


        # Check if the goal is reached
        if self.check_goal_reached(self.position):
            reward += 200  # Reward for reaching the goal   
        #     #print("goal reached")

        reward += 0.5 #survival bonus
        # x, y = int(X), int(Y)

        #reward += 15 * self.checkpoint_reached
        
        #dis_from_goal = max(0,self.total_checkpoints - self.checkpoint_reached)
        #reward -= dis_from_goal 
        
        #color = game_map.get_at((x, y))[:3]
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
    
    def set_start(self, game_map):
        # Collect data for training
        time = 0
        map_width, map_height = game_map.get_width(), game_map.get_height()
        while True:
            Ran_X = random.randint(0, map_width - 1)
            Ran_Y = random.randint(0, map_height - 1)
            start = [Ran_X, Ran_Y]
            time += 1
            # Check if the sampled point is a valid starting location
            if game_map.get_at(start) != BORDER_COLOR:  # Check RGB values match
                self.position = start  # Set the car's position
                self.angle = random.uniform(-180, 180)  # Randomize starting angle
                self.speed = 0  # Reset speed
                #print(f"Car starting position: {self.position}, angle: {start[:3]}")
                break
            else:
                if time % 3000 == 0:
                    print("Loading initial starting point...")

    def get_safe_data(self):
        if len((dataset.safe_data_set)) < 20000:
            print(len((dataset.safe_data_set)))
            if self.is_alive():
                new_data = np.array([[self.position[0], self.position[1], self.angle, self.speed]])
                #print(new_data)
                new_data = np.array(new_data)  # Ensure new_data is a numpy array
                if new_data.shape[1] == 4:  # Check if it has 4 columns
                    dataset.safe_data_set = np.vstack((dataset.safe_data_set, new_data))
                else:
                    print("Error: new_data does not have 4 columns.")
                
        if len((dataset.safe_data_set)) >= 20000:
            dataset.save_safe_data()
            sys.exit(0)


    def get_unsafe_data(self):
        M = 4000
        k = 18  # Number of neighbors to check
        epsilon = 160  # Small threshold for closeness check
        safe_count_threshold = k // 2  # Majority rule
        all_dist = []
        #Xc = np.random.uniform(low=[0, 0, -180, 0], high=[1919, 1079, 180, 30], size=(M, 4))
        while len(dataset.unsafe_data_set) < 20000:
            #generate a random Xc set at each iteration
            #Xs U Xu U Xc 
            Xc = np.random.uniform(low=[0, 0, -180, 0], high=[1919, 1079, 180, 30], size=(M, 4))
            combined_data = np.vstack((dataset.safe_data_set, dataset.unsafe_data_set, Xc))
            #KNN to find nearest neigbor
            knn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(combined_data)
            distances, neighbors  = knn.kneighbors(Xc)
            # all_dist.append(distances)

            to_remove_indices = []
            for i, candidate in enumerate(Xc):
                #find the distance between current point 
                distances = np.linalg.norm(dataset.safe_data_set - np.array(candidate), axis=1)
                k_nearest_indices = np.argsort(distances)[:k]
                #print(dataset.safe_data_set)

                safe_count = np.sum(distances[k_nearest_indices] < epsilon)
                #print(safe_count)
                #print(candidate)
                if safe_count > k // 2:
                    to_remove_indices.append(i) # when majority of the neighbors are safe, remove the candidate
                    #print("remove", i)
            #print(to_remove_indices[:10])
            Xc = np.delete(Xc, to_remove_indices, axis=0)
            print(len(Xc))
            dataset.unsafe_data_set = np.vstack((dataset.unsafe_data_set, Xc))
            #print(len(dataset.unsafe_data_set))
            # all_dists = np.concatenate(all_dist)

            # print(f"Min Distance: {np.min(all_dists)}")
            # print(f"Max Distance: {np.max(all_dists)}")
            # print(f"Mean Distance: {np.mean(all_dists)}")
            # print(f"Median Distance: {np.median(all_dists)}")
            # print(f"Percentiles (10%, 25%, 50%, 75%, 90%): {np.percentile(all_dists, [10, 25, 50, 75, 90])}")
        dataset.save_unsafe_data()
          

    def runsimulation(self):
        #a timer to force stop
        total_steps =  20000
        map_paths = ['map.png', 'map2.png', 'map3.png','map4.png','map5.png']
        map_weights = [1, 2, 3, 4, 5]
        weight_sum = np.sum(map_weights)
        current_map_index = 0  # Start with the first map
        timeforcurrentmap = int((1/weight_sum) * total_steps)

        #setting up the pygame
        info = pygame.display.Info()
        game_map = pygame.image.load(map_paths[current_map_index]).convert()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
    
        #seting up the SAC
        env = Env.CarEnv(game_map,screen)
        #loaded the model I trained so far
        model = SAC.load("models/MlpPolicy_try_6",env, tensorboard_log="./sac_car_env/")
        pygame.init()  # Initialize Pygame
        #model = SAC("MlpPolicy", env,verbose=1, tensorboard_log="./sac_car_env/")

        # Train the agent for a set number of timesteps
        #model.learn(total_timesteps=total_steps, tb_log_name="SAC_run")
        # Save the trained model 
        #model.save("models/MlpPolicy_nural_1")

        # Initialize previous position
        #self.previous_position = self.position
        initial_position = [830, 920]
        obs, info = env.reset()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
            
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, end, info = env.step(action)
            if end:
                #dataset.save_safe_data()
                 #self.previous_position = initial_position
                self.previous_position = env.car.position #if the car crash go back to initial position
            else:
                #self.previous_position = initial_position
                self.previous_position = env.car.position #update the current car positon
            #print("step done", self.previous_position)

            env.render()  # Call render to draw the car
            total_steps += 1  # Increment step count
            
            #self.get_safe_data()
            #if total_steps % timeforcurrentmap == 0: #for training, comment out when testing
            if total_steps % 3000 == 0: #testing purpose comment out when training
                print("time step for current map : ",timeforcurrentmap)
                timeforcurrentmap = int([current_map_index + 1]/weight_sum * total_steps) # Increase interval gradually
                current_map_index = (current_map_index + 1) % len(map_paths)  # Rotate through maps
                new_map_png = map_paths[current_map_index]
                new_map = pygame.image.load(new_map_png).convert()
                env = Env.CarEnv(new_map,screen)
                obs, info = env.reset()

            if total_steps % 3000 == 0:
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
    #mycar.check_safe_data()
    #mycar.get_unsafe_data()