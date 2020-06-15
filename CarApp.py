# Importing the libraries
import numpy as np
import os
import time
import torch

from random import randint

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.logger import Logger

from PIL import Image as PILImage
from models import ReplayBuffer, TD3
from scipy.ndimage import rotate

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

seed = 0  # Random seed number
# Set seed for consistency
torch.manual_seed(seed)
np.random.seed(seed)
save_models = True

file_name = "%s_%s_%s" % ("TD3", "CarApp", str(seed))
print ("---------------------------------------")
print ("Settings: %s" % (file_name))
print ("---------------------------------------")
if not os.path.exists("./results"):
  os.makedirs("./results")
if save_models and not os.path.exists("./trained_models"):
  os.makedirs("./trained_models")

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0
last_reward = 0

# Defining coordinates to start the episode from
origins = [[587,532],[365,311]]
origin_x, origin_y = origins[np.random.randint(0, 3)]
scores = []
im = CoreImage("./images/MASK1.png")


first_update = True # Setting the first update
last_distance = 0   # Initializing the last distance

def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    global Target
    sand = np.zeros((longueur, largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    sand = np.asarray(img) / 255
    goal_x = 1174
    goal_y = 611
    first_update = False
    Target = 'A'
    global swap
    swap = 0


# Creating the car class

class Car(Widget):
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    cropsize = 28
    padsize = 28
    view = np.zeros([1,int(cropsize),int(cropsize)])

    def move(self, rotation):
        global episode_num
        global padsize
        global cropsize
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation

        # Preparing the image for the state
        tempSand = np.copy(sand)
        tempSand = np.pad(tempSand,self.padsize,constant_values=1.0)
        tempSand = tempSand[int(self.x) - self.cropsize + self.padsize:int(self.x) + self.cropsize + self.padsize,
                   int(self.y) - self.cropsize + self.padsize:int(self.y) + self.cropsize + self.padsize]
        tempSand = rotate(tempSand, angle=90 - (self.angle - 90), reshape=False, order=1, mode='constant', cval=1.0)
        tempSand[int(self.padsize)-5:int(self.padsize), int(self.padsize) - 2:int(self.padsize) + 3 ] = 0.8
        tempSand[int(self.padsize):int(self.padsize) + 5, int(self.padsize) - 2:int(self.padsize) + 3] = 0.2
        self.view=tempSand
        self.view = self.view[::2, ::2]
        self.view = np.expand_dims(self.view, 0)


# Creating the game class

class Game(Widget):
    car = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(1, 0)

    def reset(self):
        global last_distance
        global origin_x
        global origin_y
        self.car.x = origin_x
        self.car.y = origin_y
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        self.distance = np.sqrt((self.car.x - goal_x) ** 2 + (self.car.y - goal_y) ** 2)
        state = [self.car.view, orientation, -orientation, last_distance - self.distance]
        return state


    def step(self,action):
        global goal_x
        global goal_y
        global origin_x
        global origin_y
        global done
        global last_distance
        global Target
        global swap
        global distance_travelled

        rotation = action.item()
        self.car.move(rotation)
        self.distance = np.sqrt((self.car.x - goal_x) ** 2 + (self.car.y - goal_y) ** 2)
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.
        state = [self.car.view, orientation, -orientation, last_distance - self.distance]

        # moving on the sand
        if sand[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            last_reward = -5


        else:  # moving on the road
            self.car.velocity = Vector(1.5, 0).rotate(self.car.angle)
            last_reward = -2

            # moving towards the goal
            if self.distance < last_distance:
                last_reward = 1


        # Near the boundary
        if self.car.x < 10:
            self.car.x = 10
            last_reward = -10
            self.car.angle = self.car.angle + randint(1, 20)

        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -10
            self.car.angle = self.car.angle + randint(1, 20)

        if self.car.y < 10:
            self.car.y = 10
            last_reward = -10
            self.car.angle = self.car.angle + randint(1, 20)

        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -10
            self.car.angle = self.car.angle + randint(1, 20)


        # Swap goal
        if self.distance < 30:
            if swap == 1:
                origin_x = goal_x
                origin_y = goal_y
                goal_x = 1174
                goal_y = 611
                Target = 'A'
                swap = 0
                last_reward = 500
                done = True
            elif swap == 2:
                origin_x = goal_x
                origin_y = goal_y
                goal_x = 245
                goal_y = 557
                Target = 'C'
                swap = 1
                last_reward = 500
                done = True
            elif swap == 0:
                origin_x = goal_x
                origin_y = goal_y
                goal_x = 613
                goal_y = 32
                Target = 'B'
                swap = 2
                last_reward = 500
                done = True

        print(Target, goal_x, goal_y, last_distance, last_distance-self.distance, last_reward, int(self.car.x), int(self.car.y),im.read_pixel(int(self.car.x), int(self.car.y)))
        last_distance = self.distance

        return state, last_reward, done

    def evaluate_policy(self, policy, eval_episodes=10):
        avg_reward = 0.
        for _ in range(eval_episodes):
            obs = self.reset()
            done = False
            while not done:
                action = policy.select_action(np.array(obs))
                obs,reward,done = self.step(action)
                avg_reward += reward
        avg_reward /= eval_episodes
        print("---------------------------------------")
        print("Average Reward over the Evaluation Step: %f" % (avg_reward))
        print("---------------------------------------")
        return avg_reward



    def update(self, dt):
        global scores
        global first_update
        global goal_x
        global goal_y
        global longueur
        global largeur
        global last_reward

        global policy
        global done
        global episode_reward
        global replay_buffer
        global obs
        global new_obs
        global evaluations

        global episode_num
        global total_timesteps
        global timesteps_since_eval
        global episode_num
        global max_timesteps
        global max_episode_steps
        global episode_timesteps
        global distance_travelled
        longueur = self.width
        largeur = self.height
        if first_update:
            init()
            evaluations = [self.evaluate_policy(policy)]
            distance_travelled=0
            done = True
            obs = self.reset()

        if episode_reward<-2000:
            done=True


        if total_timesteps < max_timesteps:

            # If the episode is done
            if done:

                if total_timesteps != 0:
                    Logger.info("Total Timesteps: {} Episode Num: {} Reward: {} iterations: {}".format(total_timesteps, episode_num,
                                                                                  episode_reward,int(episode_timesteps)))
                    # fixing the iterations to minimum of episode timesteps or 800
                    iterations_count=min(episode_timesteps,800)
                    policy.train(replay_buffer, int(iterations_count), batch_size, discount, tau, policy_noise, noise_clip,
                                 policy_freq)

                # save the policy
                if timesteps_since_eval >= eval_freq:
                    timesteps_since_eval %= eval_freq
                    evaluations.append(self.evaluate_policy(policy))
                    policy.save(file_name, directory="./trained_models")
                    np.save("./results/%s" % (file_name), evaluations)

                # When the training step is done, we reset the state of the environment
                obs = self.reset()
                # Reset parameters
                done = False
                episode_reward = 0
                episode_timesteps = 0

                episode_num += 1

            # Before start_timesteps, random actions
            if total_timesteps < start_timesteps:
                action = np.random.uniform(low=-5, high=5, size=(1,))
            else:  # After start_timesteps, use model for actions
                action = policy.select_action(np.array(obs))

                print ("Action before adding noise:" + str(action))

                # Add noise to the action and clip it
                if expl_noise != 0:
                    action = (action + np.random.normal(0, expl_noise, size=1)).clip(
                        -5, 5)
                print ("Action:" + str(action))

            # Take the action
            new_obs,reward, done = self.step(action)

            # check if episode is done
            done_bool = 0 if (episode_timesteps + 1 == max_episode_steps) else float(done)

            # update total reward
            episode_reward += reward
            # update Replay Buffer Memory
            replay_buffer.add((obs, new_obs, action, reward, done_bool))
            if total_timesteps%100==1:
                Logger.info(" ".join([str(total_timesteps), str(obs[1:]), str(new_obs[1:]), str(action), str(reward), str(done_bool)]))

            # update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            obs = new_obs
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1
            # Saving model at every 5000 iterations
            if total_timesteps%5000==1:
                Logger.info("Saving Model %s" % (file_name))
                policy.save("%s" % (file_name), directory="./trained_models")
                np.save("./results/%s" % (file_name), evaluations)
        else:
            action = policy.select_action(np.array(obs))
            new_obs,reward, done = self.step(action)
            obs = new_obs
            total_timesteps += 1
            if total_timesteps%1000==1:
                print(total_timesteps)


class CarApp(App):
    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)
        return parent


# Initialize
start_timesteps = 3e3  # Number of timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 1e3  # How often the evaluation step is performed (after how many timesteps)
max_timesteps = 5e4  # Total number of iterations/timesteps

expl_noise = 0.1  # Exploration noise
batch_size = 100
discount = 0.99  # Discount factor gamma
tau = 0.005  # Target network update rate
policy_noise = 0.2  # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5  # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2
total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0

episode_reward=0
t0 = time.time()
distance_travelled=0
max_episode_steps = 1000
done = True # Episode over
load_model=True # Inference only

state_dim = 4
action_dim = 1
max_action = 5

replay_buffer = ReplayBuffer()
policy = TD3(state_dim, action_dim, max_action)

obs=np.array([])
new_obs=np.array([])
evaluations=[]

if load_model == True:
    total_timesteps = max_timesteps
    policy.load("%s" % (file_name), directory="./trained_models")

CarApp().run()