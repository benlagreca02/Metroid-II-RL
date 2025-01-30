# from pyboy import PyBoy
from metroid_env import MetroidEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import time

ROM_PATH = "MetroidII.gb"

# the example code for AI gym environments
print("Making env")

# use 'human' rendermode to see Samus!
env = MetroidEnv(ROM_PATH, render_mode=None, emulation_speed_factor=0)

# Should maybe use CNN?
model = PPO("MlpPolicy", env, verbose=1)

# DO THE TRAINING
model.learn(total_timesteps=10000, progress_bar=True)



# vec_env = model.get_env()
