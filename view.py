# TODO write code to load and watch a model perform here

import gymnasium as gym
from stable_baselines3 import PPO
from metroid_env import MetroidEnv

if __name__ == "__main__":
    # TODO make this script take command line arg for loading model

    env = MetroidEnv(render_mode='human', emulation_speed_factor=1)

    model= PPO("MlpPolicy", env, verbose=1)
    model = PPO.load('metroid_980000_steps.zip')

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, truncated, info = env.step(action)

