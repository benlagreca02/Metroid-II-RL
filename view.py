# TODO write code to load and watch a model perform here
import argparse

import gymnasium as gym
from stable_baselines3 import PPO
from metroid_env import MetroidEnv, DEFAULT_NUM_TO_TICK

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Path to model to load")
    parser.add_argument('model_path', type=str, help='Path to the model to be loaded')
    
    args = parser.parse_args()
    
    model_path = args.model_path

    # TODO make this script take command line arg for loading model
    env = MetroidEnv(render_mode='human',
            emulation_speed_factor=1,
            num_to_tick=DEFAULT_NUM_TO_TICK)

    model= PPO("MlpPolicy", env, verbose=1)
    
    
    model = PPO.load(model_path)

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, truncated, info = env.step(action)

