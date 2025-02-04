# from pyboy import PyBoy
from metroid_env import MetroidEnv

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
import cv2
import time
import sys
import numpy as np

ROM_PATH = "MetroidII.gb"

def main():
    # the example code for AI gym environments
    
    # use 'human' to see Samus!
    # emulation speed factor is the speed to emulate at
    # 0 means "Go as fast as you can"
    # 1 means "go at normal gameboy 'realtime' speed"
    human_env = MetroidEnv(ROM_PATH, 
            render_mode='human',
            emulation_speed_factor=1,  # speed to run emulation
            # num_to_tick=1,   # frames to skip over i.e. do an input every N frames
            debug=False)

    fast_env = gym.make("MetroidII")

    human_env.reset()
    fast_env.reset()

    truncated = False
    i = 0
    t0 = time.time()
    while i < 1000:
        action = human_env.action_space.sample()
        observation, reward, terminated, truncated, info = human_env.step(action)
        i += 1
    t1 = time.time()
    dt_human = t1-t0
    print(f"Human Time: {dt_human}")

    i = 0
    t0 = time.time()
    while i < 1000:
        action = fast_env.action_space.sample()
        observation, reward, terminated, truncated, info = fast_env.step(action)
        i += 1
    t1 = time.time()

    dt_fast = t1-t0
    print(f"Machine time: {dt_fast}")
    print(f"Machine is {dt_human/dt_fast} times faster")

    print(f"time per step HUMAN: {dt_human/1000}")
    print(f"time per step FAST: {dt_fast/1000}")
    hour = 3600 # seconds 
    # time / (time/iterations) -> time
    ans = hour / (dt_human/1000)

    print(f"One hour of human gameplay = {ans}")

    return




    # Stable baselines 3 check for 

    for _ in range(10000):

        # Just take random action for now
        action = env.action_space.sample()

        # noop, do this for manual control
        observation, reward, terminated, truncated, info = env.step(0)
        # observation, reward, terminated, truncated, info = env.step(action)

        xA, yA = env.getCoordinatesArea()
        xP, yP = env.getCoordinatesPixels()
        print(f"P: {xP, yP}\tA: {xA, yA}\tReward: {reward}")

        # Prints debugging info
        # print(env.pyboy.game_wrapper)
        
        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            print("DONE")
            observation, info = env.reset()

    env.close()



if __name__ == "__main__":
    main()
