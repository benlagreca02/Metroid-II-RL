# from pyboy import PyBoy
from metroid_env import MetroidEnv

from stable_baselines3.common.env_checker import check_env

import time

ROM_PATH = "MetroidII.gb"

def main():
    # the example code for AI gym environments
    
    # use 'human' to see Samus!
    # emulation speed factor is the speed to emulate at
    # 0 means "Go as fast as you can"
    # 1 means "go at normal gameboy 'realtime' speed"
    env = MetroidEnv(ROM_PATH, 
            render_mode='human',
            emulation_speed_factor=1,  # speed to run emulation
            num_to_tick=1,   # frames to skip over i.e. do an input every N frames
            debug=False)

    # Stable baselines 3 check for 
    # check_env(env, warn=True)


    for _ in range(10000):

        # Just take random action for now
        action = env.action_space.sample()

        # noop, do this for manual control
        # observation, reward, terminated, truncated, info = env.step(0)
        observation, reward, terminated, truncated, info = env.step(action)


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
