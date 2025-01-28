# from pyboy import PyBoy
from metroid_env import MetroidEnv

import time

ROM_PATH = "MetroidII.gb"

def main():
    # the example code for AI gym environments
    print("Making env")
    
    # use 'human' to see Samus!
    env = MetroidEnv(ROM_PATH, render_mode='human', emulation_speed_factor=1)


    for _ in range(10000):

        # Just take random action for now
        action = env.action_space.sample()

        # noop, do this for manual control
        # observation, reward, terminated, truncated, info = env.step(0)

        observation, reward, terminated, truncated, info = env.step(action)
        # toPrint = f"Reward: {reward}"
        # print(toPrint)

        # observation, reward, terminated, truncated, info = env.step(0)

        # Prints debugging info
        # print(env.pyboy.game_wrapper)
        
        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            observation, info = env.reset()

    env.close()



if __name__ == "__main__":
    main()
