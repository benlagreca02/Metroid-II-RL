from pyboy import PyBoy
from metroid_env import MetroidEnv

ROM_PATH = "MetroidII.gb"

def main():
    pyboy = PyBoy(ROM_PATH)
    env = MetroidEnv(pyboy)

    # Just take random action for now
    for _ in range(1000000):
        # this is where you would insert your policy
        action = env.action_space.sample()

        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        observation, reward, terminated, truncated, info = env.step(action)


        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            observation, info = env.reset()

    env.close()



if __name__ == "__main__":
    main()
