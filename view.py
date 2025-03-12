import argparse
from PIL import Image
from metroid_env import *

def get_action_from_model(model, obs):
    action, _states = model.predict(obs)
    return action, _states
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', nargs='?', type=str, help='Path to the model to be loaded')
    parser.add_argument('-a', '--area', action='store_true', help='Print game area')
    parser.add_argument('-d', '--debug', action='store_true', help='show pyboy debug windows')
    parser.add_argument('-c', '--coords', action='store_true', help='Print coordinate values (Pixels and Area)')
    parser.add_argument('-o', '--observation', action='store_true', help='Print observation space')
    parser.add_argument('-r', '--random_agent', action='store_true', help='take random actions')
    parser.add_argument('-g', '--generate_image', action='store_true', help='Generates a png of the observation space') 
    parser.add_argument('-s', '--shape', action='store_true', help="prints shape of observations")
    
    args = parser.parse_args()
    
    if not args.model_path:
        print("No model given", end=' ')

    if args.random_agent:
        print("Taking random actions")
    else:
        print("Using human control")

    if args.area:
        print("Printing game area from pyboy")

    if args.coords:
        print("Printing coordinates of samus")

    if args.observation:
        print("Printing observation space")

    if args.shape:
        print("Printing shape of observations")

    import gymnasium as gym
    from stable_baselines3 import PPO

    env = MetroidEnv(
            render_mode='human',
            emulation_speed_factor=1, # Because we want to watch it, run in "real time"
            num_to_tick= 1 if not args.model_path else DEFAULT_NUM_TO_TICK,
            debug=args.debug)
    
    if args.generate_image:
        print("SAVING AN IMAGE")
        for _ in range(10):
            obs, rewards, dones, truncated, info = env.step(0)

        h,w,dims = obs.shape
        image_2d = obs.reshape(h, w)

        # Create a PIL Image object
        image = Image.fromarray(image_2d)
        print(image)

        image.save("obs.png")
        return

    if args.model_path:
        print("Loading model from %s" % args.model_path)
        # has no memory, use frame stacking or LSTM in future if you use SB3
        # instead of puffer/cleanRL...
        model= PPO("CnnPolicy", env, verbose=1) 
        model = PPO.load(args.model_path)
        action_getter = lambda obs: get_action_from_model(model, obs)
    elif args.random_agent:
        action_getter = lambda obs: (env.action_space.sample(), None)
    else:
        # print("Using human player mode")
        action_getter = lambda obs: (0,None)


    obs, info = env.reset()
    # Try catch is a little redundant, but this makes it quit a lot faster and
    # more reliably

    import numpy as np
    np.set_printoptions(linewidth=np.inf)
    
    while True:
        # action, _states = model.predict(obs)
        action, _states = action_getter(obs)
        obs, rewards, dones, truncated, info = env.step(action)
        if args.area:
            print(f"AREA: \n{env.game_area()}")
        if args.coords:
            print(f"{env.getAllCoordData()}")
        if args.observation:
            print(f"\n{obs}")
        if args.shape:
            print(f"{[(k, type(o)) for k,o in obs.items()]}")


if __name__ == "__main__":
    main()
