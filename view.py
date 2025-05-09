from metroid_env import *
import argparse
from PIL import Image
import sys

def get_action_from_model(model, obs):
    action, _states = model.predict(obs)
    return action, _states
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', nargs='?', type=str, help='Path to the model to be loaded')
    parser.add_argument('-a', '--area', action='store_true', help='Print game area')
    parser.add_argument('-d', '--debug', action='store_true', help='show pyboy debug windows')
    parser.add_argument('-c', '--coords', action='store_true', help='Print coordinate values (Pixels and Area), and the number we have already explored')
    parser.add_argument('-o', '--observation', action='store_true', help='Print observation space')
    parser.add_argument('-r', '--random_agent', action='store_true', help='take random actions')
    parser.add_argument('-g', '--generate_image', action='store_true', help='Generates a png of the observation space') 
    parser.add_argument('-s', '--shape', action='store_true', help="prints shape of observations")
    parser.add_argument('-w', '--reward', action='store_true', help="print reward")
    parser.add_argument('-e', '--reset', action='store_true', help='reset after 100 steps')
    parser.add_argument('-k', '--checkpoint', action='store_true', help='generate checkpoint when process killed')
    # TODO could make this actually accept hte value instead of prompt for it
    parser.add_argument('-l', '--load_checkpoint', action='store_true', help='load from checkpoint')
    parser.add_argument('-p', '--progress', action='store_true', help='print progress of agent')
    
    args = parser.parse_args()
    
    if not args.model_path:
        print("No model given,", end=' ')

    if args.random_agent:
        print("using random agent")
    else:
        print("using human control")

    if args.area:
        print("Printing game area from pyboy")

    if args.debug:
        print("Printing pyboy's debug info")

    if args.coords:
        print("Printing coordinates of samus")

    if args.observation:
        print("Printing observation space")

    if args.shape:
        print("Printing shape of observations")
    if args.reward:
        print("Printing rewards")
    if args.reset:
        print("quick resetting enabled")
    if args.progress:
        print("printing agent progress")

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

        screen_obs = obs[SCREEN_OBS] 
        h,w,dims = screen_obs.shape
        image_2d = screen_obs.reshape(h, w)

        # Create a PIL Image object
        image = Image.fromarray(image_2d)
        print(f"IMAGE DIMS: {w, h}")

        image.save("obs.png")
        return

    if args.model_path:
        print("This isn't working for 'pufferlib' style models yet!!!!", file=sys.stderr)
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

    # for printing long things
    import numpy as np
    np.set_printoptions(linewidth=np.inf)
    
    # fields for tracking data for printing
    net_reward = 0
    truncated = False
    step = 0

    env.step(0)

    if args.load_checkpoint:
        if not os.path.exists(env.CHECKPOINT_DIR):
            raise ModuleNotFoundError("Couldn't find checkpoints folder!")
        state_files = [os.path.join(env.CHECKPOINT_DIR, name) for name in
                       os.listdir(env.CHECKPOINT_DIR)]
        print(f"FILES: {state_files}")
        # load files 
        base_name = input("give a base name (just the filename, without .state or .set: ")
        base_path = os.path.join(env.CHECKPOINT_DIR, base_name)
        print(f"Loading {base_path}")
        env.load_checkpoint(base_path)




    try:
        while not truncated:
            step += 1
            # action, _states = model.predict(obs)
            action, _states = action_getter(obs)
            obs, rewards, dones, truncated, info = env.step(action)
            if args.area:
                print(f"AREA: \n{env.game_area()}")
            if args.debug:
                print(env.pyboy.game_wrapper)
            if args.coords:
                print(f"Coords: {env.getAllCoordData()}\tLen explored: {len(env.explored)}")
            if args.observation:
                print(f"\n{obs}")
            if args.shape:
                print(f"{[(k, type(o)) for k,o in obs.items()]}")
            if args.reward:
                net_reward += rewards
                print(f"Reward: {rewards}\tNet: {net_reward}")
            if args.reset and step >= 100:
                print("RESETTING")
                step = 0
                env.reset()
            if args.progress:
                print(f"Progress: {env.progress}")
    except KeyboardInterrupt:
        if args.checkpoint:
            idx = int(input("Enter index: "))
            name = input("Enter name: ")
            base_filename = f"{idx}_{name}"

            fullpath = os.path.join(env.CHECKPOINT_DIR, base_filename)

            print(f"fullpath: {fullpath}")
            env.save_checkpoint(fullpath)

            print(f"saved {base_filename} files! (.set and .state)")


    print("Truncated!")


if __name__ == "__main__":
    main()
